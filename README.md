# DeepComplexCRN

基于深度复数循环网络的语音增强系统，支持整段推理和流式推理。<font color="red">暂源码不开放，欢迎合作交流</font>

## 目录

- [快速开始](#快速开始)
- [项目概述](#项目概述)
- [环境配置](#环境配置)
- [数据集准备](#数据集准备)
- [模型训练](#模型训练)
- [模型推理](#模型推理)
- [效果可视化](#效果可视化)
- [模型详解](#模型详解)
- [参数配置](#参数配置)
- [文件结构](#文件结构)

---

## 快速开始

```bash
# 1. 安装依赖
pip install torch librosa soundfile numpy

# 2. 准备数据集并生成列表
python genration_list.py

# 3. 训练模型
python train.py

# 4. 推理测试
python predict_forward_streaming.py
```

---

## 项目概述

DeepComplexCRN (Deep Complex Convolutional Recurrent Network) 是一种用于语音增强的深度学习模型，通过在复数频域上进行操作来有效分离噪声和语音。

**主要特性:**
- 支持整段推理和流式推理
- 复数域处理，保留相位信息
- 支持多种掩码模式 (E/C/R/CL)
- 基于SI-SNR loss训练

---

## 环境配置

### 依赖包

```
torch >= 1.0
librosa
soundfile
numpy
```

### 安装

```bash
pip install torch librosa soundfile numpy
```

---

## 数据集准备

### 1. 下载数据集并合成噪声数据

THCHS-30 中文语音数据集: http://www.openslr.org/18/

使用 `utils/synthesizer.py` 生成噪声和干净数据为原始语音添加噪声

### 2. 数据集目录结构

```
dataset/
├── THCHS-30/
│   └── data_synthesized/
│       ├── train/
│       │   └── 0dB/
│       │       ├── clean/      # 干净语音
│       │       └── noisy/      # 含噪语音 (*cafe.wav, *car.wav, *white.wav)
│       └── test/
│           └── 0dB/
│               ├── clean/
│               └── noisy/
└── dsn/
    ├── train.lst      # 训练列表
    ├── dev.lst        # 验证列表
    └── test_0.lst     # 测试列表
```

### 3. 生成列表文件

运行脚本自动生成:

```bash
python genration_list.py
```

**列表格式:** 每行 `<含噪路径> <干净路径>`

示例:
```
dataset/THCHS-30/data_synthesized/train/0dB/noisy/A2_0_cafe.wav dataset/THCHS-30/data_synthesized/train/0dB/clean/A2_0.wav
dataset/THCHS-30/data_synthesized/train/0dB/noisy/A2_0_car.wav dataset/THCHS-30/data_synthesized/train/0dB/clean/A2_0.wav
```

---

## 模型训练

### 命令

```bash
python train.py
```

### 配置参数 (config.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| batch_size | 16 | 批大小 |
| lr | 0.001 | 学习率 |
| max_epoch | 40 | 最大轮数 |
| tr_list | dataset/dsn/train.lst | 训练列表 |
| dev_list | dataset/dsn/dev.lst | 验证列表 |
| checkpoint_root | asr_res_model/dns/dccrn | 模型保存路径 |

### 训练输出

```
epoch 1 SI-SNR -10.2341
epoch 2 SI-SNR -8.5623
...
save model at epoch 3 ...
```

---

## 模型推理

### 使用方法

修改 `predict_forward_streaming.py` 中的测试路径:

```python
wave_test = 'assert/D4_750_cafe.wav'  # 替换为你的音频路径
```

运行:

```bash
python predict_forward_streaming.py
```

### 输出结果

- `denoised_wav_*.wav` - 整段推理结果
- `denoised_streaming_*.wav` - 流式推理结果

### 整段推理 vs 流式推理

| 方式 | 特点 | 适用场景 |
|------|------|----------|
| 整段推理 | 一次性处理完整音频 | 离线处理 |
| 流式推理 | 滑动窗口，实时输出 | 实时降噪 |

---

## 效果可视化

修改 `visualize_denoise.py` 中的路径:

```python
clean_wav = 'assert/D4_750.wav'
noisy_wav = 'assert/D4_750_cafe.wav'
denoised_wav = 'assert/denoised_wav_D4_750_cafe.wav'
streaming_wav = 'assert/denoised_streaming_D4_750_cafe.wav'
```

运行:

```bash
python visualize_denoise.py
```

效果保存为 `assert/comparison.png`

---

## 模型详解

### 网络架构

```
输入语音 → STFT → 复数谱 → 编码器 → LSTM×2 → 解码器 → ISTFT → 增强语音
```

### 核心组件

| 组件 | 说明 |
|------|------|
| STFT/ISTFT | 短时傅里叶变换，512点FFT，400帧长，100帧移 |
| 编码器 | 6层复数卷积，通道 [2, 32, 64, 128, 256, 256, 256] |
| 时序建模 | 双层LSTM，hidden_size=256 |
| 解码器 | 6层复数反卷积 |

### 掩码模式

| 模式 | 名称 | 说明 |
|------|------|------|
| E | Erasure | 幅值掩码 |
| R | Real | 实部/虚部分别掩码 |
| C | Complex | 复数掩码 |
| CL | Complex+LSTM | 完整模式 (默认) |

---

## 参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| win_len | 400 | 帧长 |
| win_inc | 100 | 帧移 |
| fft_len | 512 | FFT长度 |
| rnn_units | 256 | RNN隐藏层大小 |
| rnn_layers | 2 | RNN层数 |
| sr | 16000 | 采样率 |

---

## 文件结构

| 文件 | 说明 |
|------|------|
| `config.py` | 全局配置 |
| `genration_list.py` | 生成训练列表 |
| `train.py` | 训练脚本 |
| `predict_forward_streaming.py` | 推理脚本 |
| `visualize_denoise.py` | 效果可视化 |
| `models/DCCRN.py` | 模型定义 |
| `models/loss.py` | SI-SNR损失 |
| `models/conv_stft.py` | STFT/ISTFT |
| `models/complexnn.py` | 复数卷积模块 |
| `dataloader/dataloader.py` | 数据加载 |
| `utils/synthesizer.py` | 噪声合成 |
