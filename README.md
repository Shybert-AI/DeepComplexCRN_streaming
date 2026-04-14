# DeepComplexCRN

基于深度复数循环网络的语音增强系统，支持整段推理和流式推理。

> **🔴 商业许可 · 源码授权 · 模型定制 · 流式集成 🔴**  
> 本项目源码暂不公开，提供商业合作。如需获取源码、定制训练或集成流式推理，请联系：**854197093@qq.com**（请注明“语音增强合作”）。

---
问：为什么不直接开源？
答：流式推理实现复杂，且项目投入了大量研发资源，目前仅通过商业合作共享技术成果。

## ⚠️ 重要声明

- 本项目基于以下两个开源项目开发，并增加了**原创的流式推理**功能：
  - [huyanxin/DeepComplexCRN](https://github.com/huyanxin/DeepComplexCRN) (Apache-2.0 License)
  - [Chriszhangmw/DeepComplexCRN](https://github.com/Chriszhangmw/DeepComplexCRN) (Apache-2.0 License)
- 已按照 Apache-2.0 许可证的要求，保留原始版权声明和许可证副本。
- 我对本项目中**新增的流式推理功能代码**拥有完整的著作权。
- 本项目作为衍生作品，**整体以闭源形式提供商业授权**。

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
- [商业合作与服务](#商业合作与服务)

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

---

## 商业合作与服务

本项目**仅提供 Python 推理/训练代码**，源码不开放。如需以下服务，欢迎洽谈：

- 🔐 **源码授权** – 获取完整可商用源码
- 🧠 **模型定制** – 根据您的噪声场景和数据定制训练
- 📡 **流式集成** – 帮助将模型部署到实时系统（低延迟、滑动窗口优化）
- ☁️ **云端部署** – 提供 API 封装或容器化方案

**联系方式：**  
📧 **854197093@qq.com**（请注明“语音增强合作”）  
💬 也可通过 GitHub Issue 简述需求，我们会尽快回复。

> 注：所有合作均建立在商业付费基础上，感谢理解。

