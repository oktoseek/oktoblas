<p align="center">
  <img src="assets/oktoblas-logo.png" alt="OktoBLAS" width="400"/>
</p>

<h1 align="center">OktoBLAS</h1>

<p align="center">
  <strong>🏆 Beats PyTorch by up to 21% • Fused Attention 3.8x Faster 🏆</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/oktoblas/"><img src="https://img.shields.io/pypi/v/oktoblas?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://www.oktoseek.com/"><img src="https://img.shields.io/badge/OktoSeek-Official-orange" alt="OktoSeek"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-Proprietary-red" alt="License"></a>
</p>

---

## 🔥 Performance

### FP16 GEMM

| Matrix Size | OktoBLAS | PyTorch | Result |
|:-----------:|:--------:|:-------:|:------:|
| **1024×1024** | **33.9 TFLOPS** | 30.0 TFLOPS | **+13.1%** 🔥 |
| **2048×2048** | **40.6 TFLOPS** | 33.7 TFLOPS | **+20.6%** 🔥🔥 |
| **4096×4096** | **42.1 TFLOPS** | 40.1 TFLOPS | **+5.0%** ✅ |

### Fused Attention

| Configuration | OktoBLAS | PyTorch | Speedup |
|:-------------:|:--------:|:-------:|:-------:|
| B4 S256 D64 | **1.06 TFLOPS** | 0.28 TFLOPS | **3.8x** 🔥 |
| B4 S512 D64 | **1.20 TFLOPS** | 0.93 TFLOPS | **1.3x** ✅ |
| B8 S256 D64 | **1.17 TFLOPS** | 0.55 TFLOPS | **2.1x** ✅ |

> 📊 Benchmarks on **NVIDIA RTX 4070 Laptop GPU**

---

## What is OktoBLAS?

**OktoBLAS** is a proprietary, high-performance **BLAS** engine developed by **OktoSeek**. It is the core computational backbone of **OktoEngine**, our native AI training platform.

Built **100% from scratch** with **zero dependency on NVIDIA cuBLAS**.

### 🎯 Key Highlights

| | |
|---|---|
| **100% Independent** | No cuBLAS dependency |
| **Beats PyTorch** | Up to **+21% faster** 🔥 |
| **Fused Attention** | Up to **3.8x faster** 🔥 |
| **Production Ready** | Powers OktoEngine |

---

## 🌱 Energy Savings & Environmental Impact

**OktoBLAS helps save energy and reduce CO₂ emissions worldwide.**

By running AI workloads **12% faster**, OktoBLAS reduces GPU power consumption significantly:

| Scale | GPUs | Annual Energy Saved | CO₂ Reduced | Cost Saved |
|:-----:|:----:|:-------------------:|:-----------:|:----------:|
| Startup | 1-4 | 400-1,700 kWh | 160-680 kg | $60-$260 |
| SMB | 8-32 | 2,300-12,000 kWh | 0.9-4.8 ton | $350-$1,800 |
| Enterprise | 64-256 | 27,000-107,000 kWh | 11-43 ton | $4,000-$16,000 |
| **Hyperscaler** | **1024+** | **680,000+ kWh** | **272+ ton** | **$102,000+** |

### 🌍 Impact for Humanity

Every GPU-hour saved means:
- **Less electricity consumed** from power plants
- **Less CO₂ emissions** into the atmosphere
- **Lower costs** for AI research and development
- **More accessible AI** for everyone

> 📖 **[Full Enterprise Savings Analysis →](docs/ENTERPRISE_SAVINGS.md)**

This is why **OktoSeek** created OktoBLAS — not just for performance, but for a **sustainable AI future**.

---

## 🔬 OktoSeek Research Mission

One of **OktoSeek's** primary research areas is developing **new mathematical techniques and optimization methods** that reduce AI training time **without compromising model quality**.

### Why This Matters for Humanity

```
┌─────────────────────────────────────────────────────────────────────┐
│                  THE PROBLEM WE'RE SOLVING                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Today, training a large AI model costs:                           │
│                                                                     │
│   💰 $100,000 to $10,000,000+ in compute                            │
│   ⚡ 1,000,000+ kWh of electricity                                   │
│   🕐 Weeks to months of GPU time                                    │
│   🌍 Tons of CO₂ emissions                                          │
│                                                                     │
│   This means only big companies can create AI.                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### OktoSeek's Solution

By making training **faster and cheaper**, we enable:

| Benefit | Impact |
|:-------:|:------:|
| **🧑‍🔬 Researchers** | More experiments in less time |
| **🏫 Universities** | Train models on limited budgets |
| **🚀 Startups** | Compete with big tech companies |
| **🌍 Developing Nations** | Access to AI creation, not just consumption |
| **🌱 Planet Earth** | Less energy = less carbon emissions |

### The Vision

> *"We believe AI should be accessible to everyone — not just those who can afford million-dollar GPU clusters. By making training 12%+ faster with the same hardware, we're democratizing AI creation and building a more sustainable future."*
>
> — **OktoSeek Research Team**

**Faster training means:**
- ✅ More people can create AI
- ✅ More innovations in less time
- ✅ Lower barriers to entry
- ✅ Smaller environmental footprint

---

## 🔧 Architecture

OktoBLAS is the computational core of the OktoSeek platform:

```
OktoScript → OktoEngine → OktoBLAS → GPU (Tensor Cores)
```

---

## 📦 Python Package

OktoBLAS is available as a **standalone Python package**.

### Installation

```bash
pip install oktoblas
```

### Quick Start

```python
import oktoblas as ob
import numpy as np

# FP16 Matrix Multiplication (Tensor Cores)
A = np.random.randn(2048, 2048).astype(np.float16)
B = np.random.randn(2048, 2048).astype(np.float16)
C = ob.matmul_fp16(A, B)  # 40+ TFLOPS

# Fused Attention (3x faster)
Q = np.random.randn(4, 512, 64).astype(np.float32)
K = np.random.randn(4, 512, 64).astype(np.float32)
V = np.random.randn(4, 512, 64).astype(np.float32)
output = ob.attention(Q, K, V)

# Library info
ob.info()
```

### API Reference

```python
# GEMM Operations
ob.matmul(A, B)           # FP32 matrix multiplication
ob.matmul_fp16(A, B)      # FP16 with Tensor Cores

# Fused Operations
ob.attention(Q, K, V)     # Fused Q×K^T×V attention

# Utilities
ob.info()                 # Library information
ob.is_cuda_available()    # Check GPU availability
ob.get_device_info()      # GPU details
ob.benchmark(op, size)    # Run benchmarks
```

---

## 🚀 Maximum Performance Guide

For best results with OktoBLAS:

1. **Enable cuDNN benchmark**
2. **Use FP16 and Tensor Cores**
3. **Enable automatic mixed precision (AMP)**

---

## 🧪 OktoScript Integration

Within **OktoEngine**, OktoBLAS is configured through **OktoScript** v1.3+:

```okt
# okto_version: "1.3"

PROJECT "my-ai-model"

# Enable OktoBLAS as BLAS backend
BLAS {
    backend: "oktoblas"
    precision: "fp16"
}

# Accelerate operations with OktoBLAS
ACCELERATE {
    gemm: "oktoblas"
    attention: "oktoblas"
    fused_ops: true
}

# Enable Tensor Cores
TENSOR_CORES {
    enabled: true
    precision: "fp16"
}

MODEL {
    base: "gpt2"
    device: "cuda"
}

TRAIN {
    epochs: 3
    batch_size: 16
    mixed_precision: true
}

# Performance optimization
OPTIMIZE {
    cudnn_benchmark: true
    tf32: true
}
```

### Run Training

```bash
# Standard training
okto train -f train.okt

# With verbose performance logging
okto train -f train.okt --verbose --show-tflops
```

### Expected Output

```
[OktoBLAS] Device: NVIDIA RTX 4070
[OktoBLAS] FP16 GEMM: 40.6 TFLOPS (beats PyTorch!)

Step   100 | Loss: 2.45 | Speed: 520 ex/s | TFLOPS: 40.2
Step   200 | Loss: 1.89 | Speed: 518 ex/s | TFLOPS: 39.9
...
Training complete! Average: 515 ex/s
```

---

## 🌐 OktoSeek Ecosystem

OktoBLAS is a core component of the **OktoSeek AI** platform — a complete ecosystem for building, training, and deploying AI models with maximum efficiency.

| Component | Description | Status |
|:---------:|:------------|:------:|
| **OktoScript** | The AI Programming Language — DSL for model training | ⭐ [Popular](https://github.com/oktoseek/oktoscript) |
| **OktoEngine** | Native AI Training Runtime — powered by OktoBLAS | Production |
| **OktoBLAS** | High-Performance BLAS — **Beats PyTorch by 21%!** | [PyPI](https://pypi.org/project/oktoblas/) |
| **OkTensor** | GPU Tensor Library | Production |
| **OktoStudio** | AI Development IDE | Coming Soon |

---

## 📁 Examples

- [`examples/python/`](./examples/python/) — Python usage examples
- [`docs/ENTERPRISE_SAVINGS.md`](./docs/ENTERPRISE_SAVINGS.md) — Energy & Cost Savings

---

## 📜 License

**OktoBLAS Binary License** — Proprietary

Free for personal and commercial use. Redistribution and modification of binaries prohibited.

Copyright © 2025 **OktoSeek AI**. All Rights Reserved.

See [LICENSE](./LICENSE) for full terms.

---

## 🔗 Links

| | |
|---|---|
| **Website** | [oktoseek.com](https://www.oktoseek.com) |
| **PyPI** | [pypi.org/project/oktoblas](https://pypi.org/project/oktoblas/) |
| **GitHub** | [github.com/oktoseek](https://github.com/oktoseek) |
| **Twitter** | [@oktoseek](https://x.com/oktoseek) |

---

<p align="center">
  <strong>🏆 OktoBLAS — The First Independent BLAS to Beat PyTorch 🏆</strong>
</p>

<p align="center">
  Made with precision by <a href="https://www.oktoseek.com"><strong>OktoSeek AI</strong></a>
</p>
