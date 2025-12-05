<p align="center">
  <img src="./assets/okto_logo2.png" alt="OktoScript Banner" width="50%" />
</p>

<h1 align="center">OktoBLAS</h1>

<p align="center">
  <strong>The Independent BLAS Engine Powering OktoEngine</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/oktoblas/"><img src="https://img.shields.io/pypi/v/oktoblas?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://www.oktoseek.com/"><img src="https://img.shields.io/badge/OktoSeek-Official-orange" alt="OktoSeek"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-Proprietary-red" alt="License"></a>
</p>

---

## What is OktoBLAS?

**OktoBLAS** is a proprietary, high-performance **Basic Linear Algebra Subprograms (BLAS)** engine developed by **OktoSeek**. It is the core computational backbone of **OktoEngine**, our native AI training and inference platform.

Unlike wrapper libraries, OktoBLAS is built **entirely from scratch** using Rust and hand-tuned CUDA PTX assembly — with **zero dependency on NVIDIA cuBLAS**.

### 🎯 Key Highlights

| | |
|---|---|
| **100% Independent** | No cuBLAS, no external BLAS dependencies |
| **Hand-Tuned PTX** | Every kernel optimized at assembly level |
| **Tensor Core Native** | Built for NVIDIA Tensor Cores (WMMA) |
| **Production Ready** | Powers OktoEngine in production |
| **Python Available** | Also released as standalone Python package |

---

## 🏆 Performance

All benchmarks performed on **NVIDIA RTX 4070 Laptop GPU** using CUDA Events (zero overhead).

### FP16 GEMM — Tensor Cores

| Matrix Size | OktoBLAS | PyTorch/cuBLAS | Performance |
|:-----------:|:--------:|:--------------:|:-----------:|
| 1024×1024 | **29.1 TFLOPS** | 23.3 TFLOPS | **125%** ✓ |
| 2048×2048 | **35.1 TFLOPS** | 34.6 TFLOPS | **101%** ✓ |
| 3072×3072 | 36.2 TFLOPS | 38.6 TFLOPS | 94% |
| 4096×4096 | 36.5 TFLOPS | 38.9 TFLOPS | 94% |

### Fused Attention — Single Kernel

| Configuration | OktoBLAS | PyTorch | Speedup |
|:-------------:|:--------:|:-------:|:-------:|
| Batch 4, Seq 256, Dim 64 | **0.96 TFLOPS** | 0.28 TFLOPS | **3.4x** |
| Batch 4, Seq 512, Dim 64 | **1.22 TFLOPS** | 0.93 TFLOPS | **1.3x** |

### Training Throughput

| Method | Speed | vs Baseline |
|:------:|:-----:|:-----------:|
| PyTorch (cuBLAS) | 158.9 ex/s | — |
| **OktoEngine + OktoBLAS** | **~430 ex/s** | **2.7x faster** |

---

## 🔧 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        OktoSeek AI                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│   │ OktoScript  │───▶│ OktoEngine  │───▶│  OktoStudio │    │
│   │   (DSL)     │    │  (Runtime)  │    │    (IDE)    │    │
│   └─────────────┘    └──────┬──────┘    └─────────────┘    │
│                             │                               │
│                             ▼                               │
│   ┌─────────────────────────────────────────────────────┐  │
│   │                    OktoBLAS                         │  │
│   │         Proprietary BLAS Engine (Rust + PTX)        │  │
│   │                                                     │  │
│   │  ┌─────────┐  ┌─────────┐  ┌─────────────────────┐ │  │
│   │  │ FP16    │  │ FP32    │  │  Fused Operations   │ │  │
│   │  │ GEMM    │  │ GEMM    │  │  (Attention, etc.)  │ │  │
│   │  └─────────┘  └─────────┘  └─────────────────────┘ │  │
│   └─────────────────────────────────────────────────────┘  │
│                             │                               │
│                             ▼                               │
│   ┌─────────────────────────────────────────────────────┐  │
│   │              NVIDIA GPU (Tensor Cores)              │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Python Package

OktoBLAS is also available as a **standalone Python package** for developers who want to leverage our BLAS engine outside of OktoEngine.

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
C = ob.matmul_fp16(A, B)  # 35+ TFLOPS

# Fused Attention (3x faster)
Q = np.random.randn(4, 512, 64).astype(np.float32)
K = np.random.randn(4, 512, 64).astype(np.float32)
V = np.random.randn(4, 512, 64).astype(np.float32)
output = ob.attention(Q, K, V)

# Library info
ob.info()
```

### Output

```
============================================================
OktoBLAS by OktoSeek
High-Performance BLAS Library
============================================================
Version: 1.0.1
License: Proprietary (c) 2025 OktoSeek AI
Status: Native extension loaded
Backend: CUDA PTX (Tensor Cores)

Features:
  - FP16/FP32 GEMM with Tensor Cores
  - Fused Attention kernel
  - 100% Independent (no cuBLAS)

https://www.oktoseek.com
============================================================
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

## 🧪 OktoScript Integration

Within **OktoEngine**, OktoBLAS is configured through **OktoScript** v1.3+:

```okt
# okto_version: "1.3"

PROJECT "my-ai-model"

# Enable OktoBLAS as BLAS backend
BLAS {
    backend: "oktoblas"
    precision: "fp16"
    streams: 4
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
}
```

```bash
okto train -f train.okt
```

---

## 🌐 OktoSeek Ecosystem

OktoBLAS is a core component of the **OktoSeek AI** platform:

| Component | Description | Status |
|:---------:|:------------|:------:|
| **OktoScript** | AI programming language | [Available](https://github.com/oktoseek/oktoscript) |
| **OktoEngine** | Native AI training runtime | Production |
| **OktoBLAS** | High-performance BLAS engine | [PyPI](https://pypi.org/project/oktoblas/) |
| **OkTensor** | GPU tensor library | Production |
| **OktoStudio** | AI development IDE | Coming Soon |

---

## 📁 Examples

- [`examples/python/`](./examples/python/) — Python usage examples
- [`examples/oktoblas-benchmark/`](./examples/oktoblas-benchmark/) — Complete OktoScript training example

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
  <strong>OktoBLAS</strong> — The BLAS engine built for AI
</p>

<p align="center">
  Made with precision by <a href="https://www.oktoseek.com"><strong>OktoSeek AI</strong></a>
</p>


