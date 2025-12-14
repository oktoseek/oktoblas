<p align="center">
  <img src="assets/oktoblas-logo.png" alt="OktoBLAS" width="400"/>
</p>

<h1 align="center">OktoBLAS</h1>

<p align="center">
  <strong>🏆 Matches PyTorch Performance • 6,234 ex/s Training • Up to +21% in GEMM • Fused Attention 3.8x Faster 🏆</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/oktoblas/"><img src="https://img.shields.io/pypi/v/oktoblas?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://www.oktoseek.com/"><img src="https://img.shields.io/badge/OktoSeek-Official-orange" alt="OktoSeek"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-Proprietary-red" alt="License"></a>
  <a href="https://doi.org/10.5281/zenodo.17932053"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.17932053.svg" alt="DOI"></a>
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

### 🚀 OktoTensor (v1.0.9) - Training Performance

**NEW**: GPU-resident tensor class eliminates conversion overhead!

| Dataset | OktoTensor | Traditional | Improvement |
|:-------:|:----------:|:-----------:|:-----------:|
| **ShareGPT** (vocab=9033) | **6,234 ex/s** | ~500 ex/s | **12.5x** 🔥🔥🔥 |
| **OpenOrca** (vocab=32000) | **2,406 ex/s** | ~200 ex/s | **12x** 🔥🔥 |

> 📊 Benchmarks on **NVIDIA RTX 4070 Laptop GPU**

---

## 🚀 Why OktoTensor?

**OktoTensor** is the GPU-resident execution model that makes OktoBLAS truly competitive in Python.

### The Problem

Traditional Python workflows introduce significant overhead when data is not kept close to the compute units. This creates a gap between kernel performance (fast) and real-world usage (slow).

### The Solution

**OktoTensor** introduces a GPU-resident execution model that keeps tensors persistent on the device, dramatically improving throughput and benchmark stability.

### Results

✅ **Much higher examples/sec** (6,234 vs ~500)  
✅ **Predictable performance** (consistent across runs)  
✅ **Better scaling** with larger models and vocabularies  
✅ **Matches PyTorch** in real-world training workloads

### What This Means

**OktoBLAS kernels are fast** (31.7+ TFLOPS internally).  
**OktoTensor makes this usable in Python** (6,234 ex/s).  
**OktoEngine is where everything comes together** (native Rust runtime).

> 💡 **Key Insight**: OktoTensor is not just an optimization — it's a fundamental shift to GPU-resident computing that eliminates the Python binding bottleneck.

---

## What is OktoBLAS?

**OktoBLAS** is a proprietary, high-performance **BLAS** engine developed by **OktoSeek**. It is the core computational backbone of **OktoEngine**, our native AI training platform.

Built **100% from scratch** with **zero dependency on NVIDIA cuBLAS**.

### 🎯 Key Highlights

| | |
|---|---|
| **100% Independent** | No cuBLAS dependency |
| **Matches PyTorch** | **6,234 ex/s** training, up to **+21%** in GEMM 🔥 |
| **OktoTensor** | GPU-resident runtime eliminates Python overhead |
| **Fused Attention** | Up to **3.8x faster** 🔥 |
| **Production Ready** | Powers OktoEngine |

---

> 📖 **[Energy Savings & Environmental Impact →](docs/ENTERPRISE_SAVINGS.md)**
> 
> 📖 **[OktoSeek Research Mission →](docs/RESEARCH_MISSION.md)**

---

## 🔧 Architecture

OktoBLAS is the computational core of the OktoSeek platform:

```
OktoScript → OktoEngine → OktoBLAS → OktoTensor → GPU (Tensor Cores)
```

### Components

| Component | Description | Status |
|:---------:|:------------|:------:|
| **OktoBLAS** | Core BLAS kernels (31.7+ TFLOPS) | ✅ Production |
| **OktoTensor** | GPU-resident tensor class (v1.0.9+) | ✅ Production |
| **OktoBLASModel** | Persistent weights on GPU | ✅ Production |
| **Fused Ops** | Attention, Linear+GELU, RMSNorm | ✅ Production |

---

## 📦 Python Package

OktoBLAS is available as a **standalone Python package**.

**Current Version**: **v1.0.9** (with OktoTensor support)

### Installation

```bash
pip install oktoblas
```

### Version Information

- **v1.0.9** (Current): OktoTensor GPU-resident runtime (6,234 ex/s)
- **v1.0.8**: OktoBLASModel with persistent weights
- **v1.0.7**: Real CUDA kernels (Python bindings)
- **v1.0.6**: Initial PyPI release

**Python Support**: 3.9, 3.10, 3.11, 3.12, 3.13  
**Platform**: Windows x64, Linux x64 (more coming soon)

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

# 🚀 NEW: OktoTensor (GPU-resident, 12.5x faster!)
# Eliminates NumPy↔CUDA conversion overhead
x = ob.OktoTensor(np_array, device="cuda")  # Upload once
w = ob.OktoTensor(weights, device="cuda")   # Upload once
result = x.matmul(w)  # Operations stay on GPU!
# Result: 6,234 ex/s (vs ~500 ex/s traditional)

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

# 🚀 OktoTensor (v1.0.9+) - GPU-resident tensors
x = ob.OktoTensor(np_array, device="cuda")  # Upload once, stays on GPU
w = ob.OktoTensor(weights, device="cuda")   # Upload once, stays on GPU
result = x.matmul(w)  # Zero conversion overhead!
result_numpy = result.cpu()  # Explicit conversion when needed

# Utilities
ob.info()                 # Library information
ob.is_cuda_available()    # Check GPU availability
ob.get_device_info()      # GPU details
ob.benchmark(op, size)    # Run benchmarks
```

---

## 🚀 OktoTensor Usage

### Quick Start

```python
import oktoblas as ob
import numpy as np

# Upload weights once (stays on GPU)
w1 = ob.OktoTensor(weight1, device="cuda")
w2 = ob.OktoTensor(weight2, device="cuda")

# Training loop - operations stay on GPU!
for batch in batches:
    x = ob.OktoTensor(batch, device="cuda")  # Only input converts
    h = x.matmul(w1)  # No conversion!
    y = h.matmul(w2)  # No conversion!
    loss = y.cpu()  # Explicit conversion only when needed
```

### Performance Results

| Method | ShareGPT (ex/s) | OpenOrca (ex/s) | Improvement |
|:------:|:---------------:|:---------------:|:-----------:|
| Traditional | ~500 | ~200 | Baseline |
| **OktoTensor** | **6,234** | **2,406** | **12.5x** 🔥🔥🔥 |

See [`examples/python/benchmark_oktotensor_simple.py`](./examples/python/benchmark_oktotensor_simple.py) for complete benchmark.

---

## 🧪 Benchmarks: OktoBLAS vs PyTorch

### Training Performance (Python)

Tested on **NVIDIA RTX 4070 Laptop GPU** with real datasets:

| Method | ShareGPT (ex/s) | OpenOrca (ex/s) | Notes |
|:------:|:---------------:|:---------------:|:-----:|
| **PyTorch** | ~6,000-7,000 | ~2,000-3,000 | Baseline (cuBLAS) |
| **OktoBLAS (OktoTensor)** | **6,234** | **2,406** | **Matches PyTorch!** 🔥 |

**Configuration**: 2-layer MLP, batch_size=32, seq_len=64

### Run Benchmarks

```bash
# OktoTensor benchmark
python benchmark_oktotensor_simple.py

# PyTorch benchmark (run separately to avoid CUDA context conflicts)
python benchmark_pytorch.py
```

> 📖 **[Complete Benchmark Methodology →](docs/benchmarks/BENCHMARK_METHODOLOGY.md)**

---

## 🚀 Maximum Performance Guide

### For Python Users

**Use OktoTensor for best performance** (v1.0.9+):

```python
import oktoblas as ob
import numpy as np

# ✅ BEST: Use OktoTensor (GPU-resident)
w1 = ob.OktoTensor(weight1, device="cuda")  # Upload once
w2 = ob.OktoTensor(weight2, device="cuda")  # Upload once

for batch in batches:
    x = ob.OktoTensor(batch, device="cuda")
    result = x.matmul(w1).matmul(w2)  # 6,234+ ex/s!
```

**Result**: **6,234 ex/s** (matches PyTorch!)

### For OktoEngine Users

1. **Enable cuDNN benchmark**
2. **Use FP16 and Tensor Cores**
3. **Enable automatic mixed precision (AMP)**
4. **Use OktoTensor in OktoScript** (coming soon)

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
| **OktoBLAS** | High-Performance BLAS — **Matches PyTorch, up to +21% in GEMM** | [PyPI](https://pypi.org/project/oktoblas/) |
| **OktoTensor** | GPU-resident Python runtime | Production (v1.0.9+) |
| **OktoStudio** | AI Development IDE | Coming Soon |

---

## 📁 Examples

- [`examples/python/`](./examples/python/) — Python usage examples
- [`docs/ENTERPRISE_SAVINGS.md`](./docs/ENTERPRISE_SAVINGS.md) — Energy & Cost Savings

---

## 📚 Zenodo Archive

This repository is archived on Zenodo as scientific evidence and prior-art record:

**Zenodo DOI**: [10.5281/zenodo.17932053](https://doi.org/10.5281/zenodo.17932053)

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
  <strong>🏆 OktoBLAS — Independent BLAS that Matches PyTorch Performance 🏆</strong>
</p>

<p align="center">
  Made with precision by <a href="https://www.oktoseek.com"><strong>OktoSeek AI</strong></a>
</p>
