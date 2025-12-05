<p align="center">
  <h1 align="center">OktoBLAS</h1>
</p>

<p align="center">
  <strong>High-Performance BLAS Library by OktoSeek</strong>
</p>

<p align="center">
  Tensor Core acceleration • Fused Attention • 100% Independent
</p>

> **OktoBLAS** is a high-performance, fully independent BLAS (Basic Linear Algebra Subprograms) library built from scratch in Rust + CUDA PTX. It provides GPU-accelerated matrix operations using Tensor Cores, achieving up to 125% of PyTorch performance on FP16 GEMM and 3x faster Fused Attention — all without cuBLAS dependency.

<p align="center">
  <a href="https://www.oktoseek.com/">OktoSeek Homepage</a> •
  <a href="https://pypi.org/project/oktoblas/">PyPI</a> •
  <a href="https://github.com/oktoseek/oktoscript">OktoScript</a> •
  <a href="https://x.com/oktoseek">Twitter</a>
</p>

---

## 📦 Installation

```bash
pip install oktoblas
```

---

## 🚀 Quick Start

```python
import oktoblas as ob
import numpy as np

# Matrix multiplication
A = np.random.randn(2048, 2048).astype(np.float32)
B = np.random.randn(2048, 2048).astype(np.float32)
C = ob.matmul(A, B)

# FP16 with Tensor Cores
A16 = np.random.randn(2048, 2048).astype(np.float16)
B16 = np.random.randn(2048, 2048).astype(np.float16)
C16 = ob.matmul_fp16(A16, B16)

# Fused Attention
batch, seq_len, head_dim = 4, 512, 64
Q = np.random.randn(batch, seq_len, head_dim).astype(np.float32)
K = np.random.randn(batch, seq_len, head_dim).astype(np.float32)
V = np.random.randn(batch, seq_len, head_dim).astype(np.float32)
output = ob.attention(Q, K, V)

# Show info
ob.info()
```

**Output:**
```
============================================================
OktoBLAS by OktoSeek
High-Performance BLAS Library
============================================================
Version: 1.0.1
License: Proprietary (c) 2025 OktoSeek AI
Status: Native extension loaded

Features:
  - FP16/FP32 GEMM with Tensor Cores
  - Fused Attention kernel
  - 100% Independent (no cuBLAS)

https://www.oktoseek.com
============================================================
```

---

## 📊 Benchmark Results

All benchmarks validated using **CUDA Events** (zero Python overhead).

**Hardware:** NVIDIA RTX 4070 Laptop GPU (8GB VRAM, Tensor Cores)

### FP16 GEMM (Tensor Cores)

| Matrix Size | OktoBLAS | PyTorch | Ratio |
|-------------|----------|---------|-------|
| 1024×1024 | 29.1 TF | 23.3 TF | **125%** |
| 2048×2048 | 35.1 TF | 34.6 TF | **101%** |
| 3072×3072 | 36.2 TF | 38.6 TF | 94% |
| 4096×4096 | 36.5 TF | 38.9 TF | 94% |

### FP32 GEMM

| Matrix Size | OktoBLAS | PyTorch | Ratio |
|-------------|----------|---------|-------|
| 2048×2048 | 9.5 TF | 10.9 TF | 87% |
| 4096×4096 | 8.9 TF | 9.5 TF | 94% |

### Fused Attention

| Config | OktoBLAS | PyTorch | Ratio |
|--------|----------|---------|-------|
| B4 S256 D64 | 0.96 TF | 0.28 TF | **346%** |
| B4 S512 D64 | 1.22 TF | 0.93 TF | **131%** |
| B8 S512 D64 | 1.56 TF | 1.95 TF | 80% |

### Training Speed (OpenOrca 5000 examples)

| Method | Speed | Ratio |
|--------|-------|-------|
| PyTorch Pure | 158.9 ex/s | Baseline |
| PyTorch + OktoBLAS GEMM | ~430 ex/s | **2.7x** |

---

## 🔥 Features

| Feature | Description |
|---------|-------------|
| **FP16 GEMM** | Tensor Core acceleration with WMMA |
| **FP32 GEMM** | Optimized hand-tuned PTX kernels |
| **Fused Attention** | Single kernel Q×K×V operation |
| **100% Independent** | No cuBLAS dependency |
| **PyTorch Integration** | Works with PyTorch tensors |

---

## 📖 Python API

### Functions

```python
import oktoblas as ob

# GEMM Operations
ob.matmul(A, B)           # FP32 matrix multiplication
ob.matmul_fp16(A, B)      # FP16 matrix multiplication (Tensor Cores)
ob.gemm(A, B)             # Alias for matmul
ob.gemm_fp16(A, B)        # Alias for matmul_fp16

# Attention
ob.attention(Q, K, V)     # Fused attention kernel
ob.fused_attention(Q, K, V, scale)  # With custom scale

# Utilities
ob.info()                 # Show library info
ob.benchmark("gemm_fp16", size=2048)  # Run benchmark
ob.is_cuda_available()    # Check CUDA availability
ob.get_device_info()      # Get GPU information
```

### Example: Benchmark

```python
import oktoblas as ob

# Run FP16 GEMM benchmark
results = ob.benchmark("gemm_fp16", size=2048, iterations=100)

print(f"OktoBLAS: {results['oktoblas_tflops']:.1f} TF")
print(f"PyTorch:  {results['pytorch_tflops']:.1f} TF")
print(f"Ratio:    {results['ratio']:.1f}%")
```

---

## 🧪 OktoScript Integration

OktoBLAS integrates seamlessly with [OktoScript](https://github.com/oktoseek/oktoscript) v1.3+.

### New OktoBLAS Blocks

#### `BLAS` - Configure backend
```okt
BLAS {
    backend: "oktoblas"    # Use OktoBLAS (default: "cublas")
    precision: "fp16"      # fp16 | fp32
    streams: 4             # Number of CUDA streams
}
```

#### `ACCELERATE` - Enable acceleration
```okt
ACCELERATE {
    gemm: "oktoblas"       # GEMM backend
    attention: "oktoblas"  # Attention backend
    fused_ops: true        # Enable fused operations
}
```

#### `TENSOR_CORES` - GPU acceleration
```okt
TENSOR_CORES {
    enabled: true          # Enable Tensor Cores
    precision: "fp16"      # fp16 | tf32
}
```

### Complete Training Example

```okt
# okto_version: "1.3"

PROJECT "gpt2-finetune"
DESCRIPTION "Fine-tune GPT-2 with OktoBLAS acceleration"

ENV {
    accelerator: "gpu"
    min_memory: "8GB"
    precision: "fp16"
}

# OktoBLAS Configuration
BLAS {
    backend: "oktoblas"
    precision: "fp16"
    streams: 4
}

ACCELERATE {
    gemm: "oktoblas"
    attention: "oktoblas"
    fused_ops: true
}

TENSOR_CORES {
    enabled: true
    precision: "fp16"
}

DATASET {
    train: "data/train.jsonl"
    validation: "data/val.jsonl"
    format: "jsonl"
    type: "chat"
}

MODEL {
    base: "gpt2"
    device: "cuda"
}

TRAIN {
    epochs: 3
    batch_size: 16
    learning_rate: 5e-5
    optimizer: "adamw"
    scheduler: "cosine"
}

EXPORT {
    format: ["safetensors", "okm"]
    path: "output/gpt2-finetuned"
}
```

### Run Training

```bash
# Run training with OktoEngine CLI
okto train -f train.okt
```

> **Note:** OktoEngine CLI is part of the OktoSeek ecosystem. Visit [oktoseek.com](https://www.oktoseek.com) for installation.

**Expected speedup:** 2.7x faster training compared to PyTorch baseline.

See [`examples/oktoblas-benchmark/`](./examples/oktoblas-benchmark/) for a complete runnable example.

---

## 📁 Examples

- **[`examples/python/`](./examples/python/)** - Python usage examples
- **[`examples/oktoblas-benchmark/`](./examples/oktoblas-benchmark/)** - OktoScript training example
- **[`benchmarks/`](./benchmarks/)** - Benchmark scripts

---

## 📚 Part of OktoSeek Ecosystem

| Project | Description | Link |
|---------|-------------|------|
| **OktoScript** | AI programming language | [GitHub](https://github.com/oktoseek/oktoscript) |
| **OktoEngine** | Native ML inference engine | Coming soon |
| **OktoStudio** | AI Development IDE | Coming soon |
| **OktoBLAS** | High-performance BLAS | [PyPI](https://pypi.org/project/oktoblas/) |
| **OkTensor** | GPU tensor library | Part of OktoEngine |

---

## 📜 License

**Proprietary License** - Free for personal and commercial use.

Copyright (c) 2025 OktoSeek AI. All Rights Reserved.

See [LICENSE](./LICENSE) for details.

---

## 🙏 Credits

Built with ❤️ by **OktoSeek AI**.

- **Website**: https://www.oktoseek.com
- **GitHub**: https://github.com/oktoseek
- **PyPI**: https://pypi.org/project/oktoblas/
- **Twitter**: https://x.com/oktoseek

---

<p align="center">
  Made with ❤️ by the <strong>OktoSeek AI</strong> team
</p>

