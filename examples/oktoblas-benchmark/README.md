# OktoBLAS Benchmark

Complete training example using OktoBLAS with OktoScript.

## Structure

```
oktoblas-benchmark/
├── scripts/
│   └── train.okt        # Training script (v1.3)
├── dataset/
│   ├── train.jsonl      # Training data (1000 examples)
│   └── val.jsonl        # Validation data (100 examples)
└── README.md
```

## Quick Start

```bash
# Run training with OktoEngine CLI
cd oktoblas-benchmark
okto train -f scripts/train.okt
```

> OktoEngine CLI available at [oktoseek.com](https://www.oktoseek.com)

## OktoBLAS Blocks Used

This example demonstrates the new OktoScript v1.3 blocks:

### `BLAS` Block
```okt
BLAS {
    backend: "oktoblas"   # Use OktoBLAS instead of cuBLAS
    precision: "fp16"     # FP16 for Tensor Cores
    streams: 4            # 4 CUDA streams for parallelism
}
```

### `ACCELERATE` Block
```okt
ACCELERATE {
    gemm: "oktoblas"      # OktoBLAS for matrix multiplication
    attention: "oktoblas" # OktoBLAS for attention
    fused_ops: true       # Enable fused operations
}
```

### `TENSOR_CORES` Block
```okt
TENSOR_CORES {
    enabled: true         # Enable Tensor Cores
    precision: "fp16"     # FP16 precision
}
```

## Expected Results

| Metric | Value |
|--------|-------|
| Training Speed | ~430 examples/s |
| Speedup vs PyTorch | 2.7x |
| Final Loss | < 0.5 |
| Training Time | ~5 min |

## Dataset

The dataset is a subset of OpenOrca formatted as chat conversations:

```json
{
  "question": "What is machine learning?",
  "response": "Machine learning is..."
}
```

## Export

After training, the model is exported to:
- `export/oktoblas-benchmark/model.safetensors`
- `export/oktoblas-benchmark/model.okm`

---

Part of [OktoBLAS](https://github.com/oktoseek/oktoblas) • [OktoScript](https://github.com/oktoseek/oktoscript)

