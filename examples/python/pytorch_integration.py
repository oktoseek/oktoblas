"""
OktoBLAS - PyTorch Integration Example
======================================

This example demonstrates how to use OktoBLAS with PyTorch.

Installation:
    pip install oktoblas torch

"""

import oktoblas as ob
import numpy as np
import time

def main():
    print("=" * 60)
    print("OktoBLAS + PyTorch Integration")
    print("=" * 60)
    
    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
    except ImportError:
        print("PyTorch not installed. Install with: pip install torch")
        return
    
    # Benchmark comparison
    print("\n" + "-" * 60)
    print("FP16 GEMM Benchmark (2048x2048)")
    print("-" * 60)
    
    size = 2048
    iterations = 100
    
    # Prepare data
    A_np = np.random.randn(size, size).astype(np.float16)
    B_np = np.random.randn(size, size).astype(np.float16)
    
    # PyTorch benchmark
    if torch.cuda.is_available():
        A_torch = torch.from_numpy(A_np).cuda()
        B_torch = torch.from_numpy(B_np).cuda()
        
        # Warmup
        for _ in range(10):
            _ = torch.matmul(A_torch, B_torch)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            C_torch = torch.matmul(A_torch, B_torch)
        torch.cuda.synchronize()
        pytorch_time = (time.perf_counter() - start) / iterations * 1000  # ms
        
        flops = 2 * size * size * size
        pytorch_tflops = flops / (pytorch_time / 1000) / 1e12
        print(f"PyTorch:  {pytorch_time:.3f} ms ({pytorch_tflops:.1f} TFLOPS)")
    
    # OktoBLAS benchmark
    # Warmup
    for _ in range(10):
        _ = ob.matmul_fp16(A_np, B_np)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        C_ob = ob.matmul_fp16(A_np, B_np)
    oktoblas_time = (time.perf_counter() - start) / iterations * 1000  # ms
    
    oktoblas_tflops = flops / (oktoblas_time / 1000) / 1e12
    print(f"OktoBLAS: {oktoblas_time:.3f} ms ({oktoblas_tflops:.1f} TFLOPS)")
    
    if torch.cuda.is_available():
        ratio = oktoblas_tflops / pytorch_tflops * 100
        print(f"\nRatio: {ratio:.1f}% of PyTorch")
        if ratio > 100:
            print("🏆 OktoBLAS WINS!")
    
    # Verify correctness
    print("\n" + "-" * 60)
    print("Correctness Check")
    print("-" * 60)
    
    # Small matrix for verification
    A_small = np.random.randn(64, 64).astype(np.float32)
    B_small = np.random.randn(64, 64).astype(np.float32)
    
    C_numpy = np.matmul(A_small, B_small)
    C_oktoblas = ob.matmul(A_small, B_small)
    
    diff = np.abs(C_numpy - C_oktoblas).max()
    print(f"Max difference from NumPy: {diff:.6f}")
    print(f"Correctness: {'✅ PASS' if diff < 0.01 else '❌ FAIL'}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()




