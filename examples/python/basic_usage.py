"""
OktoBLAS - Basic Usage Example
==============================

This example demonstrates basic OktoBLAS operations.

Installation:
    pip install oktoblas

"""

import oktoblas as ob
import numpy as np

def main():
    print("=" * 60)
    print("OktoBLAS Basic Usage Example")
    print("=" * 60)
    
    # Show library info
    print("\n1. Library Info:")
    ob.info()
    
    # FP32 Matrix Multiplication
    print("\n2. FP32 GEMM:")
    A = np.random.randn(1024, 1024).astype(np.float32)
    B = np.random.randn(1024, 1024).astype(np.float32)
    C = ob.matmul(A, B)
    print(f"   A: {A.shape} @ B: {B.shape} = C: {C.shape}")
    print(f"   Result sample: {C[0, 0]:.4f}")
    
    # FP16 Matrix Multiplication (Tensor Cores)
    print("\n3. FP16 GEMM (Tensor Cores):")
    A16 = np.random.randn(1024, 1024).astype(np.float16)
    B16 = np.random.randn(1024, 1024).astype(np.float16)
    C16 = ob.matmul_fp16(A16, B16)
    print(f"   A: {A16.shape} @ B: {B16.shape} = C: {C16.shape}")
    print(f"   Result sample: {C16[0, 0]:.4f}")
    
    # Fused Attention
    print("\n4. Fused Attention:")
    batch, seq_len, head_dim = 4, 256, 64
    Q = np.random.randn(batch, seq_len, head_dim).astype(np.float32)
    K = np.random.randn(batch, seq_len, head_dim).astype(np.float32)
    V = np.random.randn(batch, seq_len, head_dim).astype(np.float32)
    output = ob.attention(Q, K, V)
    print(f"   Q: {Q.shape}, K: {K.shape}, V: {V.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Result sample: {output[0, 0, 0]:.4f}")
    
    # Check CUDA availability
    print("\n5. CUDA Status:")
    print(f"   CUDA Available: {ob.is_cuda_available()}")
    
    # Benchmark
    print("\n6. Benchmark (FP16 GEMM 2048x2048):")
    try:
        results = ob.benchmark("gemm_fp16", size=2048, iterations=50)
        print(f"   OktoBLAS: {results['oktoblas_tflops']:.1f} TFLOPS")
        if 'pytorch_tflops' in results:
            print(f"   PyTorch:  {results['pytorch_tflops']:.1f} TFLOPS")
            print(f"   Ratio:    {results['ratio']:.1f}%")
    except Exception as e:
        print(f"   Benchmark skipped: {e}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()

