"""
OktoBLAS Benchmark Chart Generator
==================================
Generates comparison charts with REAL benchmark data

Run: python generate_benchmark_chart.py
"""

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# REAL BENCHMARK DATA (December 2025)
# ============================================================

# Quick Test (100 examples)
quick_test = {
    'modes': ['PyTorch FP32\n(Baseline)', 'OktoBLAS FP16\n(Tensor Cores)'],
    'time': [1.97, 1.07],
    'speed': [50.8, 93.7],
    'speedup': [1.0, 1.85]
}

# Speed Test (Matrix Operations)
speed_test = {
    'modes': ['PyTorch FP32\n(Baseline)', 'OktoBLAS FP16\n(Tensor Cores)', 'OktoBLAS TURBO\n(Fused)'],
    'time_ms': [9.73, 4.86, 3.63],
    'speedup': [1.0, 2.0, 2.68]
}

# GEMM Kernels
gemm_data = {
    'operations': ['FP16 GEMM\n1024', 'FP16 GEMM\n2048', 'Fused\nAttention'],
    'pytorch': [23.3, 34.6, 0.28],
    'oktoblas': [29.1, 35.1, 0.96]
}

# ============================================================
# CHART GENERATION
# ============================================================

plt.style.use('dark_background')
fig = plt.figure(figsize=(14, 10))

fig.suptitle('OktoBLAS Performance Benchmark\nby OktoSeek', 
             fontsize=18, fontweight='bold', color='#00ff88', y=0.98)

# Colors
pytorch_color = '#ff6b6b'
oktoblas_color = '#4ecdc4'
turbo_color = '#ffd93d'

# ============================================================
# Chart 1: Training Speed (Top Left)
# ============================================================
ax1 = fig.add_subplot(2, 2, 1)
x = np.arange(len(quick_test['modes']))
colors = [pytorch_color, oktoblas_color]
bars = ax1.bar(x, quick_test['speed'], color=colors, alpha=0.85, edgecolor='white', linewidth=2)

ax1.set_ylabel('Speed (examples/sec)', fontsize=12, fontweight='bold')
ax1.set_title('📊 Training Speed (100 examples)\n(Higher is Better)', fontsize=13, fontweight='bold', pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(quick_test['modes'], fontsize=10)
ax1.set_ylim(0, 120)
ax1.grid(True, alpha=0.2, axis='y')

for bar, val, speedup in zip(bars, quick_test['speed'], quick_test['speedup']):
    label = f'{val:.1f} ex/s'
    if speedup > 1:
        label += f'\n(+{(speedup-1)*100:.0f}%)'
    ax1.annotate(label, xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='white')

# ============================================================
# Chart 2: Matrix Ops Speed (Top Right)
# ============================================================
ax2 = fig.add_subplot(2, 2, 2)
x = np.arange(len(speed_test['modes']))
colors = [pytorch_color, oktoblas_color, turbo_color]
bars = ax2.bar(x, speed_test['time_ms'], color=colors, alpha=0.85, edgecolor='white', linewidth=2)

ax2.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
ax2.set_title('⚡ Matrix Ops Speed\n(Lower is Better)', fontsize=13, fontweight='bold', pad=10)
ax2.set_xticks(x)
ax2.set_xticklabels(speed_test['modes'], fontsize=9)
ax2.set_ylim(0, 12)
ax2.grid(True, alpha=0.2, axis='y')

for bar, val, speedup in zip(bars, speed_test['time_ms'], speed_test['speedup']):
    label = f'{val:.2f}ms'
    if speedup > 1:
        label += f'\n({speedup:.2f}x)'
    ax2.annotate(label, xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='white')

# ============================================================
# Chart 3: GEMM Performance (Bottom Left)
# ============================================================
ax3 = fig.add_subplot(2, 2, 3)
x_gemm = np.arange(len(gemm_data['operations']))
width = 0.35

bars1 = ax3.bar(x_gemm - width/2, gemm_data['pytorch'], width, label='PyTorch', 
                color=pytorch_color, alpha=0.85, edgecolor='white', linewidth=1.5)
bars2 = ax3.bar(x_gemm + width/2, gemm_data['oktoblas'], width, label='OktoBLAS', 
                color=oktoblas_color, alpha=0.85, edgecolor='white', linewidth=1.5)

ax3.set_ylabel('TFLOPS', fontsize=12, fontweight='bold')
ax3.set_title('🚀 GEMM Kernel Performance\n(Higher is Better)', fontsize=13, fontweight='bold', pad=10)
ax3.set_xticks(x_gemm)
ax3.set_xticklabels(gemm_data['operations'], fontsize=9)
ax3.legend(loc='upper left', fontsize=10)
ax3.grid(True, alpha=0.2, axis='y')

for i, (p, o) in enumerate(zip(gemm_data['pytorch'], gemm_data['oktoblas'])):
    speedup = (o - p) / p * 100
    if speedup > 0:
        ax3.annotate(f'+{speedup:.0f}%', 
                    xy=(x_gemm[i] + width/2, o),
                    ha='center', va='bottom', fontsize=9, color='#00ff88', fontweight='bold')

# ============================================================
# Chart 4: Summary Box (Bottom Right)
# ============================================================
ax4 = fig.add_subplot(2, 2, 4)
ax4.axis('off')

summary_text = """
╔══════════════════════════════════════════════════╗
║           OktoBLAS BENCHMARK SUMMARY             ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  🚀 TRAINING SPEED (100 examples)                ║
║  ────────────────────────────────────────────    ║
║  PyTorch FP32:      50.8 ex/s  (baseline)        ║
║  OktoBLAS FP16:     93.7 ex/s  (+85% faster)     ║
║                                                  ║
║  ⚡ MATRIX OPS SPEED                             ║
║  ────────────────────────────────────────────    ║
║  PyTorch FP32:      9.73 ms    (baseline)        ║
║  OktoBLAS FP16:     4.86 ms    (2.00x faster)    ║
║  OktoBLAS TURBO:    3.63 ms    (2.68x faster)    ║
║                                                  ║
║  🔥 SPEEDUP SUMMARY                              ║
║  ────────────────────────────────────────────    ║
║  • Training:        +85% faster                  ║
║  • Matrix Ops:      +100% faster                 ║
║  • TURBO Mode:      +168% faster                 ║
║  • FP16 GEMM 1024:  +25% TFLOPS                  ║
║  • Fused Attention: +243% TFLOPS                 ║
║                                                  ║
╚══════════════════════════════════════════════════╝
"""

ax4.text(0.5, 0.5, summary_text, transform=ax4.transAxes, fontsize=9,
         verticalalignment='center', horizontalalignment='center',
         fontfamily='monospace', color='white',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e', 
                  edgecolor='#4ecdc4', linewidth=2))

plt.tight_layout(rect=[0, 0.02, 1, 0.95])

# Save
plt.savefig('benchmark_comparison.png', dpi=150, facecolor='#0d0d0d', 
            edgecolor='none', bbox_inches='tight', pad_inches=0.3)
print("✅ Saved: benchmark_comparison.png")

print("\n📊 Chart generated with REAL benchmark data!")
print("   Training: 1.85x faster")
print("   Matrix Ops: 2.68x faster")
