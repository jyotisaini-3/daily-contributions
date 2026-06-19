# GPU Roofline Analysis — CUDA SGEMM Kernels

A complete roofline model analysis pipeline for CUDA matrix-multiplication kernels, from naive baseline to tensor-core optimized, profiled with **NVIDIA Nsight Compute (`ncu`)** and visualised in a Jupyter notebook.

## What is the roofline model?

The roofline model plots **arithmetic intensity** (FLOPs / byte of memory traffic) against **attainable performance** (TFLOP/s). Every kernel falls into one of two regimes:

```
     Peak FP32 perf ─────────────────────────────
                                              /
  Performance                               /   Compute-bound
  (TFLOP/s)              Memory-bound      /
                         ________________ /
                        /
    Peak BW × AI  ─────

                  0    AI = FLOPs / Bytes
```

## Kernels analysed

| Kernel | Techniques | AI (FP32) | Perf |
|--------|-----------|-----------|------|
| `sgemm_naive` | global-mem only | ~0.5 FLOPs/B | ~2% peak |
| `sgemm_tiled` | shared-mem tiling (32×32) | ~8 FLOPs/B | ~18% peak |
| `sgemm_vectorized` | 128-bit LDS + register blocking | ~32 FLOPs/B | ~55% peak |
| `sgemm_tensor_core` | `wmma::fragment` F16 tensor cores | ~64 FLOPs/B | ~82% peak |

## Quick start

```bash
git clone https://github.com/jyotisaini-3/gpu-roofline-analysis
cd gpu-roofline-analysis
pip install -r requirements.txt

# Build kernels
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=86  # or 90 for H100
cmake --build build -j$(nproc)

# Profile with Nsight Compute
bash scripts/run_ncu.sh

# Visualise
jupyter notebook notebooks/roofline_analysis.ipynb
```

## Repository layout

```
gpu-roofline-analysis/
├── kernels/
│   ├── sgemm_naive.cu          # Baseline — global memory only
│   ├── sgemm_tiled.cu          # Shared-memory tiling
│   ├── sgemm_vectorized.cu     # float4 loads + register blocking
│   └── sgemm_tensor_core.cu   # WMMA F16 tensor core
├── scripts/
│   └── run_ncu.sh              # Nsight Compute profiling script
├── notebooks/
│   └── roofline_analysis.ipynb # Roofline plot + analysis
├── data/                       # ncu CSV outputs (committed)
└── CMakeLists.txt
```

## References
- [Roofline: An Insightful Visual Performance Model](https://doi.org/10.1145/1498765.1498785) — Williams et al., 2009
- [Nsight Compute CLI Guide](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
- [CUDA Best Practices — Memory Optimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
