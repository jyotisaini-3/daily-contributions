# Portfolio Scaffolds — Setup Guide

Three GPU/aerospace project scaffolds ready to push as standalone GitHub repos.

## Step 1: Clone this repo locally

```bash
git clone https://github.com/jyotisaini-3/daily-contributions
cd daily-contributions
git checkout claude/sharp-lovelace-1fdfvw
```

## Step 2: Fork these repos on GitHub (click Fork on each)

| Repo | Why |
|------|-----|
| https://github.com/NVIDIA/cutlass | Backs "CUTLASS custom GEMM" CV claim |
| https://github.com/Dao-AILab/flash-attention | Warp-level attention, Hopper kernels |
| https://github.com/NVIDIA/TensorRT-LLM | Production LLM inference |
| https://github.com/NVIDIA/cccl | CUDA Core Compute Libraries |
| https://github.com/AICL-Lab/cuda-kernel-academy | SGEMM → tensor core learning path |
| https://github.com/AICL-Lab/mini-inference-engine | GEMM benchmarks + inference |
| https://github.com/AICL-Lab/modern-ai-kernels | FlashAttention + normalization |
| https://github.com/Ray-Rose/shapedcloud | SCvx powered-descent (flight-grade Rust) |
| https://github.com/root3315/unisat | CubeSat/CanSat flight software |
| https://github.com/meta-pytorch/tritonparse | Triton kernel compiler tracer |

## Step 3: Create and push the three scaffold repos

Run this script from the `scaffolds/` directory:

```bash
cd scaffolds
bash push_scaffolds.sh
```

Or manually:

```bash
# 1. hopper-warp-gemm
cd scaffolds/hopper-warp-gemm
git init && git add . && git commit -m "feat: Hopper SM90 warp-specialized GEMM kernel"
gh repo create jyotisaini-3/hopper-warp-gemm --public --source=. --push
# or: git remote add origin https://github.com/jyotisaini-3/hopper-warp-gemm && git push -u origin main

# 2. gpu-roofline-analysis  
cd ../gpu-roofline-analysis
git init && git add . && git commit -m "feat: roofline analysis pipeline for CUDA SGEMM kernels"
gh repo create jyotisaini-3/gpu-roofline-analysis --public --source=. --push

# 3. powered-descent-gpu
cd ../powered-descent-gpu
git init && git add . && git commit -m "feat: GPU-accelerated batch SCvx powered-descent trajectory solver"
gh repo create jyotisaini-3/powered-descent-gpu --public --source=. --push
```

## Step 4: Pin repos to your GitHub profile

1. Go to https://github.com/jyotisaini-3
2. Click **Customize your pins**
3. Pin: `hopper-warp-gemm`, `gpu-roofline-analysis`, `powered-descent-gpu`
4. Also pin your forks of `NVIDIA/cutlass` and `Dao-AILab/flash-attention`

## Step 5: What to build next (closes CV gap fully)

| Task | Time | Impact |
|------|------|--------|
| Run the bench in `hopper-warp-gemm` on an H100 and add real numbers to README | 2h | Verifiable benchmark |
| Run `run_ncu.sh`, fill real ncu CSV data into notebook, export `roofline_h100.png` | 3h | Closes roofline claim |
| Add `wgmma.mma_async` PTX to the GEMM kernel (replace fma loop) | 4h | True SM90 tensor-core path |
| Run `monte_carlo --batch 1024` on GPU, commit `results.csv` + trajectory plot | 2h | Shows GNC/aerospace depth |
