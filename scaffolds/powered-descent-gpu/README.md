# Powered Descent GPU — Batch SCvx Trajectory Solver

GPU-accelerated **successive convexification (SCvx)** for powered-descent guidance, targeting rocket landing scenarios similar to SpaceX Falcon 9 and Starship.

Runs **thousands of Monte-Carlo trajectory rollouts in parallel** on a single GPU, enabling real-time uncertainty quantification and onboard candidate-solution generation.

## Problem statement

Find minimum-fuel trajectory `u(t)` for a rocket descending from apogee to landing:

```
minimise   ∫₀ᵀ ‖u(t)‖ dt
subject to  ẋ = f(x, u)          (6-DoF dynamics)
            ‖u‖ ≤ T_max           (max thrust)
            ‖u‖ ≥ T_min           (min throttle — engine-off forbidden)
            x(0) = x₀, x(T) = 0  (boundary conditions)
            γ(t) ≥ γ_min          (glideslope cone)
```

SCvx linearises around a reference trajectory each iteration, solving a sequence of convex QPs until convergence (typically 5–15 iterations).

## GPU parallelism strategy

```
Host (CPU)                     Device (GPU)
──────────                     ─────────────
Generate N initial guesses  →  N concurrent SCvx solvers (one per CUDA stream)
                               Each solver:
                                 ├─ Dynamics propagation (RK4 kernel)
                                 ├─ Jacobian computation (finite diff, parallel)
                                 ├─ QP construction (batched GEMM via cuBLAS)
                                 └─ QP solve (cuSOLVER batched LU)
Select best solution        ←  Parallel reduction (argmin fuel)
```

## Repository layout

```
powered-descent-gpu/
├── src/
│   ├── trajectory_solver.cu    # Main SCvx iteration loop
│   ├── dynamics.cuh            # 6-DoF rocket dynamics + RK4 integrator
│   ├── qp_solver.cuh           # cuSOLVER batched LU wrapper
│   └── monte_carlo.cu          # Batch launcher + argmin reduction
├── python/
│   ├── visualize.py            # 3-D trajectory plots (matplotlib)
│   └── generate_ics.py         # Monte-Carlo initial condition sampler
├── tests/
│   └── test_dynamics.cu        # Unit tests for dynamics kernel
└── CMakeLists.txt
```

## Quick start

```bash
git clone https://github.com/jyotisaini-3/powered-descent-gpu
cd powered-descent-gpu
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j$(nproc)

# Run 1024 parallel trajectories
./build/monte_carlo --batch 1024 --output results.csv

# Visualise
pip install matplotlib numpy
python python/visualize.py results.csv
```

## References
- Malyuta et al., [*Convex Optimization for Trajectory Generation*](https://arxiv.org/abs/2106.09125), IEEE CSM 2022
- Açıkmeşe & Ploen, [*Convex Programming Approach to Powered Descent*](https://doi.org/10.2514/1.27553), JGCD 2007 (original lossless convexification)
- SpaceX Falcon 9 landing: powered-descent problem at human-scale inertia/thrust ratios
