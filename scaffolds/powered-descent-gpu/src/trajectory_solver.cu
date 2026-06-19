/**
 * SCvx trajectory solver — one instance per CUDA thread block.
 * Each block independently solves its own powered-descent problem
 * with a different initial condition (Monte-Carlo batch).
 *
 * SCvx iteration:
 *   1. Propagate reference trajectory (RK4)
 *   2. Linearise dynamics around reference (finite-difference Jacobians)
 *   3. Discretise → state-transition matrices (Φ, B_bar)
 *   4. Form convex QP: min ‖u‖₁  s.t.  Ax = b, C u ≤ d
 *   5. Solve QP (projected gradient / ADMM for GPU-friendliness)
 *   6. Check trust-region convergence; accept or shrink
 *   Repeat until ‖Δu‖ < tol or max_iter reached.
 */

#include "dynamics.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

constexpr int N_NODES   = 50;    // temporal discretisation nodes
constexpr int MAX_ITER  = 20;    // max SCvx outer iterations
constexpr float TOL_SOL = 1e-4f; // convergence tolerance

struct Trajectory {
  State    states[N_NODES];
  Control  controls[N_NODES - 1];
  float    fuel_used;  // kg
  int      iters;      // SCvx iterations to convergence
  bool     converged;
};

// Simple thrust guess: constant hover + small tilt toward target
__device__
void initial_guess(Trajectory& traj, const State& x0, const RocketParams& p) {
  float T_hover = x0.mass * fabsf(p.g[2]);  // |m*g|
  for (int k = 0; k < N_NODES; ++k) {
    // Linear interpolation of position
    float t = (float)k / (N_NODES - 1);
    traj.states[k] = x0;
    traj.states[k].r[0] = x0.r[0] * (1.f - t);
    traj.states[k].r[1] = x0.r[1] * (1.f - t);
    traj.states[k].r[2] = x0.r[2] * (1.f - t);
    traj.states[k].v[0] = -x0.r[0] / (float)N_NODES;
    traj.states[k].v[1] = -x0.r[1] / (float)N_NODES;
    traj.states[k].v[2] = x0.v[2] + (0.f - x0.v[2]) * t;
    traj.states[k].mass = x0.mass - (x0.mass - p.mass_dry) * t;
  }
  for (int k = 0; k < N_NODES - 1; ++k) {
    traj.controls[k].T[0] = -p.g[0] * x0.mass * 0.1f;
    traj.controls[k].T[1] = -p.g[1] * x0.mass * 0.1f;
    traj.controls[k].T[2] = T_hover;
  }
}

// SCvx convergence check: RMS change in control
__device__
float control_rms_change(const Control* u_new, const Control* u_ref) {
  float sum = 0.f;
  for (int k = 0; k < N_NODES - 1; ++k) {
    for (int i = 0; i < 3; ++i) {
      float d = u_new[k].T[i] - u_ref[k].T[i];
      sum += d * d;
    }
  }
  return sqrtf(sum / (3.f * (N_NODES - 1)));
}

// One SCvx solve (simplified — full implementation hooks cuSOLVER batched LU)
__global__
void scvx_batch_kernel(
    const State*    initial_conditions,   // [batch_size]
    const RocketParams* params,
    Trajectory*     results,              // [batch_size]
    int             batch_size,
    float           tf)                   // final time [s]
{
  int idx = blockIdx.x;
  if (idx >= batch_size) return;

  const State&       x0 = initial_conditions[idx];
  const RocketParams& p  = *params;
  Trajectory&        traj = results[idx];

  // Initialise with heuristic guess
  initial_guess(traj, x0, p);

  float dt = tf / (N_NODES - 1);
  Control u_prev[N_NODES - 1];

  for (int iter = 0; iter < MAX_ITER; ++iter) {
    // Save previous controls for convergence check
    for (int k = 0; k < N_NODES-1; ++k) u_prev[k] = traj.controls[k];

    // Step 1: propagate reference trajectory with RK4
    State x = x0;
    for (int k = 0; k < N_NODES - 1; ++k) {
      traj.states[k] = x;
      rk4_step(x, traj.controls[k], p, dt);
    }
    traj.states[N_NODES-1] = x;

    // Step 2–4 (placeholder): in full implementation, form & solve QP
    // using cuSOLVER batched LU per block. Here we do a simple gradient step
    // on fuel cost ∫‖u‖dt to illustrate the structure.
    for (int k = 0; k < N_NODES - 1; ++k) {
      float T_norm = sqrtf(
        traj.controls[k].T[0]*traj.controls[k].T[0] +
        traj.controls[k].T[1]*traj.controls[k].T[1] +
        traj.controls[k].T[2]*traj.controls[k].T[2] + 1e-6f);
      // Gradient of ‖u‖ w.r.t. u = u/‖u‖; descent step
      float lr = 10.f;
      for (int i = 0; i < 3; ++i)
        traj.controls[k].T[i] -= lr * traj.controls[k].T[i] / T_norm;
      // Project onto thrust constraints
      float T_new_norm = sqrtf(
        traj.controls[k].T[0]*traj.controls[k].T[0] +
        traj.controls[k].T[1]*traj.controls[k].T[1] +
        traj.controls[k].T[2]*traj.controls[k].T[2] + 1e-6f);
      if (T_new_norm < p.T_min) {
        float s = p.T_min / T_new_norm;
        for (int i=0;i<3;i++) traj.controls[k].T[i] *= s;
      } else if (T_new_norm > p.T_max) {
        float s = p.T_max / T_new_norm;
        for (int i=0;i<3;i++) traj.controls[k].T[i] *= s;
      }
    }

    // Convergence check
    float rms = control_rms_change(traj.controls, u_prev);
    if (rms < TOL_SOL) {
      traj.iters     = iter + 1;
      traj.converged = true;
      break;
    }
    if (iter == MAX_ITER - 1) {
      traj.iters     = MAX_ITER;
      traj.converged = false;
    }
  }

  // Compute fuel used
  traj.fuel_used = x0.mass - traj.states[N_NODES-1].mass;
}

// Host launcher
void solve_batch(
    const State* d_ics, const RocketParams* d_params,
    Trajectory* d_results, int batch_size, float tf)
{
  scvx_batch_kernel<<<batch_size, 1>>>(d_ics, d_params, d_results, batch_size, tf);
  cudaDeviceSynchronize();
}
