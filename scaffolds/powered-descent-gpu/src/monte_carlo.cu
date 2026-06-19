#include "dynamics.cuh"
#include "trajectory_solver.cu"  // include definition for single TU
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_CUDA(x) do { cudaError_t e=(x); if(e){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} } while(0)

// Parallel argmin: find trajectory with minimum fuel
__global__ void argmin_fuel_kernel(const Trajectory* trajs, int N, int* best_idx, float* best_fuel) {
  __shared__ float s_fuel[256];
  __shared__ int   s_idx[256];
  int tid = threadIdx.x;
  int i   = blockIdx.x * blockDim.x + tid;

  s_fuel[tid] = (i < N && trajs[i].converged) ? trajs[i].fuel_used : 1e30f;
  s_idx[tid]  = i;
  __syncthreads();

  // Reduction
  for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
    if (tid < stride && s_fuel[tid + stride] < s_fuel[tid]) {
      s_fuel[tid] = s_fuel[tid + stride];
      s_idx[tid]  = s_idx[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) { *best_fuel = s_fuel[0]; *best_idx = s_idx[0]; }
}

void sample_initial_conditions(State* ics, int N, const RocketParams& p) {
  srand(42);
  for (int i = 0; i < N; ++i) {
    // Position: 500–2000m altitude, ±200m lateral spread
    ics[i].r[0] = ((float)rand()/RAND_MAX - 0.5f) * 400.f;
    ics[i].r[1] = ((float)rand()/RAND_MAX - 0.5f) * 400.f;
    ics[i].r[2] = 500.f + (float)rand()/RAND_MAX * 1500.f;
    // Velocity: mostly downward, small horizontal
    ics[i].v[0] = ((float)rand()/RAND_MAX - 0.5f) * 10.f;
    ics[i].v[1] = ((float)rand()/RAND_MAX - 0.5f) * 10.f;
    ics[i].v[2] = -50.f - (float)rand()/RAND_MAX * 50.f;
    // Attitude: near identity quaternion
    ics[i].q[0] = 1.f; ics[i].q[1] = ics[i].q[2] = ics[i].q[3] = 0.f;
    // Angular velocity: small perturbation
    ics[i].w[0] = ((float)rand()/RAND_MAX-0.5f)*0.05f;
    ics[i].w[1] = ((float)rand()/RAND_MAX-0.5f)*0.05f;
    ics[i].w[2] = 0.f;
    ics[i].mass = p.mass_wet;
  }
}

int main(int argc, char** argv) {
  int BATCH = 256;
  for (int i = 1; i < argc-1; ++i)
    if (strcmp(argv[i], "--batch") == 0) BATCH = atoi(argv[i+1]);

  // Falcon-9-like parameters (scaled)
  RocketParams p;
  p.g[0]=0; p.g[1]=0; p.g[2]=-9.806f;
  p.Isp=311.f; p.g0=9.806f;
  p.T_min=180e3f; p.T_max=845e3f;
  p.mass_dry=22200.f; p.mass_wet=25000.f;
  p.r_T[0]=0; p.r_T[1]=0; p.r_T[2]=-20.f;
  p.J[0]=p.J[1]=1.5e7f; p.J[2]=1e5f;

  State* h_ics = new State[BATCH];
  sample_initial_conditions(h_ics, BATCH, p);

  State*       d_ics;    CHECK_CUDA(cudaMalloc(&d_ics, BATCH*sizeof(State)));
  RocketParams* d_p;    CHECK_CUDA(cudaMalloc(&d_p,   sizeof(RocketParams)));
  Trajectory*  d_res;   CHECK_CUDA(cudaMalloc(&d_res, BATCH*sizeof(Trajectory)));
  int*   d_best_idx;    CHECK_CUDA(cudaMalloc(&d_best_idx, sizeof(int)));
  float* d_best_fuel;   CHECK_CUDA(cudaMalloc(&d_best_fuel, sizeof(float)));

  CHECK_CUDA(cudaMemcpy(d_ics, h_ics, BATCH*sizeof(State), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_p,   &p,    sizeof(RocketParams), cudaMemcpyHostToDevice));

  cudaEvent_t t0, t1;
  CHECK_CUDA(cudaEventCreate(&t0)); CHECK_CUDA(cudaEventCreate(&t1));
  CHECK_CUDA(cudaEventRecord(t0));

  solve_batch(d_ics, d_p, d_res, BATCH, 60.f);  // 60s flight time

  CHECK_CUDA(cudaEventRecord(t1)); CHECK_CUDA(cudaEventSynchronize(t1));
  float ms; cudaEventElapsedTime(&ms, t0, t1);

  argmin_fuel_kernel<<<1, 256>>>(d_res, BATCH, d_best_idx, d_best_fuel);
  CHECK_CUDA(cudaDeviceSynchronize());

  int   h_best; float h_fuel;
  CHECK_CUDA(cudaMemcpy(&h_best, d_best_idx,  sizeof(int),   cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(&h_fuel, d_best_fuel, sizeof(float), cudaMemcpyDeviceToHost));

  printf("Batch: %d trajectories | Time: %.1f ms (%.1f traj/s)\n",
         BATCH, ms, BATCH / (ms/1000.f));
  printf("Best trajectory index: %d | Fuel used: %.1f kg\n", h_best, h_fuel);

  cudaFree(d_ics); cudaFree(d_p); cudaFree(d_res);
  cudaFree(d_best_idx); cudaFree(d_best_fuel);
  delete[] h_ics;
  return 0;
}
