/**
 * Tensor-core SGEMM using WMMA (warp-level matrix multiply-accumulate).
 * Loads 16x16 FP16 A/B fragments, accumulates into FP32 C.
 * Achieves ~8x the FP32 FLOP rate vs CUDA cores on Ampere.
 */
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <stdio.h>

using namespace nvcuda;

constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
constexpr int WARP_TILE_M = 2, WARP_TILE_N = 2;  // warps handle 2x2 WMMA tiles

__global__ void sgemm_tensor_core_kernel(
    const half*  __restrict__ A,   // [M,K] FP16
    const half*  __restrict__ B,   // [K,N] FP16
    float*       __restrict__ C,   // [M,N] FP32
    int M, int N, int K)
{
  // Each warp computes WARP_TILE_M x WARP_TILE_N output tiles
  int warp_row = (blockIdx.y * blockDim.y + threadIdx.y) / 32 * WMMA_M * WARP_TILE_M;
  int warp_col = (blockIdx.x * blockDim.x + threadIdx.x) * WMMA_N * WARP_TILE_N;

  // Accumulator fragments
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
    acc[WARP_TILE_M][WARP_TILE_N];
  for (int i = 0; i < WARP_TILE_M; ++i)
    for (int j = 0; j < WARP_TILE_N; ++j)
      wmma::fill_fragment(acc[i][j], 0.f);

  for (int k = 0; k < K; k += WMMA_K) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[WARP_TILE_N];

    for (int i = 0; i < WARP_TILE_M; ++i)
      if (warp_row + i*WMMA_M < M && k < K)
        wmma::load_matrix_sync(a_frag[i], A + (warp_row+i*WMMA_M)*K + k, K);

    for (int j = 0; j < WARP_TILE_N; ++j)
      if (k < K && warp_col + j*WMMA_N < N)
        wmma::load_matrix_sync(b_frag[j], B + k*N + warp_col+j*WMMA_N, N);

    for (int i = 0; i < WARP_TILE_M; ++i)
      for (int j = 0; j < WARP_TILE_N; ++j)
        wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
  }

  for (int i = 0; i < WARP_TILE_M; ++i)
    for (int j = 0; j < WARP_TILE_N; ++j)
      if (warp_row+i*WMMA_M < M && warp_col+j*WMMA_N < N)
        wmma::store_matrix_sync(C+(warp_row+i*WMMA_M)*N+warp_col+j*WMMA_N,
                                acc[i][j], N, wmma::mem_row_major);
}

int main() {
  const int N = 4096;
  half *dAh, *dBh; float *dC;
  cudaMalloc(&dAh,N*N*sizeof(half)); cudaMalloc(&dBh,N*N*sizeof(half));
  cudaMalloc(&dC, N*N*sizeof(float));
  cudaMemset(dAh,0,N*N*sizeof(half)); cudaMemset(dBh,0,N*N*sizeof(half));
  cudaMemset(dC,0,N*N*sizeof(float));

  dim3 block(128, 4);
  dim3 grid((N+WMMA_N*WARP_TILE_N-1)/(WMMA_N*WARP_TILE_N),
            (N+WMMA_M*WARP_TILE_M*4-1)/(WMMA_M*WARP_TILE_M*4));

  cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
  sgemm_tensor_core_kernel<<<grid,block>>>(dAh,dBh,dC,N,N,N);
  cudaDeviceSynchronize();

  cudaEventRecord(t0);
  for(int i=0;i<20;i++)
    sgemm_tensor_core_kernel<<<grid,block>>>(dAh,dBh,dC,N,N,N);
  cudaEventRecord(t1); cudaEventSynchronize(t1);

  float ms; cudaEventElapsedTime(&ms,t0,t1); ms/=20;
  double tflops = 2.0*N*N*N/(ms*1e9);
  printf("tensor_core %dx%d: %.2f ms  %.2f TFLOP/s\n",N,N,ms,tflops);

  cudaFree(dAh); cudaFree(dBh); cudaFree(dC);
}
