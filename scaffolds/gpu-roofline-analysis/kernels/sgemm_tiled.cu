/**
 * Tiled SGEMM using shared memory.
 * Each CTA loads a TILE×TILE block of A and B into __shared__, then
 * computes the partial dot product before moving to the next K-tile.
 * Reduces global-memory traffic by factor TILE, boosting arithmetic intensity.
 */
#include <cuda_runtime.h>
#include <stdio.h>

constexpr int TILE = 32;

__global__ void sgemm_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K, float alpha, float beta)
{
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.f;

  for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
    // Cooperatively load tile of A
    As[threadIdx.y][threadIdx.x] =
      (row < M && t*TILE+threadIdx.x < K) ? A[row*K + t*TILE+threadIdx.x] : 0.f;
    // Cooperatively load tile of B
    Bs[threadIdx.y][threadIdx.x] =
      (t*TILE+threadIdx.y < K && col < N) ? B[(t*TILE+threadIdx.y)*N + col] : 0.f;
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE; ++k)
      acc = __fmaf_rn(As[threadIdx.y][k], Bs[k][threadIdx.x], acc);
    __syncthreads();
  }

  if (row < M && col < N)
    C[row*N+col] = alpha*acc + beta*C[row*N+col];
}

void sgemm_tiled(const float* A, const float* B, float* C,
                 int M, int N, int K, float alpha, float beta, cudaStream_t s)
{
  dim3 block(TILE, TILE);
  dim3 grid((N+TILE-1)/TILE, (M+TILE-1)/TILE);
  sgemm_tiled_kernel<<<grid,block,0,s>>>(A,B,C,M,N,K,alpha,beta);
}

int main() {
  const int N = 4096;
  float *dA,*dB,*dC;
  cudaMalloc(&dA,N*N*4); cudaMalloc(&dB,N*N*4); cudaMalloc(&dC,N*N*4);
  cudaMemset(dA,1,N*N*4); cudaMemset(dB,1,N*N*4); cudaMemset(dC,0,N*N*4);

  cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
  // Warmup
  sgemm_tiled(dA,dB,dC,N,N,N,1.f,0.f,0);
  cudaDeviceSynchronize();

  cudaEventRecord(t0);
  for(int i=0;i<20;i++) sgemm_tiled(dA,dB,dC,N,N,N,1.f,0.f,0);
  cudaEventRecord(t1); cudaEventSynchronize(t1);

  float ms; cudaEventElapsedTime(&ms,t0,t1); ms/=20;
  double tflops = 2.0*N*N*N/(ms*1e9);
  printf("tiled  %dx%d: %.2f ms  %.2f TFLOP/s\n",N,N,ms,tflops);

  cudaFree(dA); cudaFree(dB); cudaFree(dC);
}
