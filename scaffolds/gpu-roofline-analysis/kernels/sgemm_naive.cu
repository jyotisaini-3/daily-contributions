/**
 * Naive SGEMM: one thread per output element, straight global-memory reads.
 * Arithmetic intensity ≈ 2N / (3N²/N) = 2/3 FLOPs/byte for large square matrices.
 * Expected: heavily memory-bound, ~2-5% of peak FP32 throughput.
 */
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void sgemm_naive_kernel(
    const float* __restrict__ A,  // [M,K]
    const float* __restrict__ B,  // [K,N]
    float*       __restrict__ C,  // [M,N]
    int M, int N, int K, float alpha, float beta)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;

  float acc = 0.f;
  for (int k = 0; k < K; ++k)
    acc += A[row * K + k] * B[k * N + col];

  C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

void sgemm_naive(const float* A, const float* B, float* C,
                 int M, int N, int K, float alpha, float beta, cudaStream_t s)
{
  dim3 block(16, 16);
  dim3 grid((N+15)/16, (M+15)/16);
  sgemm_naive_kernel<<<grid, block, 0, s>>>(A, B, C, M, N, K, alpha, beta);
}

int main() {
  // Minimal smoke test: 1024x1024 matrix
  const int N = 1024;
  float *dA, *dB, *dC;
  cudaMalloc(&dA, N*N*4); cudaMalloc(&dB, N*N*4); cudaMalloc(&dC, N*N*4);
  cudaMemset(dA,0,N*N*4); cudaMemset(dB,0,N*N*4); cudaMemset(dC,0,N*N*4);

  cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
  cudaEventRecord(t0);
  for(int i=0;i<10;i++) sgemm_naive(dA,dB,dC,N,N,N,1.f,0.f,0);
  cudaEventRecord(t1); cudaEventSynchronize(t1);

  float ms; cudaEventElapsedTime(&ms,t0,t1); ms/=10;
  double tflops = 2.0*N*N*N/(ms*1e9);
  printf("naive  %dx%d: %.2f ms  %.2f TFLOP/s\n", N,N,ms,tflops);

  cudaFree(dA); cudaFree(dB); cudaFree(dC);
}
