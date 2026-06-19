#include "warp_specialized_gemm.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(x) do { cudaError_t e = (x); if(e) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

double measure_tflops(int M, int N, int K, double ms) {
  return 2.0 * M * N * K / (ms * 1e9);  // 2 FLOPs per FMA, ms→s
}

void random_fill(float* ptr, int n) {
  for (int i = 0; i < n; ++i) ptr[i] = (float)rand() / RAND_MAX - 0.5f;
}

int main() {
  srand(42);
  const int M = 4096, N = 4096, K = 4096;
  const int WARMUP = 5, ITERS = 20;

  float *hA = new float[M*K], *hB = new float[K*N];
  random_fill(hA, M*K);
  random_fill(hB, K*N);

  float *dA, *dB, *dC;
  CHECK_CUDA(cudaMalloc(&dA, M*K*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dB, K*N*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dC, M*N*sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dA, hA, M*K*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB, K*N*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(dC, 0, M*N*sizeof(float)));

  GemmParams params{dA, dB, dC, M, N, K, 1.f, 0.f};
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  // --- cuBLAS baseline ---
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetStream(handle, stream);

  cudaEvent_t t0, t1;
  CHECK_CUDA(cudaEventCreate(&t0));
  CHECK_CUDA(cudaEventCreate(&t1));

  float alpha = 1.f, beta = 0.f;
  for (int i = 0; i < WARMUP; ++i)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  CHECK_CUDA(cudaEventRecord(t0, stream));
  for (int i = 0; i < ITERS; ++i)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, N, &alpha, dB, N, dA, K, &beta, dC, N);
  CHECK_CUDA(cudaEventRecord(t1, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  float ms_cublas;
  cudaEventElapsedTime(&ms_cublas, t0, t1);
  ms_cublas /= ITERS;

  // --- Warp-specialized kernel ---
  CHECK_CUDA(cudaMemset(dC, 0, M*N*sizeof(float)));
  for (int i = 0; i < WARMUP; ++i) hopper_warp_gemm(params, stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  CHECK_CUDA(cudaEventRecord(t0, stream));
  for (int i = 0; i < ITERS; ++i) hopper_warp_gemm(params, stream);
  CHECK_CUDA(cudaEventRecord(t1, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  float ms_custom;
  cudaEventElapsedTime(&ms_custom, t0, t1);
  ms_custom /= ITERS;

  printf("Matrix: %dx%dx%d (FP32)\n", M, N, K);
  printf("cuBLAS:  %6.2f ms  →  %6.2f TFLOP/s\n", ms_cublas, measure_tflops(M,N,K,ms_cublas));
  printf("Hopper warp-spec: %6.2f ms  →  %6.2f TFLOP/s\n", ms_custom, measure_tflops(M,N,K,ms_custom));
  printf("Ratio (custom/cuBLAS): %.3f\n", ms_cublas / ms_custom);

  cublasDestroy(handle);
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  delete[] hA; delete[] hB;
  return 0;
}
