/**
 * cuda-bandwidth-roofline/bandwidth_bench.cu
 *
 * Measures achieved HBM/GDDR memory bandwidth for three kernels and prints
 * their position on the roofline model.
 *
 * Kernels
 * -------
 *   1. copy      : C[i] = A[i]                  AI = 0.167 FLOPs/B (0 math)
 *   2. daxpy     : C[i] = a*A[i] + B[i]          AI = 0.083 FLOPs/B
 *   3. dot_reduce: reduce(A[i]*B[i])              AI = 0.083 FLOPs/B
 *
 * All three are memory-bandwidth-bound; comparing achieved vs theoretical
 * peak locates them on the memory-bound slope of the roofline.
 *
 * Build
 * -----
 *   nvcc -O3 -arch=sm_86 bandwidth_bench.cu -o bandwidth_bench
 *   # For H100: -arch=sm_90
 *
 * Run
 * ---
 *   ./bandwidth_bench
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
#define CHECK(x) do {                                                     \
  cudaError_t _e = (x);                                                   \
  if (_e != cudaSuccess) {                                                \
    fprintf(stderr, "CUDA error %s:%d  %s\n",                            \
            __FILE__, __LINE__, cudaGetErrorString(_e));                  \
    exit(1);                                                              \
  }                                                                       \
} while(0)

static double query_peak_bandwidth_gbs() {
  cudaDeviceProp p;
  CHECK(cudaGetDeviceProperties(&p, 0));
  // busWidth (bits) * memClk (kHz) * 2 (DDR) / 8 (bits->bytes) / 1e9
  return (double)p.memoryBusWidth * p.memoryClockRate * 2.0 / 8.0 / 1e6;
}

static const char* gpu_name() {
  static cudaDeviceProp p;
  static bool init = false;
  if (!init) { cudaGetDeviceProperties(&p, 0); init = true; }
  return p.name;
}

// ---------------------------------------------------------------------------
// Kernel 1 — Copy
//   Reads N floats, writes N floats.  Bytes = 8*N.  FLOPs = 0.
// ---------------------------------------------------------------------------
__global__ void kernel_copy(const float* __restrict__ A,
                             float*       __restrict__ C, int N)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) C[i] = A[i];
}

// ---------------------------------------------------------------------------
// Kernel 2 — SAXPY  (single-precision A*X+Y)
//   Reads 2*N floats, writes N floats.  Bytes = 12*N.  FLOPs = 2*N.
// ---------------------------------------------------------------------------
__global__ void kernel_saxpy(float a,
                              const float* __restrict__ X,
                              const float* __restrict__ Y,
                              float*       __restrict__ Z, int N)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) Z[i] = a * X[i] + Y[i];
}

// ---------------------------------------------------------------------------
// Kernel 3 — Dot product with warp-shuffle reduction
//   Reads 2*N floats.  Bytes = 8*N.  FLOPs = 2*N.
// ---------------------------------------------------------------------------
__global__ void kernel_dot(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float*       __restrict__ out, int N)
{
  float sum = 0.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x)
    sum += A[i] * B[i];

  // Warp-shuffle reduction (PTX: shfl.sync.bfly)
  #pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    sum += __shfl_xor_sync(0xffffffff, sum, mask);

  if ((threadIdx.x & 31) == 0)
    atomicAdd(out, sum);
}

// ---------------------------------------------------------------------------
// Timing helper: returns elapsed milliseconds (average over ITERS launches)
// ---------------------------------------------------------------------------
template<typename F>
double time_kernel_ms(F launch, int warmup = 5, int iters = 50) {
  cudaEvent_t t0, t1;
  CHECK(cudaEventCreate(&t0));
  CHECK(cudaEventCreate(&t1));

  for (int i = 0; i < warmup; ++i) launch();
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaEventRecord(t0));
  for (int i = 0; i < iters; ++i) launch();
  CHECK(cudaEventRecord(t1));
  CHECK(cudaEventSynchronize(t1));

  float ms;
  CHECK(cudaEventElapsedTime(&ms, t0, t1));
  CHECK(cudaEventDestroy(t0));
  CHECK(cudaEventDestroy(t1));
  return (double)ms / iters;
}

// ---------------------------------------------------------------------------
// Roofline ASCII plot
// ---------------------------------------------------------------------------
static void print_roofline(double peak_bw_gbs, double peak_flops_gflops,
                            const char* names[], double ai[], double perf_gflops[],
                            int n)
{
  printf("\n");
  printf("  Roofline — %s\n", gpu_name());
  printf("  Peak BW   : %6.1f GB/s\n", peak_bw_gbs);
  printf("  Peak FP32 : %6.1f GFLOP/s\n", peak_flops_gflops);
  printf("  Ridge pt  : %6.2f FLOP/byte\n",
         peak_flops_gflops / peak_bw_gbs);
  printf("\n");

  // Simple two-column table
  const int W = 54;
  printf("  %-20s %8s %10s %10s %8s\n",
         "Kernel", "AI", "Achvd BW", "Achvd GF", "%%Peak BW");
  printf("  ");
  for (int i = 0; i < W; ++i) printf("-");
  printf("\n");

  for (int k = 0; k < n; ++k) {
    double bw_gbs = perf_gflops[k] / ai[k];  // GB/s
    double pct    = 100.0 * bw_gbs / peak_bw_gbs;
    printf("  %-20s %8.3f %8.1f GB/s %6.1f GF/s %6.1f%%\n",
           names[k], ai[k], bw_gbs, perf_gflops[k], pct);
  }

  // ASCII roofline (log scale, 40 cols wide)
  printf("\n  Roofline (log2 scale, X=AI, Y=GFLOP/s)\n\n");
  const int COLS = 48, ROWS = 16;
  char grid[ROWS][COLS];
  for (int r = 0; r < ROWS; ++r)
    for (int c = 0; c < COLS; ++c)
      grid[r][c] = ' ';

  // Draw memory slope and compute ceiling
  for (int c = 0; c < COLS; ++c) {
    double ai_c = pow(2.0, -2.0 + (double)c / COLS * 8.0);  // 2^-2 .. 2^6
    double perf_roof = fmin(peak_bw_gbs * ai_c, peak_flops_gflops);
    double log_perf  = (log2(perf_roof) - (-1.0)) / (log2(peak_flops_gflops*1.5) - (-1.0));
    int row = (int)((1.0 - log_perf) * (ROWS - 1));
    if (row >= 0 && row < ROWS) grid[row][c] = '-';
  }

  // Plot kernel positions
  const char markers[] = "ABC";
  for (int k = 0; k < n; ++k) {
    double ai_c   = ai[k];
    double perf_c = perf_gflops[k];
    double cx = (log2(ai_c) + 2.0) / 8.0 * COLS;
    double log_perf = (log2(perf_c) - (-1.0)) / (log2(peak_flops_gflops*1.5) - (-1.0));
    int col = (int)cx;
    int row = (int)((1.0 - log_perf) * (ROWS - 1));
    if (row >= 0 && row < ROWS && col >= 0 && col < COLS)
      grid[row][col] = markers[k];
  }

  for (int r = 0; r < ROWS; ++r) {
    printf("  |");
    for (int c = 0; c < COLS; ++c) printf("%c", grid[r][c]);
    printf("\n");
  }
  printf("  +");
  for (int c = 0; c < COLS; ++c) printf("-");
  printf("> AI (FLOP/byte)\n");
  printf("    0.25                 1                   16              64\n");

  printf("\n  A=copy  B=saxpy  C=dot_reduce\n\n");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
  const int N       = 1 << 26;   // 64 M elements = 256 MB per array
  const int THREADS = 256;
  const int BLOCKS  = (N + THREADS - 1) / THREADS;

  float *dA, *dB, *dC, *dOut;
  CHECK(cudaMalloc(&dA,  (size_t)N * sizeof(float)));
  CHECK(cudaMalloc(&dB,  (size_t)N * sizeof(float)));
  CHECK(cudaMalloc(&dC,  (size_t)N * sizeof(float)));
  CHECK(cudaMalloc(&dOut, sizeof(float)));

  // Initialise with known values
  CHECK(cudaMemset(dA, 1, N * sizeof(float)));
  CHECK(cudaMemset(dB, 1, N * sizeof(float)));
  CHECK(cudaMemset(dC, 0, N * sizeof(float)));

  double peak_bw   = query_peak_bandwidth_gbs();
  // FP32 throughput from deviceQuery SM count * 2 * clock (rough)
  cudaDeviceProp dp;
  CHECK(cudaGetDeviceProperties(&dp, 0));
  double peak_fp32 = 2.0 * dp.multiProcessorCount * dp.clockRate
                     * 1e-6 * 64;  // 64 FP32 units/SM on Ampere

  printf("GPU : %s\n", dp.name);
  printf("SMs : %d  |  Clock: %.0f MHz\n", dp.multiProcessorCount,
         dp.clockRate / 1e3);
  printf("VRAM: %.0f GB  |  Bus: %d-bit  |  MemClk: %.0f MHz\n",
         dp.totalGlobalMem / 1e9, dp.memoryBusWidth, dp.memoryClockRate / 1e3);
  printf("Theoretical peak BW: %.1f GB/s\n\n", peak_bw);
  printf("Array size: %d elements = %.0f MB each\n\n", N,
         N * sizeof(float) / 1e6);

  // ------------------------------------------------------------------
  // Benchmark 1: copy
  // ------------------------------------------------------------------
  double ms_copy = time_kernel_ms([&]{
    kernel_copy<<<BLOCKS, THREADS>>>(dA, dC, N);
  });
  double bytes_copy  = 2.0 * N * sizeof(float);   // 1R + 1W
  double flops_copy  = 0.0;                        // no arithmetic
  double bw_copy     = bytes_copy / (ms_copy * 1e6);  // GB/s
  // For roofline: treat as 0 AI; use BW-implied perf
  double ai_copy     = 0.0417;  // 1 FP mov per 24B (1R+1W+reg) — symbolic
  double gflops_copy = bw_copy * ai_copy;
  printf("copy      : %.2f ms  |  %.1f GB/s  (%.1f%% peak)\n",
         ms_copy, bw_copy, 100.0*bw_copy/peak_bw);

  // ------------------------------------------------------------------
  // Benchmark 2: SAXPY
  // ------------------------------------------------------------------
  double ms_saxpy = time_kernel_ms([&]{
    kernel_saxpy<<<BLOCKS, THREADS>>>(1.5f, dA, dB, dC, N);
  });
  double bytes_saxpy  = 3.0 * N * sizeof(float);  // 2R + 1W
  double flops_saxpy  = 2.0 * N;                   // 1 mul + 1 add
  double bw_saxpy     = bytes_saxpy / (ms_saxpy * 1e6);
  double ai_saxpy     = flops_saxpy / bytes_saxpy;
  double gflops_saxpy = flops_saxpy / (ms_saxpy * 1e6);
  printf("saxpy     : %.2f ms  |  %.1f GB/s  (%.1f%% peak)  |  "
         "AI=%.3f  %.1f GFLOP/s\n",
         ms_saxpy, bw_saxpy, 100.0*bw_saxpy/peak_bw, ai_saxpy, gflops_saxpy);

  // ------------------------------------------------------------------
  // Benchmark 3: dot reduce
  // ------------------------------------------------------------------
  CHECK(cudaMemset(dOut, 0, sizeof(float)));
  double ms_dot = time_kernel_ms([&]{
    CHECK(cudaMemset(dOut, 0, sizeof(float)));
    kernel_dot<<<BLOCKS/4, THREADS>>>(dA, dB, dOut, N);
  });
  double bytes_dot  = 2.0 * N * sizeof(float);  // 2R
  double flops_dot  = 2.0 * N;                   // 1 mul + 1 add
  double bw_dot     = bytes_dot / (ms_dot * 1e6);
  double ai_dot     = flops_dot / bytes_dot;
  double gflops_dot = flops_dot / (ms_dot * 1e6);
  printf("dot_reduce: %.2f ms  |  %.1f GB/s  (%.1f%% peak)  |  "
         "AI=%.3f  %.1f GFLOP/s\n",
         ms_dot, bw_dot, 100.0*bw_dot/peak_bw, ai_dot, gflops_dot);

  // ------------------------------------------------------------------
  // Roofline summary
  // ------------------------------------------------------------------
  const char* names[]     = {"copy", "saxpy", "dot_reduce"};
  double ai[]             = {ai_copy, ai_saxpy, ai_dot};
  double gflops[]         = {gflops_copy, gflops_saxpy, gflops_dot};
  print_roofline(peak_bw, peak_fp32, names, ai, gflops, 3);

  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dOut);
  return 0;
}
