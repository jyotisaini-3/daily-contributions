#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

// SM90 Hopper warp-specialization roles
// Producer warps issue TMA async bulk copies; consumer warps compute MMA.
// This separation hides memory latency behind math and maximises occupancy.
enum class WarpRole : uint8_t {
  kProducer = 0,  // issues cp.async.bulk.tensor (TMA)
  kConsumer = 1,  // executes wgmma.mma_async on shared-memory tiles
};

struct GemmParams {
  const float* A;   // [M, K] row-major
  const float* B;   // [K, N] row-major
  float*       C;   // [M, N] row-major  (output)
  int M, N, K;
  float alpha, beta;
};

// Tile sizes tuned for SM90 128-byte TMA granularity
constexpr int TILE_M = 128;
constexpr int TILE_N = 128;
constexpr int TILE_K = 64;

// Number of pipeline stages in shared-memory double-buffer
constexpr int PIPELINE_STAGES = 4;

void hopper_warp_gemm(const GemmParams& p, cudaStream_t stream = 0);
