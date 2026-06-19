cmake_minimum_required(VERSION 3.25)
project(hopper_warp_gemm CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 90)  # Hopper SM90
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# CUTLASS 3.x — clone alongside this repo or set CUTLASS_DIR
if(NOT DEFINED CUTLASS_DIR)
  set(CUTLASS_DIR "${CMAKE_SOURCE_DIR}/../cutlass")
endif()

find_package(CUDAToolkit REQUIRED)

include_directories(
  ${CUTLASS_DIR}/include
  ${CUTLASS_DIR}/tools/util/include
  ${CMAKE_SOURCE_DIR}/include
)

add_executable(hopper_gemm_bench
  src/hopper_warp_gemm.cu
  benchmarks/bench.cu
)

target_link_libraries(hopper_gemm_bench
  CUDA::cudart
  CUDA::cublas
)

target_compile_options(hopper_gemm_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -lineinfo
    -O3
  >
)
