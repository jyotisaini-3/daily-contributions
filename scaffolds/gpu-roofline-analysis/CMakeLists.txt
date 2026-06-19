cmake_minimum_required(VERSION 3.20)
project(gpu_roofline CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 86 90)  # Ampere + Hopper
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

foreach(kernel sgemm_naive sgemm_tiled sgemm_vectorized sgemm_tensor_core)
  add_executable(${kernel} kernels/${kernel}.cu)
  target_link_libraries(${kernel} CUDA::cudart)
  target_compile_options(${kernel} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo --expt-relaxed-constexpr>)
endforeach()
