cmake_minimum_required(VERSION 3.20)
project(cuda_bandwidth_roofline CUDA)

# Change to 90 for H100, 80 for A100, 89 for L40
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_executable(bandwidth_bench bandwidth_bench.cu)
target_link_libraries(bandwidth_bench CUDA::cudart)
target_compile_options(bandwidth_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)
