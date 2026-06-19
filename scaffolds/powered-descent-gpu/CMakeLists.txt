cmake_minimum_required(VERSION 3.20)
project(powered_descent_gpu CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 86 90)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_executable(monte_carlo
  src/trajectory_solver.cu
  src/monte_carlo.cu
)
target_link_libraries(monte_carlo
  CUDA::cudart CUDA::cublas CUDA::cusolver)
target_compile_options(monte_carlo PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo --expt-relaxed-constexpr>)

add_executable(test_dynamics tests/test_dynamics.cu)
target_link_libraries(test_dynamics CUDA::cudart)
