#include "../src/dynamics.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// Run dynamics unit tests on GPU
__global__ void test_dynamics_kernel(int* pass_count) {
  RocketParams p;
  p.g[0]=0; p.g[1]=0; p.g[2]=-9.806f;
  p.Isp=311.f; p.g0=9.806f;
  p.T_min=180e3f; p.T_max=845e3f;
  p.mass_dry=22200.f; p.mass_wet=25000.f;
  p.r_T[0]=p.r_T[1]=0; p.r_T[2]=-20.f;
  p.J[0]=p.J[1]=1.5e7f; p.J[2]=1e5f;

  // Test 1: hovering (T = -m*g upward) → zero acceleration
  {
    State x = {};
    x.q[0]=1.f; x.mass=25000.f;
    Control u; u.T[0]=0; u.T[1]=0; u.T[2]=25000.f*9.806f;
    State xdot;
    dynamics(x, u, p, xdot);
    bool ok = fabsf(xdot.v[2]) < 0.01f;
    if (ok) atomicAdd(pass_count, 1);
    printf("Test 1 (hover): %s (vz_dot=%.4f, expect ~0)\n", ok?"PASS":"FAIL", xdot.v[2]);
  }

  // Test 2: zero thrust → free fall
  {
    State x = {};
    x.q[0]=1.f; x.mass=25000.f;
    Control u = {};
    State xdot;
    dynamics(x, u, p, xdot);
    bool ok = fabsf(xdot.v[2] - (-9.806f)) < 0.01f;
    if (ok) atomicAdd(pass_count, 1);
    printf("Test 2 (free fall): %s (vz_dot=%.4f, expect -9.806)\n", ok?"PASS":"FAIL", xdot.v[2]);
  }

  // Test 3: RK4 step conserves quaternion norm
  {
    State x = {};
    x.q[0]=1.f; x.mass=24000.f;
    x.w[0]=0.1f; x.w[1]=0.05f;
    Control u; u.T[0]=0; u.T[1]=0; u.T[2]=24000.f*9.806f;
    rk4_step(x, u, p, 0.1f);
    float qnorm = sqrtf(x.q[0]*x.q[0]+x.q[1]*x.q[1]+x.q[2]*x.q[2]+x.q[3]*x.q[3]);
    bool ok = fabsf(qnorm - 1.f) < 1e-5f;
    if (ok) atomicAdd(pass_count, 1);
    printf("Test 3 (quat norm): %s (norm=%.7f, expect 1.0)\n", ok?"PASS":"FAIL", qnorm);
  }
}

int main() {
  int *d_pass; cudaMalloc(&d_pass, sizeof(int)); cudaMemset(d_pass, 0, sizeof(int));
  test_dynamics_kernel<<<1, 1>>>(d_pass);
  cudaDeviceSynchronize();
  int h_pass; cudaMemcpy(&h_pass, d_pass, sizeof(int), cudaMemcpyDeviceToHost);
  printf("\n%d/3 tests passed\n", h_pass);
  cudaFree(d_pass);
  return (h_pass == 3) ? 0 : 1;
}
