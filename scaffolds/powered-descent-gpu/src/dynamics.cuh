#pragma once
#include <cuda_runtime.h>
#include <math.h>

// ---------------------------------------------------------------------------
// 6-DoF rocket state vector
//   x = [rx, ry, rz,   vx, vy, vz,   qw, qx, qy, qz,   wx, wy, wz,   mass]
//         position(3)  velocity(3)  quaternion(4)  ang_vel(3)  wet mass(1)
//   Total: 14 states
// ---------------------------------------------------------------------------
struct State {
  float r[3];   // position [m]
  float v[3];   // velocity [m/s]
  float q[4];   // attitude quaternion [w,x,y,z]
  float w[3];   // angular velocity [rad/s]
  float mass;   // current mass [kg]
};

// ---------------------------------------------------------------------------
// Control vector: thrust direction + magnitude
// ---------------------------------------------------------------------------
struct Control {
  float T[3];   // thrust vector in body frame [N]
};

// ---------------------------------------------------------------------------
// Rocket physical parameters
// ---------------------------------------------------------------------------
struct RocketParams {
  float g[3];       // gravity [m/s^2]  e.g. {0, 0, -9.806}
  float Isp;        // specific impulse [s]
  float g0;         // standard gravity 9.806 [m/s^2]
  float T_min;      // minimum thrust [N]
  float T_max;      // maximum thrust [N]
  float mass_dry;   // dry mass [kg]
  float mass_wet;   // initial wet mass [kg]
  float r_T[3];     // thruster moment arm [m]
  float J[3];       // principal moments of inertia [kg.m^2]
};

// ---------------------------------------------------------------------------
// Quaternion helpers (device)
// ---------------------------------------------------------------------------
__device__ __forceinline__
void quat_mult(const float* q1, const float* q2, float* out) {
  // Hamilton product: q1 * q2
  out[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3];
  out[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2];
  out[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1];
  out[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0];
}

__device__ __forceinline__
void body_to_inertial(const float* q, const float* v_body, float* v_inertial) {
  // Rotate vector v_body by quaternion q into inertial frame
  // Uses: v_i = q ⊗ [0,v_b] ⊗ q*
  float tmp[4] = {0, v_body[0], v_body[1], v_body[2]};
  float q_conj[4] = {q[0], -q[1], -q[2], -q[3]};
  float mid[4], result[4];
  quat_mult(q, tmp, mid);
  quat_mult(mid, q_conj, result);
  v_inertial[0] = result[1];
  v_inertial[1] = result[2];
  v_inertial[2] = result[3];
}

// ---------------------------------------------------------------------------
// Continuous-time dynamics  ẋ = f(x, u)
// ---------------------------------------------------------------------------
__device__
void dynamics(const State& x, const Control& u, const RocketParams& p, State& xdot) {
  // Position derivative = velocity
  xdot.r[0] = x.v[0]; xdot.r[1] = x.v[1]; xdot.r[2] = x.v[2];

  // Thrust in inertial frame
  float T_inertial[3];
  body_to_inertial(x.q, u.T, T_inertial);

  // Velocity derivative = gravity + thrust/mass
  for (int i = 0; i < 3; ++i)
    xdot.v[i] = p.g[i] + T_inertial[i] / x.mass;

  // Quaternion derivative = 0.5 * q ⊗ [0, ω]
  float omega_quat[4] = {0, x.w[0], x.w[1], x.w[2]};
  float qdot[4];
  quat_mult(x.q, omega_quat, qdot);
  xdot.q[0] = 0.5f * qdot[0]; xdot.q[1] = 0.5f * qdot[1];
  xdot.q[2] = 0.5f * qdot[2]; xdot.q[3] = 0.5f * qdot[3];

  // Angular velocity derivative (Euler's equations, simplified diagonal J)
  // τ = r_T × T (cross product of moment arm and thrust)
  float tau[3] = {
    p.r_T[1]*u.T[2] - p.r_T[2]*u.T[1],
    p.r_T[2]*u.T[0] - p.r_T[0]*u.T[2],
    p.r_T[0]*u.T[1] - p.r_T[1]*u.T[0]
  };
  xdot.w[0] = (tau[0] - (p.J[2]-p.J[1])*x.w[1]*x.w[2]) / p.J[0];
  xdot.w[1] = (tau[1] - (p.J[0]-p.J[2])*x.w[2]*x.w[0]) / p.J[1];
  xdot.w[2] = (tau[2] - (p.J[1]-p.J[0])*x.w[0]*x.w[1]) / p.J[2];

  // Mass depletion: ṁ = -‖T‖ / (Isp * g0)
  float T_norm = sqrtf(u.T[0]*u.T[0] + u.T[1]*u.T[1] + u.T[2]*u.T[2]);
  xdot.mass = -T_norm / (p.Isp * p.g0);
}

// ---------------------------------------------------------------------------
// RK4 integrator step
// ---------------------------------------------------------------------------
__device__
void rk4_step(State& x, const Control& u, const RocketParams& p, float dt) {
  State k1, k2, k3, k4, xtmp;
  dynamics(x, u, p, k1);

  // k2
  #define EULER(dst, src, deriv, h) do { \
    dst.r[0]=src.r[0]+(h)*deriv.r[0]; dst.r[1]=src.r[1]+(h)*deriv.r[1]; dst.r[2]=src.r[2]+(h)*deriv.r[2]; \
    dst.v[0]=src.v[0]+(h)*deriv.v[0]; dst.v[1]=src.v[1]+(h)*deriv.v[1]; dst.v[2]=src.v[2]+(h)*deriv.v[2]; \
    dst.q[0]=src.q[0]+(h)*deriv.q[0]; dst.q[1]=src.q[1]+(h)*deriv.q[1]; \
    dst.q[2]=src.q[2]+(h)*deriv.q[2]; dst.q[3]=src.q[3]+(h)*deriv.q[3]; \
    dst.w[0]=src.w[0]+(h)*deriv.w[0]; dst.w[1]=src.w[1]+(h)*deriv.w[1]; dst.w[2]=src.w[2]+(h)*deriv.w[2]; \
    dst.mass=src.mass+(h)*deriv.mass; \
  } while(0)

  EULER(xtmp, x, k1, 0.5f*dt); dynamics(xtmp, u, p, k2);
  EULER(xtmp, x, k2, 0.5f*dt); dynamics(xtmp, u, p, k3);
  EULER(xtmp, x, k3,      dt); dynamics(xtmp, u, p, k4);
  #undef EULER

  #define COMBINE(field) x.field = x.field + (dt/6.f)*(k1.field+2.f*k2.field+2.f*k3.field+k4.field)
  for(int i=0;i<3;i++){COMBINE(r[i]);COMBINE(v[i]);COMBINE(w[i]);}
  for(int i=0;i<4;i++) COMBINE(q[i]);
  COMBINE(mass);
  #undef COMBINE

  // Re-normalise quaternion to prevent drift
  float qnorm = sqrtf(x.q[0]*x.q[0]+x.q[1]*x.q[1]+x.q[2]*x.q[2]+x.q[3]*x.q[3]);
  for(int i=0;i<4;i++) x.q[i] /= qnorm;
}
