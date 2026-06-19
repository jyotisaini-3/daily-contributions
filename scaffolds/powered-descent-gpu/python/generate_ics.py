"""Generate Monte-Carlo initial conditions for the powered-descent batch solver."""
import numpy as np
import argparse


def generate(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ics = np.zeros((n, 14))  # [rx,ry,rz, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz, mass]

    # Position: 500–2000 m altitude, ±200 m lateral spread
    ics[:, 0] = rng.uniform(-200, 200, n)   # rx
    ics[:, 1] = rng.uniform(-200, 200, n)   # ry
    ics[:, 2] = rng.uniform(500, 2000, n)   # rz (altitude)

    # Velocity: mostly downward
    ics[:, 3] = rng.uniform(-10, 10, n)     # vx
    ics[:, 4] = rng.uniform(-10, 10, n)     # vy
    ics[:, 5] = rng.uniform(-100, -30, n)   # vz (descending)

    # Attitude: near identity
    ics[:, 6] = 1.0  # qw
    ics[:, 7:10] = rng.normal(0, 0.01, (n, 3))  # small perturbations
    # Normalise quaternion
    qnorm = np.linalg.norm(ics[:, 6:10], axis=1, keepdims=True)
    ics[:, 6:10] /= qnorm

    # Angular velocity: small
    ics[:, 10:13] = rng.normal(0, 0.02, (n, 3))

    # Mass: wet mass 25000 kg
    ics[:, 13] = 25000.0

    return ics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('-o', default='initial_conditions.csv')
    args = parser.parse_args()

    ics = generate(args.n, args.seed)
    header = 'rx,ry,rz,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz,mass'
    np.savetxt(args.o, ics, delimiter=',', header=header, comments='')
    print(f'Written {args.n} initial conditions to {args.o}')
