"""Visualise batch powered-descent trajectories from Monte-Carlo solver output."""
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path


def load_results(csv_path: str) -> np.ndarray:
    """Load trajectory CSV: columns = [traj_id, t, rx, ry, rz, vx, vy, vz, fuel]."""
    return np.loadtxt(csv_path, delimiter=',', skiprows=1)


def plot_trajectories(data: np.ndarray, best_idx: int = 0, max_show: int = 50):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    traj_ids = np.unique(data[:, 0]).astype(int)
    cmap = plt.cm.viridis

    for i, tid in enumerate(traj_ids[:max_show]):
        mask = data[:, 0] == tid
        rx, ry, rz = data[mask, 2], data[mask, 3], data[mask, 4]
        fuel = data[mask, 8][-1]
        color = cmap(fuel / data[:, 8].max())
        alpha = 0.8 if tid == best_idx else 0.2
        lw    = 2.0 if tid == best_idx else 0.5
        ax.plot(rx, ry, rz, color=color, alpha=alpha, lw=lw)

    # Landing pad
    theta = np.linspace(0, 2*np.pi, 50)
    ax.plot(10*np.cos(theta), 10*np.sin(theta), 0, 'r-', lw=2, label='Landing pad')

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Altitude [m]')
    ax.set_title(f'Powered Descent — {len(traj_ids)} Monte-Carlo trajectories\n'
                 f'Best (yellow): traj {best_idx}')

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(data[:, 8])
    fig.colorbar(sm, ax=ax, label='Fuel used [kg]', shrink=0.5)

    plt.tight_layout()
    out = Path('trajectories_3d.png')
    plt.savefig(out, dpi=150)
    print(f'Saved: {out}')
    plt.show()


if __name__ == '__main__':
    csv = sys.argv[1] if len(sys.argv) > 1 else 'results.csv'
    best = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    data = load_results(csv)
    plot_trajectories(data, best_idx=best)
