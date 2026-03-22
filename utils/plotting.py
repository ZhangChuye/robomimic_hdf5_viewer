"""Publication-quality (CoRL-style) plotting utilities for RBY1 teleop HDF5 data."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
#  Style
# ═══════════════════════════════════════════════════════════════════════════════

CORL_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "legend.framealpha": 0.85,
    "figure.titlesize": 20,
    "figure.titleweight": "bold",
    "lines.linewidth": 2.0,
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "figure.constrained_layout.use": True,
}

OBS_COLOR = "#1f77b4"
CMD_COLOR = "#d62728"
OBS_STYLE = dict(color=OBS_COLOR, linestyle="-", linewidth=2.2, alpha=0.95)
CMD_STYLE = dict(color=CMD_COLOR, linestyle="--", linewidth=1.8, alpha=0.85)


def setup_style():
    plt.rcParams.update(CORL_RC)


# ═══════════════════════════════════════════════════════════════════════════════
#  Layout helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _grid(n):
    """Return (rows, cols) for n subplots."""
    if n <= 2:  return 1, n
    if n <= 4:  return 2, 2
    if n <= 6:  return 2, 3
    if n <= 8:  return 2, 4
    if n <= 9:  return 3, 3
    if n <= 12: return 3, 4
    return (n + 3) // 4, 4


def _is_all_zero(arr):
    if arr is None:
        return True
    return np.allclose(arr, 0, atol=1e-8)


# ═══════════════════════════════════════════════════════════════════════════════
#  Geometry: 4×4 matrix → position + Euler
# ═══════════════════════════════════════════════════════════════════════════════

def _flat4x4_to_pos_euler(flat):
    """(N, 16) row-major 4×4 → (N, 3) position, (N, 3) euler [roll, pitch, yaw]."""
    M = flat.reshape(-1, 4, 4)
    pos = M[:, :3, 3]
    R = M[:, :3, :3]
    sy = np.sqrt(R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2)
    sing = sy < 1e-6
    roll = np.where(~sing, np.arctan2(R[:, 2, 1], R[:, 2, 2]),
                    np.arctan2(-R[:, 1, 2], R[:, 1, 1]))
    pitch = np.arctan2(-R[:, 2, 0], sy)
    yaw = np.where(~sing, np.arctan2(R[:, 1, 0], R[:, 0, 0]), 0.0)
    return pos, np.column_stack([roll, pitch, yaw])


# ═══════════════════════════════════════════════════════════════════════════════
#  Joint-group / hand-group definitions
# ═══════════════════════════════════════════════════════════════════════════════

# obs_slice: indices into robot0_joint_pos (26-dim)
# act_slice: indices into actions/joint   (49-dim)
BODY_GROUPS = [
    ("Left Arm",  "left_arm",  slice(17, 24), slice(0, 7),  [f"J{i}" for i in range(7)]),
    ("Right Arm", "right_arm", slice(10, 17), slice(7, 14), [f"J{i}" for i in range(7)]),
    ("Torso",     "torso",     slice(4, 10),  slice(14, 20),[f"J{i}" for i in range(6)]),
    ("Head",      "head",      slice(24, 26), slice(20, 22),["Pan", "Tilt"]),
]

HAND_GROUPS = [
    ("Left Hand",  "left_hand",  "hand_left_qpos",  "hand_left_cmd_qpos",  slice(25, 37),
     ["Th.Bnd", "Th.R1", "Th.R2", "Idx.Bnd", "Idx.1", "Idx.2",
      "Mid.1", "Mid.2", "Rng.1", "Rng.2", "Pnk.1", "Pnk.2"]),
    ("Right Hand", "right_hand", "hand_right_qpos", "hand_right_cmd_qpos", slice(37, 49),
     ["Th.Bnd", "Th.R1", "Th.R2", "Idx.Bnd", "Idx.1", "Idx.2",
      "Mid.1", "Mid.2", "Rng.1", "Rng.2", "Pnk.1", "Pnk.2"]),
]

BASE_LABELS = ["Δx (m)", "Δy (m)", "Δyaw (rad)"]


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 2 — Per-group joint: Command vs Observation
# ═══════════════════════════════════════════════════════════════════════════════

def plot_joint_comparison(t, obs, cmd, labels, title, save_path):
    """
    Grid of subplots: one per DOF. Solid blue = observation, dashed red = command.
    obs, cmd: (N, D).  labels: list[D].
    """
    setup_style()
    D = obs.shape[1]
    nr, nc = _grid(D)
    fig, axes = plt.subplots(nr, nc, figsize=(nc * 4.5 + 1.0, nr * 3.2 + 1.5))
    axes = np.atleast_2d(axes).reshape(nr, nc)
    fig.suptitle(title, fontsize=20, fontweight="bold")

    for i in range(D):
        ax = axes[i // nc, i % nc]
        ax.plot(t, obs[:, i], label="Observation", **OBS_STYLE)
        ax.plot(t, cmd[:, i], label="Command", **CMD_STYLE)
        ax.set_title(labels[i], fontsize=14, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Angle (rad)", fontsize=12)
        if i == 0:
            ax.legend(loc="upper right", fontsize=11)

    for i in range(D, nr * nc):
        axes[i // nc, i % nc].set_visible(False)

    fig.savefig(save_path)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 3 — Per-EEF Cartesian: Command vs Observation
# ═══════════════════════════════════════════════════════════════════════════════

_CART_POS_LABELS = ["X (m)", "Y (m)", "Z (m)"]
_CART_ORI_LABELS = ["Roll (rad)", "Pitch (rad)", "Yaw (rad)"]


def plot_cartesian_comparison(t, obs_flat, cmd_flat, side_name, save_path):
    """
    2×3 grid: top row = XYZ position, bottom row = RPY orientation.
    obs_flat, cmd_flat: (N, 16).
    """
    setup_style()
    obs_pos, obs_eul = _flat4x4_to_pos_euler(obs_flat)
    cmd_pos, cmd_eul = _flat4x4_to_pos_euler(cmd_flat)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(f"{side_name} EEF: Command vs. Observation", fontsize=20, fontweight="bold")

    for j in range(3):
        ax = axes[0, j]
        ax.plot(t, obs_pos[:, j], label="Observation", **OBS_STYLE)
        ax.plot(t, cmd_pos[:, j], label="Command", **CMD_STYLE)
        ax.set_title(_CART_POS_LABELS[j], fontsize=14, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Position (m)", fontsize=12)
        if j == 0:
            ax.legend(fontsize=11)

        ax = axes[1, j]
        ax.plot(t, obs_eul[:, j], label="Observation", **OBS_STYLE)
        ax.plot(t, cmd_eul[:, j], label="Command", **CMD_STYLE)
        ax.set_title(_CART_ORI_LABELS[j], fontsize=14, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Angle (rad)", fontsize=12)

    fig.savefig(save_path)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Plots 4 & 7 — All-joints overview (action-only or proprio-only)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_joint_overview(t, groups, title, save_path):
    """
    Stacked subplots, one per body group. Each subplot overlays all DOFs.
    groups: list of (group_name, (N,D) array, list[D] labels).
    """
    setup_style()
    n = len(groups)
    fig, axes = plt.subplots(n, 1, figsize=(16, n * 3.0 + 1.5), sharex=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=20, fontweight="bold")

    for idx, (name, data, labels) in enumerate(groups):
        ax = axes[idx]
        D = data.shape[1]
        cmap = plt.cm.tab10 if D <= 10 else plt.cm.tab20
        colors = [cmap(i / max(D - 1, 1)) for i in range(D)]
        for j in range(D):
            ax.plot(t, data[:, j], color=colors[j], linewidth=1.8, label=labels[j])
        ax.set_ylabel("Angle (rad)", fontsize=12)
        ax.set_title(name, fontsize=15, fontweight="bold", loc="left")
        ax.legend(loc="upper right", ncol=min(D, 6), fontsize=9, framealpha=0.8)

    axes[-1].set_xlabel("Time (s)", fontsize=13)
    fig.savefig(save_path)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Plots 5 & 6 — All-Cartesian overview (action or proprio)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_cartesian_overview(t, eef_list, title, save_path):
    """
    Stacked subplots: each EEF gets a position row and an orientation row.
    eef_list: list of (side_name, (N,16) flat matrix).
    """
    setup_style()
    n_eef = len(eef_list)
    fig, axes = plt.subplots(n_eef * 2, 1, figsize=(16, n_eef * 6.0 + 1.5), sharex=True)
    if n_eef * 2 == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=20, fontweight="bold")

    pos_colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    ori_colors = ["#ff7f00", "#984ea3", "#a65628"]

    for i, (side, flat) in enumerate(eef_list):
        pos, eul = _flat4x4_to_pos_euler(flat)
        ax_p = axes[i * 2]
        ax_o = axes[i * 2 + 1]

        for j, (lbl, c) in enumerate(zip(_CART_POS_LABELS, pos_colors)):
            ax_p.plot(t, pos[:, j], color=c, linewidth=2.0, label=lbl)
        ax_p.set_title(f"{side} — Position", fontsize=15, fontweight="bold", loc="left")
        ax_p.set_ylabel("Position (m)", fontsize=12)
        ax_p.legend(loc="upper right", fontsize=11)

        for j, (lbl, c) in enumerate(zip(_CART_ORI_LABELS, ori_colors)):
            ax_o.plot(t, eul[:, j], color=c, linewidth=2.0, label=lbl)
        ax_o.set_title(f"{side} — Orientation", fontsize=15, fontweight="bold", loc="left")
        ax_o.set_ylabel("Angle (rad)", fontsize=12)
        ax_o.legend(loc="upper right", fontsize=11)

    axes[-1].set_xlabel("Time (s)", fontsize=13)
    fig.savefig(save_path)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 1 — Aria video → mp4
# ═══════════════════════════════════════════════════════════════════════════════

def save_video(images, fps, save_path):
    """
    Save (N, H, W, 3) uint8 RGB array as mp4.
    Tries cv2 first, falls back to imageio.
    """
    save_path = str(save_path)
    N, H, W, _ = images.shape
    try:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (W, H))
        for i in range(N):
            writer.write(cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR))
        writer.release()
    except ImportError:
        import imageio
        writer = imageio.get_writer(save_path, fps=fps, codec="libx264")
        for i in range(N):
            writer.append_data(images[i])
        writer.close()
