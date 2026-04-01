"""Load demo data from robomimic-style HDF5 files."""

import h5py
import numpy as np
from pathlib import Path


def list_demos(path):
    """Return sorted list of demo keys in the HDF5 file."""
    with h5py.File(path, "r") as f:
        return sorted(f["data"].keys())


def load_demo(path, demo_key="demo_0"):
    """Load one demo episode. Returns dict with numpy arrays."""
    with h5py.File(path, "r") as f:
        demo = f["data"][demo_key]
        obs = demo["obs"]

        def _get(grp, key):
            return grp[key][:] if key in grp else None

        N = obs["robot0_joint_pos"].shape[0]
        ts = obs["robot_ts"][:]
        t = ts - ts[0]  # relative time in seconds

        d = dict(
            N=N,
            t=t,
            robot_ts=ts,
            joint_pos=obs["robot0_joint_pos"][:],
            joint_vel=_get(obs, "robot0_joint_vel"),
            hand_left_qpos=_get(obs, "hand_left_qpos"),
            hand_right_qpos=_get(obs, "hand_right_qpos"),
            hand_left_cmd_qpos=_get(obs, "hand_left_cmd_qpos"),
            hand_right_cmd_qpos=_get(obs, "hand_right_cmd_qpos"),
            eef_right_proprio=_get(obs, "eef_right_proprio"),
            eef_left_proprio=_get(obs, "eef_left_proprio"),
            aria_image=_get(obs, "aria_image"),
        )

        if "actions" in demo:
            acts = demo["actions"]
            if isinstance(acts, h5py.Group):
                d["actions_joint"] = _get(acts, "joint")
                d["actions_eef"] = _get(acts, "eef")
            else:
                # Flat dataset — treat entire array as joint actions
                d["actions_joint"] = acts[:]

        return d
