The script/format for using the collected hdf5 files is as follows:
```python
"""
Teleop data collection: ring buffer + pluggable dataset writers (e.g. roboMimic HDF5).
Collector holds only in-memory buffer; format-specific saving is delegated to DatasetWriter.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import h5py
import numpy as np
import cv2

# --- Episode data (format-agnostic) ---
# TODO: add async data saving support

ROBOT_POS_DIM = 26   # Model-M: base 4, torso 6, right 7, left 7, head 2
ROBOT_VEL_DIM = 26
HAND_QPOS_DIM = 12  # XHand
HAND_FINGER_DIM = 12
HAND_TACTILE_FORCE_DIM = 5 * 120 * 3
HAND_TACTILE_TEMP_DIM = 5 * 20
Q_GOAL_LEFT_DIM = 7
Q_GOAL_RIGHT_DIM = 7
Q_GOAL_TORSO_DIM = 6
Q_GOAL_HEAD_DIM = 2
Q_EEF_DIM = 4 * 4  # 4x4 transformation matrix for proprio and command
BASE_CMD_DIM = 3     # goal_x, goal_y, goal_yaw
ARIA_RESOLUTION= (768, 768)


@dataclass
class EpisodeData:
    """One episode of teleop data; all arrays have same length (num_samples)."""
    num_samples: int
    robot_pos: np.ndarray      # (N, ROBOT_POS_DIM)
    robot_vel: np.ndarray      # (N, ROBOT_VEL_DIM)
    robot_ts: np.ndarray       # (N,)
    hand_left_qpos: np.ndarray  # (N, HAND_QPOS_DIM)
    hand_right_qpos: np.ndarray
    hand_ts_left: np.ndarray   # (N,)
    hand_ts_right: np.ndarray
    hand_left_finger_pos: np.ndarray
    hand_right_finger_pos: np.ndarray
    hand_left_cmd_qpos: np.ndarray
    hand_right_cmd_qpos: np.ndarray
    hand_left_finger_raw_pos: np.ndarray
    hand_right_finger_raw_pos: np.ndarray
    hand_left_tactile_force: np.ndarray
    hand_right_tactile_force: np.ndarray
    hand_left_tactile_temp: np.ndarray
    hand_right_tactile_temp: np.ndarray
    # Actions (what was commanded)
    q_goal_left: np.ndarray   # (N, 7)
    q_goal_right: np.ndarray  # (N, 7)
    q_goal_torso: np.ndarray  # (N, 6)
    q_goal_head: np.ndarray   # (N, 2)
    eef_right_proprio: np.ndarray  # (N, 16) 4x4 matrix flattened
    eef_left_proprio: np.ndarray   # (N, 16)
    eef_right_cmd: np.ndarray      # (N, 16)
    eef_left_cmd: np.ndarray       # (N, 16)
    delta_x: np.ndarray        # (N,)
    delta_y: np.ndarray
    delta_yaw: np.ndarray
    cmd_ts: np.ndarray        # (N,)
    aria_ts: np.ndarray        # (N,) or zeros if not streaming
    aria_image: Optional[np.ndarray] = None  # (N, H, W, 3) uint8 or None


# --- DatasetWriter protocol ---

@runtime_checkable
class DatasetWriter(Protocol):
    """Protocol for writing an episode to a dataset file. Other formats can implement this."""

    def write_episode(self, demo_key: str, data: EpisodeData, path: Path | str) -> None:
        """Write one episode to the given path. May append (e.g. HDF5) or overwrite."""
        ...


# --- Robomimic-style HDF5 writer ---

DEFAULT_ENV_ARGS = {
    "env_name": "RBY1Teleop",
    "env_type": "robosuite",
    "env_kwargs": {},
}


class RobomimicHDF5Writer:
    """Writes EpisodeData to HDF5 in roboMimic-like layout: data/demo_k/obs/, actions, etc."""

    def __init__(self, env_args: Optional[dict] = None):
        self.env_args = env_args or DEFAULT_ENV_ARGS

    def write_episode(self, demo_key: str, data: EpisodeData, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if path.exists() else "w"
        with h5py.File(path, mode) as f:
            if "data" not in f:
                data_grp = f.create_group("data")
                data_grp.attrs["total"] = 0
                data_grp.attrs["env_args"] = json.dumps(self.env_args)
            else:
                data_grp = f["data"]

            n = data.num_samples
            demo_grp = data_grp.create_group(demo_key)
            demo_grp.attrs["num_samples"] = n

            obs = demo_grp.create_group("obs")
            obs.create_dataset("robot0_joint_pos", data=data.robot_pos.astype(np.float32))
            obs.create_dataset("robot0_joint_vel", data=data.robot_vel.astype(np.float32))
            obs.create_dataset("robot_ts", data=data.robot_ts.astype(np.float64))
            obs.create_dataset("hand_left_qpos", data=data.hand_left_qpos.astype(np.float32))
            obs.create_dataset("hand_right_qpos", data=data.hand_right_qpos.astype(np.float32))
            obs.create_dataset("hand_left_cmd_qpos", data=data.hand_left_cmd_qpos.astype(np.float32))
            obs.create_dataset("hand_right_cmd_qpos", data=data.hand_right_cmd_qpos.astype(np.float32))
            obs.create_dataset(
                "hand_left_finger_pos", data=data.hand_left_finger_pos.astype(np.float32)
            )
            obs.create_dataset(
                "hand_right_finger_pos", data=data.hand_right_finger_pos.astype(np.float32)
            )
            obs.create_dataset(
                "hand_left_finger_raw_pos", data=data.hand_left_finger_raw_pos.astype(np.float32)
            )
            obs.create_dataset(
                "hand_right_finger_raw_pos", data=data.hand_right_finger_raw_pos.astype(np.float32)
            )
            obs.create_dataset(
                "hand_left_tactile_force", data=data.hand_left_tactile_force.astype(np.float32)
            )
            obs.create_dataset(
                "hand_right_tactile_force", data=data.hand_right_tactile_force.astype(np.float32)
            )
            obs.create_dataset(
                "hand_left_tactile_temp", data=data.hand_left_tactile_temp.astype(np.float32)
            )
            obs.create_dataset(
                "hand_right_tactile_temp", data=data.hand_right_tactile_temp.astype(np.float32)
            )
            obs.create_dataset("eef_right_proprio", data=data.eef_right_proprio.astype(np.float32))
            obs.create_dataset("eef_left_proprio", data=data.eef_left_proprio.astype(np.float32))
            # obs.create_dataset("eef_right_cmd", data=data.eef_right_cmd.astype(np.float32))
            # obs.create_dataset("eef_left_cmd", data=data.eef_left_cmd.astype(np.float32))
            obs.create_dataset("hand_ts_left", data=data.hand_ts_left.astype(np.float64))
            obs.create_dataset("hand_ts_right", data=data.hand_ts_right.astype(np.float64))
            obs.create_dataset("aria_ts", data=data.aria_ts.astype(np.float64))
            obs.create_dataset("cmd_ts", data=data.cmd_ts.astype(np.float64))
            if data.aria_image is not None:
                obs.create_dataset("aria_image", data=data.aria_image, compression="gzip")

            # actions_joint: (N, 49) = 7+7+6+2+1+1+1+12+12
            actions_joint = np.hstack([
                data.q_goal_left,       # (N, 7)
                data.q_goal_right,      # (N, 7)
                data.q_goal_torso,      # (N, 6)
                data.q_goal_head,       # (N, 2)
                data.delta_x.reshape(-1, 1),   # (N, 1)
                data.delta_y.reshape(-1, 1),   # (N, 1)
                data.delta_yaw.reshape(-1, 1), # (N, 1)
                data.hand_left_cmd_qpos,    # (N, 12)
                data.hand_right_cmd_qpos,   # (N, 12)
            ])
            
            actions_arm = np.hstack([
                data.q_goal_left,       # (N, 7)
                data.q_goal_right,      # (N, 7)
            ])
            
            actions_arm_hand = np.hstack([
                data.q_goal_left,       # (N, 7)
                data.q_goal_right,      # (N, 7)
                data.hand_left_cmd_qpos,    # (N, 12)
                data.hand_right_cmd_qpos,   # (N, 12)
            ])
            
            actions_arm_head_torso_base = np.hstack([
                data.q_goal_left,       # (N, 7)
                data.q_goal_right,      # (N, 7)
                data.q_goal_torso,      # (N, 6)
                data.q_goal_head,       # (N, 2)
                data.delta_x.reshape(-1, 1),   # (N, 1)
                data.delta_y.reshape(-1, 1),   # (N, 1)
                data.delta_yaw.reshape(-1, 1), # (N, 1)
            ])
            
            # action_eef: (N, 56) = 16+16+12+12
            # TODO: don't store transformation matrix, using p and axis angle (or half rotation)
            action_eef = np.hstack([
                data.eef_right_cmd,  # (N, 16)
                data.eef_left_cmd,   # (N, 16)
                data.hand_left_cmd_qpos, # (N, 12)
                data.hand_right_cmd_qpos, # (N, 12)
            ])
            
                
            act = demo_grp.create_group("actions")
            act.create_dataset("joint", data=actions_joint.astype(np.float32))
            act.create_dataset("eef", data=action_eef.astype(np.float32))
            act.create_dataset("joint_arm", data=actions_arm.astype(np.float32))
            act.create_dataset("joint_arm_hand", data=actions_arm_hand.astype(np.float32))
            act.create_dataset("joint_arm_head_torso_base", data=actions_arm_head_torso_base.astype(np.float32))
            # demo_grp.create_dataset("actions", data=actions.astype(np.float32))
            demo_grp.create_dataset("rewards", data=np.zeros(n, dtype=np.float32))
            demo_grp.create_dataset("dones", data=np.zeros(n, dtype=np.float32))

            total = data_grp.attrs.get("total", 0) + n
            data_grp.attrs["total"] = int(total)

        print(f"\033[92m[RobomimicHDF5Writer] Saved {demo_key} ({n} samples) to {path}\033[0m")


# --- TeleopDataCollector (ring buffer only) ---

def _to_f32_vec(x: Optional[np.ndarray | list], dim: int) -> np.ndarray:
    if x is None or (isinstance(x, (list, np.ndarray)) and len(x) == 0):
        return np.zeros(dim, dtype=np.float32)
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(arr) >= dim:
        return arr[:dim].copy()
    out = np.zeros(dim, dtype=np.float32)
    out[: len(arr)] = arr
    return out


class TeleopDataCollector:
    """
    Ring buffer for one episode. Records robot state, hand command, Aria (optional), and actions.
    Flush builds EpisodeData and delegates to a DatasetWriter (e.g. RobomimicHDF5Writer).
    TODO: add support for streaming data saving
    """

    def __init__(
        self,
        writer: DatasetWriter,
        path: Optional[Path | str] = None,
        buffer_size: int = 60 * 100,
        save_in_background: bool = True,
        aria_resolution: Optional[tuple] = (640, 640),
    ):
        self.writer = writer
        self.default_path = Path(path) if path else None
        self.buffer_size = buffer_size
        self.save_in_background = save_in_background
        self.head = 0
        self._lock = threading.Lock()

        n = buffer_size
        self._robot_pos = np.zeros((n, ROBOT_POS_DIM), dtype=np.float32)
        self._robot_vel = np.zeros((n, ROBOT_VEL_DIM), dtype=np.float32)
        self._robot_ts = np.zeros(n, dtype=np.float64)
        self._hand_left_qpos = np.zeros((n, HAND_QPOS_DIM), dtype=np.float32)
        self._hand_right_qpos = np.zeros((n, HAND_QPOS_DIM), dtype=np.float32)
        self._hand_left_cmd_qpos = np.zeros((n, HAND_QPOS_DIM), dtype=np.float32)
        self._hand_right_cmd_qpos = np.zeros((n, HAND_QPOS_DIM), dtype=np.float32)
        self._hand_ts_left = np.zeros(n, dtype=np.float64)
        self._hand_ts_right = np.zeros(n, dtype=np.float64)
        self._hand_left_finger_pos = np.zeros((n, HAND_FINGER_DIM), dtype=np.float32)
        self._hand_right_finger_pos = np.zeros((n, HAND_FINGER_DIM), dtype=np.float32)
        self._hand_left_finger_raw_pos = np.zeros((n, HAND_FINGER_DIM), dtype=np.float32)
        self._hand_right_finger_raw_pos = np.zeros((n, HAND_FINGER_DIM), dtype=np.float32)
        self._hand_left_tactile_force = np.zeros((n, HAND_TACTILE_FORCE_DIM), dtype=np.float32)
        self._hand_right_tactile_force = np.zeros((n, HAND_TACTILE_FORCE_DIM), dtype=np.float32)
        self._hand_left_tactile_temp = np.zeros((n, HAND_TACTILE_TEMP_DIM), dtype=np.float32)
        self._hand_right_tactile_temp = np.zeros((n, HAND_TACTILE_TEMP_DIM), dtype=np.float32)
        self._aria_ts = np.zeros(n, dtype=np.float64)
        self._aria_image: Optional[np.ndarray] = None
        self._q_goal_left = np.zeros((n, Q_GOAL_LEFT_DIM), dtype=np.float32)
        self._q_goal_right = np.zeros((n, Q_GOAL_RIGHT_DIM), dtype=np.float32)
        self._q_goal_torso = np.zeros((n, Q_GOAL_TORSO_DIM), dtype=np.float32)
        self._q_goal_head = np.zeros((n, Q_GOAL_HEAD_DIM), dtype=np.float32)
        self._eef_right_proprio = np.zeros((n, Q_EEF_DIM), dtype=np.float32)
        self._eef_left_proprio = np.zeros((n, Q_EEF_DIM), dtype=np.float32)
        self._eef_right_cmd = np.zeros((n, Q_EEF_DIM), dtype=np.float32)
        self._eef_left_cmd = np.zeros((n, Q_EEF_DIM), dtype=np.float32)
        self._delta_x = np.zeros(n, dtype=np.float32)
        self._delta_y = np.zeros(n, dtype=np.float32)
        self._delta_yaw = np.zeros(n, dtype=np.float32)
        self._cmd_ts = np.zeros(n, dtype=np.float64)
        self._has_aria_image = False
        self._aria_image_shape: Optional[tuple] = None
        self._aria_resolution = aria_resolution

    @property
    def sample_count(self) -> int:
        """Current number of samples in the buffer (for debugging)."""
        with self._lock:
            return min(self.head, self.buffer_size)

    def start_episode(self, aria_image_shape: Optional[tuple] = None) -> None:
        """Reset buffer for a new episode. If aria_image will be provided, pass (H, W, 3)."""
        with self._lock:
            self.head = 0
            self._has_aria_image = aria_image_shape is not None
            self._aria_image_shape = aria_image_shape
            if aria_image_shape is not None and self._aria_image is None:
                n = self.buffer_size
                H, W, C = aria_image_shape
                H, W = self._aria_resolution if self._aria_resolution is not None else (H, W)
                self._aria_image = np.zeros((n, H, W, C), dtype=np.uint8)

    def record(
        self,
        robot_pos: np.ndarray,
        robot_vel: np.ndarray,
        robot_ts: float,
        q_goal_left: np.ndarray,
        q_goal_right: np.ndarray,
        q_goal_torso: Optional[np.ndarray],
        q_goal_head: np.ndarray,
        eef_right_proprio: np.ndarray,
        eef_left_proprio: np.ndarray,
        eef_right_cmd: np.ndarray,
        eef_left_cmd: np.ndarray,
        delta_x: float,
        delta_y: float,
        delta_yaw: float,
        cmd_ts: float,
        hand_left_qpos: np.ndarray | list,
        hand_right_qpos: np.ndarray | list,
        hand_left_cmd_qpos: np.ndarray | list,
        hand_right_cmd_qpos: np.ndarray | list,
        hand_ts_left: float,
        hand_ts_right: float,
        hand_left_finger_pos: Optional[np.ndarray | list] = None,
        hand_right_finger_pos: Optional[np.ndarray | list] = None,
        hand_left_finger_raw_pos: Optional[np.ndarray | list] = None,
        hand_right_finger_raw_pos: Optional[np.ndarray | list] = None,
        hand_left_tactile_force: Optional[np.ndarray | list] = None,
        hand_right_tactile_force: Optional[np.ndarray | list] = None,
        hand_left_tactile_temp: Optional[np.ndarray | list] = None,
        hand_right_tactile_temp: Optional[np.ndarray | list] = None,
        aria_ts: Optional[float] = None,
        aria_image: Optional[np.ndarray] = None,
    ) -> None:
        """Append one sample into the ring buffer. Call from main loop only."""
        with self._lock:
            idx = self.head % self.buffer_size
            self._robot_pos[idx] = _to_f32_vec(robot_pos, ROBOT_POS_DIM)
            self._robot_vel[idx] = _to_f32_vec(robot_vel, ROBOT_VEL_DIM)
            self._robot_ts[idx] = float(robot_ts)
            self._hand_left_qpos[idx] = _to_f32_vec(hand_left_qpos, HAND_QPOS_DIM)
            self._hand_right_qpos[idx] = _to_f32_vec(hand_right_qpos, HAND_QPOS_DIM)
            self._hand_left_cmd_qpos[idx] = _to_f32_vec(hand_left_cmd_qpos, HAND_QPOS_DIM)
            self._hand_right_cmd_qpos[idx] = _to_f32_vec(hand_right_cmd_qpos, HAND_QPOS_DIM)
            self._hand_ts_left[idx] = float(hand_ts_left)
            self._hand_ts_right[idx] = float(hand_ts_right)
            self._hand_left_finger_pos[idx] = _to_f32_vec(hand_left_finger_pos, HAND_FINGER_DIM)
            self._hand_right_finger_pos[idx] = _to_f32_vec(hand_right_finger_pos, HAND_FINGER_DIM)
            self._hand_left_finger_raw_pos[idx] = _to_f32_vec(
                hand_left_finger_raw_pos, HAND_FINGER_DIM
            )
            self._hand_right_finger_raw_pos[idx] = _to_f32_vec(
                hand_right_finger_raw_pos, HAND_FINGER_DIM
            )
            self._hand_left_tactile_force[idx] = _to_f32_vec(
                hand_left_tactile_force, HAND_TACTILE_FORCE_DIM
            )
            self._hand_right_tactile_force[idx] = _to_f32_vec(
                hand_right_tactile_force, HAND_TACTILE_FORCE_DIM
            )
            self._hand_left_tactile_temp[idx] = _to_f32_vec(
                hand_left_tactile_temp, HAND_TACTILE_TEMP_DIM
            )
            self._hand_right_tactile_temp[idx] = _to_f32_vec(
                hand_right_tactile_temp, HAND_TACTILE_TEMP_DIM
            )
            self._aria_ts[idx] = float(aria_ts) if aria_ts is not None else 0.0
            self._q_goal_left[idx] = _to_f32_vec(q_goal_left, Q_GOAL_LEFT_DIM)
            self._q_goal_right[idx] = _to_f32_vec(q_goal_right, Q_GOAL_RIGHT_DIM)
            self._q_goal_torso[idx] = _to_f32_vec(q_goal_torso, Q_GOAL_TORSO_DIM)
            self._q_goal_head[idx] = _to_f32_vec(q_goal_head, Q_GOAL_HEAD_DIM)
            self._eef_right_proprio[idx] = _to_f32_vec(eef_right_proprio, Q_EEF_DIM)
            self._eef_left_proprio[idx] = _to_f32_vec(eef_left_proprio, Q_EEF_DIM)
            self._eef_right_cmd[idx] = _to_f32_vec(eef_right_cmd, Q_EEF_DIM)
            self._eef_left_cmd[idx] = _to_f32_vec(eef_left_cmd, Q_EEF_DIM)
            self._delta_x[idx] = float(delta_x)
            self._delta_y[idx] = float(delta_y)
            self._delta_yaw[idx] = float(delta_yaw)
            self._cmd_ts[idx] = float(cmd_ts)
            if aria_image is not None and self._aria_image is not None:
                aria_img_uint8 = np.asarray(aria_image, dtype=np.uint8)
                resized_img = cv2.resize(
                    aria_img_uint8,
                    (self._aria_image.shape[2], self._aria_image.shape[1]),  # (W, H)
                    interpolation=cv2.INTER_LINEAR,
                )
                self._aria_image[idx] = resized_img
            self.head += 1

    def _build_episode_data(self) -> EpisodeData:
        with self._lock:
            n = min(self.head, self.buffer_size)
            if n == 0:
                raise ValueError(
                    "No samples in buffer. "
                    "Ensure start_episode() was called and record() is invoked each loop. "
                    "Check record() uses keyword args: goal_x, goal_y, goal_yaw (not delta_*)."
                )
            start = (self.head - n) % self.buffer_size if self.head >= self.buffer_size else 0
            idx = (np.arange(n) + start) % self.buffer_size

            def slice_buf(arr: np.ndarray) -> np.ndarray:
                if arr.ndim == 1:
                    return arr[idx].copy()
                return arr[idx].copy()

            aria_img: Optional[np.ndarray] = None
            if self._aria_image is not None and self._has_aria_image:
                aria_img = slice_buf(self._aria_image)

            return EpisodeData(
                num_samples=n,
                robot_pos=slice_buf(self._robot_pos),
                robot_vel=slice_buf(self._robot_vel),
                robot_ts=slice_buf(self._robot_ts),
                hand_left_qpos=slice_buf(self._hand_left_qpos),
                hand_right_qpos=slice_buf(self._hand_right_qpos),
                hand_left_cmd_qpos=slice_buf(self._hand_left_cmd_qpos),
                hand_right_cmd_qpos=slice_buf(self._hand_right_cmd_qpos),
                hand_ts_left=slice_buf(self._hand_ts_left),
                hand_ts_right=slice_buf(self._hand_ts_right),
                hand_left_finger_pos=slice_buf(self._hand_left_finger_pos),
                hand_right_finger_pos=slice_buf(self._hand_right_finger_pos),
                hand_left_finger_raw_pos=slice_buf(self._hand_left_finger_raw_pos),
                hand_right_finger_raw_pos=slice_buf(self._hand_right_finger_raw_pos),
                hand_left_tactile_force=slice_buf(self._hand_left_tactile_force),
                hand_right_tactile_force=slice_buf(self._hand_right_tactile_force),
                hand_left_tactile_temp=slice_buf(self._hand_left_tactile_temp),
                hand_right_tactile_temp=slice_buf(self._hand_right_tactile_temp),
                aria_ts=slice_buf(self._aria_ts),
                aria_image=aria_img,
                q_goal_left=slice_buf(self._q_goal_left),
                q_goal_right=slice_buf(self._q_goal_right),
                q_goal_torso=slice_buf(self._q_goal_torso),
                q_goal_head=slice_buf(self._q_goal_head),
                eef_right_proprio=slice_buf(self._eef_right_proprio),
                eef_left_proprio=slice_buf(self._eef_left_proprio),
                eef_right_cmd=slice_buf(self._eef_right_cmd),
                eef_left_cmd=slice_buf(self._eef_left_cmd),
                delta_x=slice_buf(self._delta_x),
                delta_y=slice_buf(self._delta_y),
                delta_yaw=slice_buf(self._delta_yaw),
                cmd_ts=slice_buf(self._cmd_ts),
            )

    def flush_and_save(self, demo_key: str, path: Optional[Path | str] = None) -> None:
        """Build EpisodeData from current buffer and call writer. Optionally run in background."""
        out_path = Path(path) if path else self.default_path
        if out_path is None:
            raise ValueError("No path provided to flush_and_save and no default_path set")
        data = self._build_episode_data()

        def _write():
            self.writer.write_episode(demo_key, data, out_path)

        if self.save_in_background:
            t = threading.Thread(target=_write, daemon=True)
            t.start()
        else:
            _write()
            
    def fail_demo(self) -> None:
        """Discard the current demo completely and reset the in-memory buffer."""
        with self._lock:
            self.head = 0

            self._robot_pos.fill(0)
            self._robot_vel.fill(0)
            self._robot_ts.fill(0)

            self._hand_left_qpos.fill(0)
            self._hand_right_qpos.fill(0)
            self._hand_left_cmd_qpos.fill(0)
            self._hand_right_cmd_qpos.fill(0)
            self._hand_ts_left.fill(0)
            self._hand_ts_right.fill(0)

            self._hand_left_finger_pos.fill(0)
            self._hand_right_finger_pos.fill(0)
            self._hand_left_finger_raw_pos.fill(0)
            self._hand_right_finger_raw_pos.fill(0)

            self._hand_left_tactile_force.fill(0)
            self._hand_right_tactile_force.fill(0)
            self._hand_left_tactile_temp.fill(0)
            self._hand_right_tactile_temp.fill(0)

            self._aria_ts.fill(0)
            if self._aria_image is not None:
                self._aria_image.fill(0)

            self._q_goal_left.fill(0)
            self._q_goal_right.fill(0)
            self._q_goal_torso.fill(0)
            self._q_goal_head.fill(0)

            self._eef_right_proprio.fill(0)
            self._eef_left_proprio.fill(0)
            self._eef_right_cmd.fill(0)
            self._eef_left_cmd.fill(0)

            self._delta_x.fill(0)
            self._delta_y.fill(0)
            self._delta_yaw.fill(0)
            self._cmd_ts.fill(0)

            self._has_aria_image = False
            self._aria_image_shape = None
        
            
```