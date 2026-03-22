"""
Replay recorded HDF5 teleoperation data in MuJoCo viewer.
Self-contained: requires only h5py, numpy, and mujoco (no project dependencies).

Supports two replay modes:
  - obs:    kinematic replay from recorded joint positions (faithful to hardware)
  - action: physics-based replay from recorded action commands

Controls:
  SPACE  pause / resume
  R      restart from frame 0
  ]      speed up (x2)
  [      slow down (x0.5)
  .      step one frame (while paused)
  Q      quit

Author: Chuizheng Kong, Yunho Cho
"""

import argparse
import time
import numpy as np
from pathlib import Path

import h5py
import mujoco
import mujoco.viewer

_HERE = Path(__file__).parent
_ASSETS = _HERE.parent / "assets"
_XML = _ASSETS / "rby1_with_xhand" / "model_v1.3_xhand_act.xml"

# ── MuJoCo joint names (must match the XML) ──────────────────────────────────

TORSO_JOINTS = [f"torso_{i}" for i in range(6)]
RIGHT_ARM_JOINTS = [f"right_arm_{i}" for i in range(7)]
LEFT_ARM_JOINTS = [f"left_arm_{i}" for i in range(7)]
HEAD_JOINTS = ["head_0", "head_1"]

LEFT_HAND_JOINTS = [
    "left_hand_thumb_bend_joint", "left_hand_thumb_rota_joint1", "left_hand_thumb_rota_joint2",
    "left_hand_index_bend_joint", "left_hand_index_joint1", "left_hand_index_joint2",
    "left_hand_mid_joint1", "left_hand_mid_joint2",
    "left_hand_ring_joint1", "left_hand_ring_joint2",
    "left_hand_pinky_joint1", "left_hand_pinky_joint2",
]
RIGHT_HAND_JOINTS = [
    "right_hand_thumb_bend_joint", "right_hand_thumb_rota_joint1", "right_hand_thumb_rota_joint2",
    "right_hand_index_bend_joint", "right_hand_index_joint1", "right_hand_index_joint2",
    "right_hand_mid_joint1", "right_hand_mid_joint2",
    "right_hand_ring_joint1", "right_hand_ring_joint2",
    "right_hand_pinky_joint1", "right_hand_pinky_joint2",
]

# HDF5 robot0_joint_pos index ranges  (from meta_rby1_hdf5.json)
# base(4) + torso(6) + right_arm(7) + left_arm(7) + head(2) = 26
OBS_TORSO = slice(4, 10)
OBS_RIGHT_ARM = slice(10, 17)
OBS_LEFT_ARM = slice(17, 24)
OBS_HEAD = slice(24, 26)

# HDF5 actions/joint composition  (from meta_rby1_hdf5.json)
ACT_LEFT_ARM = slice(0, 7)
ACT_RIGHT_ARM = slice(7, 14)
ACT_TORSO = slice(14, 20)
ACT_HEAD = slice(20, 22)
ACT_LEFT_HAND = slice(25, 37)
ACT_RIGHT_HAND = slice(37, 49)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_qpos_addrs(model, joint_names):
    """Return list of qpos addresses for the given joint names."""
    addrs = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            print(f"  warning: joint '{name}' not found in model")
            addrs.append(None)
        else:
            addrs.append(model.jnt_qposadr[jid])
    return addrs


def _resolve_ctrl_ids(model, actuator_names):
    """Return list of ctrl indices for the given actuator names."""
    ids = []
    for name in actuator_names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            print(f"  warning: actuator '{name}' not found in model")
            ids.append(None)
        else:
            ids.append(aid)
    return ids


def _set_qpos(data, addrs, values):
    """Write `values` into data.qpos at the resolved addresses."""
    for addr, v in zip(addrs, values):
        if addr is not None:
            data.qpos[addr] = v


def _set_ctrl(data, ids, values):
    """Write `values` into data.ctrl at the resolved actuator ids."""
    for cid, v in zip(ids, values):
        if cid is not None:
            data.ctrl[cid] = v


# ── HDF5 loader ──────────────────────────────────────────────────────────────

class HDF5DemoData:
    """Load a single demo episode from the robomimic-style HDF5 file."""

    def __init__(self, hdf5_path, demo_key="demo_0"):
        self.path = Path(hdf5_path)
        if not self.path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.path}")

        with h5py.File(self.path, "r") as f:
            if "data" not in f:
                raise KeyError("HDF5 file has no 'data' group")
            if demo_key not in f["data"]:
                available = list(f["data"].keys())
                raise KeyError(
                    f"Demo '{demo_key}' not found. Available: {available}"
                )

            demo = f["data"][demo_key]
            obs = demo["obs"]

            self.num_samples = int(demo.attrs.get("num_samples", obs["robot0_joint_pos"].shape[0]))
            self.robot_pos = obs["robot0_joint_pos"][:]
            self.robot_ts = obs["robot_ts"][:]

            self.hand_left_qpos = obs["hand_left_qpos"][:] if "hand_left_qpos" in obs else None
            self.hand_right_qpos = obs["hand_right_qpos"][:] if "hand_right_qpos" in obs else None

            self.actions_joint = (
                demo["actions"]["joint"][:] if "actions" in demo and "joint" in demo["actions"]
                else None
            )

        if self.num_samples > 1:
            dt = np.diff(self.robot_ts)
            self.median_dt = float(np.median(dt[dt > 0])) if np.any(dt > 0) else 1.0 / 30.0
        else:
            self.median_dt = 1.0 / 30.0

        duration = self.median_dt * self.num_samples
        print(
            f"Loaded '{demo_key}' from {self.path.name}: "
            f"{self.num_samples} frames, "
            f"dt≈{self.median_dt * 1000:.1f}ms ({1.0 / self.median_dt:.1f}Hz), "
            f"duration≈{duration:.1f}s"
        )


# ── Keyboard callback ────────────────────────────────────────────────────────

class ReplayKeyCallback:
    def __init__(self):
        self.paused = False
        self.reset_requested = False
        self.speed_up = False
        self.speed_down = False
        self.step_forward = False
        self.quit_requested = False

    def __call__(self, key):
        if key == ord(" "):
            self.paused = not self.paused
            print("PAUSED" if self.paused else "PLAYING")
        elif key in (ord("r"), ord("R")):
            self.reset_requested = True
        elif key == ord("]"):
            self.speed_up = True
        elif key == ord("["):
            self.speed_down = True
        elif key == ord("."):
            self.step_forward = True
        elif key in (ord("q"), ord("Q")):
            self.quit_requested = True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Replay HDF5 teleoperation data in MuJoCo viewer"
    )
    parser.add_argument("hdf5_file", type=str, help="Path to the HDF5 file")
    parser.add_argument("--demo_key", type=str, default="demo_0", help="Demo key to replay")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--loop", action="store_true", help="Loop playback")
    parser.add_argument(
        "--mode", choices=["obs", "action"], default="obs",
        help="'obs' = kinematic replay from recorded positions (default); "
             "'action' = physics replay from recorded action commands",
    )
    parser.add_argument("--xml", type=str, default=None, help="Override MuJoCo XML model path")
    args = parser.parse_args()

    # ── Load MuJoCo model ──
    xml_path = Path(args.xml) if args.xml else _XML
    if not xml_path.exists():
        print(f"Error: MuJoCo XML not found at {xml_path}")
        return
    print(f"Loading MuJoCo model: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
    data = mujoco.MjData(model)
    model.opt.timestep = 0.005

    # ── Load demo data ──
    demo = HDF5DemoData(args.hdf5_file, args.demo_key)
    if args.mode == "action" and demo.actions_joint is None:
        print("Warning: actions/joint not found in HDF5, falling back to obs mode")
        args.mode = "obs"

    # ── Build joint / actuator mappings ──
    print("Resolving joint and actuator mappings...")
    qpos_torso = _resolve_qpos_addrs(model, TORSO_JOINTS)
    qpos_right_arm = _resolve_qpos_addrs(model, RIGHT_ARM_JOINTS)
    qpos_left_arm = _resolve_qpos_addrs(model, LEFT_ARM_JOINTS)
    qpos_head = _resolve_qpos_addrs(model, HEAD_JOINTS)
    qpos_left_hand = _resolve_qpos_addrs(model, LEFT_HAND_JOINTS)
    qpos_right_hand = _resolve_qpos_addrs(model, RIGHT_HAND_JOINTS)

    # Actuator names: for hands the actuator name == joint name
    ctrl_torso = _resolve_ctrl_ids(model, [f"link{i+1}_act" for i in range(6)])
    ctrl_left_arm = _resolve_ctrl_ids(model, [f"left_arm_{i+1}_act" for i in range(7)])
    ctrl_right_arm = _resolve_ctrl_ids(model, [f"right_arm_{i+1}_act" for i in range(7)])
    ctrl_head = _resolve_ctrl_ids(model, ["head_0_act", "head_1_act"])
    ctrl_left_hand = _resolve_ctrl_ids(model, LEFT_HAND_JOINTS)
    ctrl_right_hand = _resolve_ctrl_ids(model, RIGHT_HAND_JOINTS)

    # ── Viewer ──
    key_cb = ReplayKeyCallback()
    speed = args.speed
    frame_idx = 0
    finished = False

    def apply_obs_frame(idx):
        """Set MuJoCo qpos from recorded observations (kinematic)."""
        pos = demo.robot_pos[idx]
        _set_qpos(data, qpos_torso, pos[OBS_TORSO])
        _set_qpos(data, qpos_right_arm, pos[OBS_RIGHT_ARM])
        _set_qpos(data, qpos_left_arm, pos[OBS_LEFT_ARM])
        _set_qpos(data, qpos_head, pos[OBS_HEAD])

        if demo.hand_left_qpos is not None:
            _set_qpos(data, qpos_left_hand, demo.hand_left_qpos[idx])
        if demo.hand_right_qpos is not None:
            _set_qpos(data, qpos_right_hand, demo.hand_right_qpos[idx])

        mujoco.mj_forward(model, data)

    def apply_action_frame(idx):
        """Set MuJoCo ctrl from recorded actions, then step physics."""
        act = demo.actions_joint[idx]
        _set_ctrl(data, ctrl_left_arm, act[ACT_LEFT_ARM])
        _set_ctrl(data, ctrl_right_arm, act[ACT_RIGHT_ARM])
        _set_ctrl(data, ctrl_torso, act[ACT_TORSO])
        _set_ctrl(data, ctrl_head, act[ACT_HEAD])
        _set_ctrl(data, ctrl_left_hand, act[ACT_LEFT_HAND])
        _set_ctrl(data, ctrl_right_hand, act[ACT_RIGHT_HAND])

        steps = max(1, int(demo.median_dt / model.opt.timestep))
        for _ in range(steps):
            mujoco.mj_step(model, data)

    apply_frame = apply_obs_frame if args.mode == "obs" else apply_action_frame

    print(
        f"\nReplay mode: {args.mode} | speed: {speed:.1f}x | "
        f"frames: {demo.num_samples} | loop: {args.loop}"
    )
    print("Controls: SPACE=pause  R=restart  [/]=speed  .=step  Q=quit\n")

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_cb,
    ) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -15
        viewer.cam.lookat[:] = [0, 0, 1.0]

        apply_frame(0)
        viewer.sync()

        while viewer.is_running():
            loop_start = time.time()

            # ── Handle key events ──
            if key_cb.quit_requested:
                break

            if key_cb.reset_requested:
                key_cb.reset_requested = False
                frame_idx = 0
                finished = False
                if args.mode == "action":
                    mujoco.mj_resetData(model, data)
                print(f"Restarted (frame 0/{demo.num_samples})")

            if key_cb.speed_up:
                key_cb.speed_up = False
                speed = min(speed * 2.0, 32.0)
                print(f"Speed: {speed:.2f}x")

            if key_cb.speed_down:
                key_cb.speed_down = False
                speed = max(speed / 2.0, 0.0625)
                print(f"Speed: {speed:.2f}x")

            # ── Advance frame ──
            advance = False
            if key_cb.step_forward:
                key_cb.step_forward = False
                advance = True
            elif not key_cb.paused and not finished:
                advance = True

            if advance and frame_idx < demo.num_samples:
                apply_frame(frame_idx)

                if frame_idx % 100 == 0 or frame_idx == demo.num_samples - 1:
                    pct = 100.0 * frame_idx / max(demo.num_samples - 1, 1)
                    t = demo.median_dt * frame_idx
                    print(
                        f"\rFrame {frame_idx:>5d}/{demo.num_samples}  "
                        f"({pct:5.1f}%)  t={t:.2f}s  speed={speed:.2f}x",
                        end="", flush=True,
                    )

                frame_idx += 1

                if frame_idx >= demo.num_samples:
                    if args.loop:
                        frame_idx = 0
                        if args.mode == "action":
                            mujoco.mj_resetData(model, data)
                        print("\n── Looping ──")
                    else:
                        finished = True
                        print("\n── Playback finished (press R to restart) ──")

            viewer.sync()

            # ── Timing ──
            elapsed = time.time() - loop_start
            target_dt = demo.median_dt / speed if speed > 0 else demo.median_dt
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    print("\nDone.")


if __name__ == "__main__":
    main()
