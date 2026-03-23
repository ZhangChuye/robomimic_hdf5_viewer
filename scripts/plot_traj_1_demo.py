"""
Visualize 3D joint/EEF trajectories from one HDF5 demo in MuJoCo.

Shows the robot frozen at its initial pose with color-graded trajectory trails
(EEF-only by default; use --all_links for every arm link).

  - Trajectory: smaller spheres, thicker segment lines, semi-transparent colors
  - Robot: geom alpha reduced so trajs read on top (see --robot_alpha)
  - Right arm / EEF: green (start) → red (end); Left: cyan → purple
  - Start/end: slightly larger markers

Usage:
    python scripts/plot_traj_1_demo.py path/to/file.hdf5
    python scripts/plot_traj_1_demo.py path/to/file.hdf5 --demo_key demo_2 --subsample 2
    python scripts/plot_traj_1_demo.py path/to/file.hdf5 --no_interactive   # PNG only

    # For headless / SSH sessions, use EGL for offscreen rendering:
    MUJOCO_GL=egl python scripts/plot_traj_1_demo.py path/to/file.hdf5 --no_interactive

    # Use first HDF5 frame as static pose instead of SDK safe ready pose:
    python scripts/plot_traj_1_demo.py path/to/file.hdf5 --use_first_frame_pose
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import mujoco
import mujoco.viewer

from utils.hdf5_loader import load_demo
from utils.plot_action_traj_in_mujoco import (
    SAFE_POSE_ROBOT0_JOINT_26,
    compute_fk_trajectories,
    add_all_trajectories,
    set_robot_pose,
    save_all_views,
    push_robot_alpha,
    pop_robot_alpha,
)

_ASSETS = Path(__file__).parent.parent / "assets"
_XML = _ASSETS / "rby1_with_xhand" / "model_v1.3_xhand_act.xml"


def main():
    parser = argparse.ArgumentParser(
        description="Plot 3D joint/EEF trajectories from one HDF5 demo in MuJoCo"
    )
    parser.add_argument("hdf5_file", type=str, help="Path to the HDF5 file")
    parser.add_argument("--demo_key", type=str, default="demo_0")
    parser.add_argument("--xml", type=str, default=None, help="Override MuJoCo XML")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for saved PNGs (default: <hdf5_dir>/traj_viz/<stem>/<demo>)")
    parser.add_argument("--no_interactive", action="store_true",
                        help="Skip interactive viewer, only save PNGs")
    parser.add_argument("--subsample", type=int, default=3,
                        help="Draw every N-th frame (default 3)")
    parser.add_argument("--all_links", action="store_true",
                        help="Plot all link trajectories (default: EEF only)")
    parser.add_argument(
        "--robot_alpha",
        type=float,
        default=1.0,
        help="Multiply all geom alphas by this (0–1) for a slightly transparent robot (default 0.62)",
    )
    parser.add_argument(
        "--use_first_frame_pose",
        action="store_true",
        help="Freeze robot at first recorded joint_pos (default: safe ready pose from reset_safe_pose)",
    )
    args = parser.parse_args()

    # ── Load model ──
    xml_path = Path(args.xml) if args.xml else _XML
    if not xml_path.exists():
        print(f"Error: XML not found at {xml_path}")
        return
    print(f"Loading model: {xml_path.name}")
    model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
    data = mujoco.MjData(model)

    # ── Load demo ──
    demo = load_demo(args.hdf5_file, args.demo_key)
    N = demo["N"]
    print(f"Loaded {args.demo_key}: {N} frames, {demo['t'][-1]:.1f}s")

    # ── Compute FK trajectories ──
    print("Computing FK trajectories...")
    trajectories = compute_fk_trajectories(
        model, data, demo["joint_pos"],
        demo.get("hand_left_qpos"), demo.get("hand_right_qpos"),
    )
    if not args.all_links:
        trajectories = {k: v for k, v in trajectories.items() if "eef" in k}

    n_bodies = len(trajectories)
    print(f"  Plotting {n_bodies} {'bodies' if args.all_links else 'EEFs'} × {N} frames")

    # ── Freeze robot at static pose (default: SDK safe reset; optional: first HDF5 frame) ──
    hand_l0 = demo["hand_left_qpos"][0] if demo.get("hand_left_qpos") is not None else None
    hand_r0 = demo["hand_right_qpos"][0] if demo.get("hand_right_qpos") is not None else None
    if args.use_first_frame_pose:
        q_static = demo["joint_pos"][0].astype(np.float64, copy=False)
        pose_name = "first recorded frame"
    else:
        q_static = SAFE_POSE_ROBOT0_JOINT_26
        pose_name = "safe ready pose (reset_safe_pose)"
    print(f"  Static robot pose: {pose_name}")
    set_robot_pose(model, data, q_static, hand_l0, hand_r0)

    rgba_backup = push_robot_alpha(model, args.robot_alpha)
    try:
        # Offscreen Renderer.scene includes the robot after update_scene; user_scn does not.
        sub = args.subsample

        def populate_scene(scn, *, clear_scene: bool = True):
            add_all_trajectories(
                scn, trajectories, subsample=sub, clear_scene=clear_scene,
            )

        # ── Save PNGs ──
        hdf5_stem = Path(args.hdf5_file).stem
        out_dir = (
            Path(args.output_dir) if args.output_dir
            else Path(args.hdf5_file).parent / "traj_viz" / hdf5_stem / args.demo_key
        )
        print(f"Saving screenshots → {out_dir}/")
        save_all_views(
            model,
            data,
            lambda scn: populate_scene(scn, clear_scene=False),
            out_dir,
        )

        # ── Interactive viewer ──
        if not args.no_interactive:
            print("\nOpening interactive viewer (close window to exit)...")
            with mujoco.viewer.launch_passive(
                model=model, data=data,
                show_left_ui=False, show_right_ui=False,
            ) as viewer:
                viewer.cam.distance = 2.6
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -22
                viewer.cam.lookat[:] = [0.0, 0.0, 0.74]

                populate_scene(viewer.user_scn)
                viewer.sync()

                while viewer.is_running():
                    time.sleep(0.05)
    finally:
        pop_robot_alpha(model, rgba_backup)

    print("Done.")


if __name__ == "__main__":
    main()
