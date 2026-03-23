"""
Headless MP4 render: replay each HDF5 demo in MuJoCo with a **growing EEF trajectory**
overlay (same FK / colors as ``plot_traj_1_demo.py`` / ``replay_hdf5_mujoco.py``).

Output (under ``DATA_DIR`` by default)::

    replay_mp4/<hdf5_stem>/<demo_key>/replay_<view>.mp4

Requires ``MUJOCO_GL=egl`` (or osmesa) for offscreen rendering and OpenCV for video encoding.

Trajectory geoms are **appended** after ``Renderer.update_scene`` (``clear_scene=False``) so the
robot mesh is visible, matching ``plot_traj_1_demo`` PNGs.

Usage::

    export MUJOCO_GL=egl
    python scripts/render_traj_mp4_for_all.py /path/to/folder_with_hdf5
    python scripts/render_traj_mp4_for_all.py /path/to/data --all_views --fps 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco

from utils.hdf5_loader import list_demos, load_demo
from utils.plot_action_traj_in_mujoco import (
    CAMERA_VIEWS,
    add_all_trajectories,
    compute_fk_trajectories,
    make_camera,
    pop_robot_alpha,
    push_robot_alpha,
    set_robot_pose,
)

_ASSETS = Path(__file__).parent.parent / "assets"
_XML = _ASSETS / "rby1_with_xhand" / "model_v1.3_xhand_act.xml"


def _median_dt(robot_ts: np.ndarray) -> float:
    if robot_ts.size < 2:
        return 1.0 / 30.0
    dt = np.diff(robot_ts)
    return float(np.median(dt[dt > 0])) if np.any(dt > 0) else 1.0 / 30.0


def render_one_demo_mp4(
    model,
    data,
    demo: dict,
    trajectories: dict,
    out_path: Path,
    cam_config: dict,
    fps: float,
    width: int,
    height: int,
    subsample: int,
    robot_alpha: float,
    max_geom: int,
) -> None:
    import cv2

    out_path.parent.mkdir(parents=True, exist_ok=True)

    rgba_backup = push_robot_alpha(model, robot_alpha)
    orig_w, orig_h = model.vis.global_.offwidth, model.vis.global_.offheight
    model.vis.global_.offwidth = max(orig_w, width)
    model.vis.global_.offheight = max(orig_h, height)

    cam = make_camera(cam_config)
    renderer = mujoco.Renderer(model, height=height, width=width, max_geom=max_geom)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(out_path),
        fourcc,
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        print(f"  ERROR: could not open VideoWriter for {out_path}")
        pop_robot_alpha(model, rgba_backup)
        renderer.close()
        return

    N = demo["N"]
    joint_pos = demo["joint_pos"]
    hl = demo.get("hand_left_qpos")
    hr = demo.get("hand_right_qpos")

    try:
        for i in range(N):
            hl_i = hl[i] if hl is not None else None
            hr_i = hr[i] if hr is not None else None
            set_robot_pose(model, data, joint_pos[i], hl_i, hr_i)

            renderer.update_scene(data, camera=cam)
            # Keep model geoms from update_scene; only append trajectory overlay geoms
            add_all_trajectories(
                renderer.scene,
                trajectories,
                subsample=subsample,
                end_idx=i,
                clear_scene=False,
            )
            rgb = renderer.render()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()
        renderer.close()
        model.vis.global_.offwidth = orig_w
        model.vis.global_.offheight = orig_h
        pop_robot_alpha(model, rgba_backup)

    print(f"  Wrote {out_path}  ({N} frames @ {fps:.1f} fps)")


def main():
    parser = argparse.ArgumentParser(
        description="Render replay MP4s with EEF trajectory for every HDF5 demo in a folder"
    )
    parser.add_argument("data_dir", type=str, help="Folder containing .hdf5 files")
    parser.add_argument("--xml", type=str, default=None, help="Override MuJoCo XML path")
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Root directory for replay_mp4/... (default: same as data_dir)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="front",
        choices=list(CAMERA_VIEWS.keys()),
        help="Camera preset when not using --all_views (default: front)",
    )
    parser.add_argument(
        "--all_views",
        action="store_true",
        help="Render one MP4 per camera in CAMERA_VIEWS",
    )
    parser.add_argument("--fps", type=float, default=None, help="Override FPS (default: from robot_ts)")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--subsample", type=int, default=3)
    parser.add_argument(
        "--robot_alpha",
        type=float,
        default=1.0,
        help="Geom alpha scale so trajectories read over the robot (same default as plot_traj_1_demo)",
    )
    parser.add_argument("--max_geom", type=int, default=20000)
    parser.add_argument("--all_links", action="store_true", help="Overlay all arm links, not only EEFs")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"Not a directory: {data_dir}")
        return

    xml_path = Path(args.xml) if args.xml else _XML
    if not xml_path.exists():
        print(f"XML not found: {xml_path}")
        return

    out_root = Path(args.output_root) if args.output_root else data_dir
    replay_root = out_root / "replay_mp4"

    print(f"Loading model: {xml_path.name}")
    model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
    data = mujoco.MjData(model)

    hdf5_files = sorted(data_dir.glob("*.hdf5"))
    if not hdf5_files:
        print(f"No .hdf5 files in {data_dir}")
        return

    views = list(CAMERA_VIEWS.items()) if args.all_views else [(args.camera, CAMERA_VIEWS[args.camera])]

    for h5_path in hdf5_files:
        stem = h5_path.stem
        try:
            demos = list_demos(h5_path)
        except Exception as e:
            print(f"Skip {h5_path.name}: {e}")
            continue

        for dk in demos:
            try:
                demo = load_demo(str(h5_path), dk)
            except Exception as e:
                print(f"Skip {stem}/{dk}: {e}")
                continue

            print(f"Processing {stem} / {dk} …")
            trajectories = compute_fk_trajectories(
                model,
                data,
                demo["joint_pos"],
                demo.get("hand_left_qpos"),
                demo.get("hand_right_qpos"),
            )
            if not args.all_links:
                trajectories = {k: v for k, v in trajectories.items() if "eef" in k}

            dt = _median_dt(demo["robot_ts"])
            fps = args.fps if args.fps is not None else min(60.0, max(1.0, 1.0 / dt))

            for view_name, cam_cfg in views:
                out_mp4 = replay_root / stem / dk / f"replay_{view_name}.mp4"
                render_one_demo_mp4(
                    model,
                    data,
                    demo,
                    trajectories,
                    out_mp4,
                    cam_cfg,
                    fps=fps,
                    width=args.width,
                    height=args.height,
                    subsample=args.subsample,
                    robot_alpha=args.robot_alpha,
                    max_geom=args.max_geom,
                )

    print("Done.")


if __name__ == "__main__":
    main()
