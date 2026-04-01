"""
Overlay every EEF trajectory from all demos in all HDF5 files under a directory.

One MuJoCo scene: static safe-pose robot + all right/left EEF paths.
Uses the **same** time colormap for every path as `plot_traj_1_demo.py`:
  - Right EEF: green (start) → red (end)
  - Left EEF:  cyan (start) → purple (end)
Sphere/line radii are ~¼ of the single-demo defaults so many overlays stay readable.

Usage:
    python scripts/overplot_all_eef_traj.py /path/to/data/rby_mustard
    python scripts/overplot_all_eef_traj.py /path/to/data --no_interactive
    MUJOCO_GL=egl python scripts/overplot_all_eef_traj.py /path/to/data --no_interactive
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import mujoco
import mujoco.viewer

from utils.hdf5_loader import list_demos, load_demo
from utils.plot_action_traj_in_mujoco import (
    SAFE_POSE_ROBOT0_JOINT_26,
    compute_fk_trajectories,
    set_robot_pose,
    save_all_views,
    push_robot_alpha,
    pop_robot_alpha,
    add_trajectory,
    RIGHT_EEF_START,
    RIGHT_EEF_END,
    LEFT_EEF_START,
    LEFT_EEF_END,
    EEF_SPHERE_RADIUS,
    EEF_LINE_RADIUS,
    EEF_MARKER_RADIUS,
)

_ASSETS = Path(__file__).parent.parent / "assets"
_XML = _ASSETS / "rby1_with_xhand" / "model_v1.3_xhand_act.xml"

# Finer geometry than single-demo viz (~¼ of base EEF radii)
_OVERLAY_RADIUS_SCALE = 0.25


def collect_eef_series(data_dir: Path):
    """Returns list of dicts with joint_pos, hand qpos, label, path, demo_key."""
    series = []
    hdf5_files = sorted(data_dir.glob("*.hdf5"))
    for path in hdf5_files:
        stem = path.stem
        try:
            demos = list_demos(path)
        except Exception as e:
            print(f"  skip {path.name}: {e}")
            continue
        for dk in demos:
            try:
                d = load_demo(str(path), dk)
            except Exception as e:
                print(f"  skip {path.name} {dk}: {e}")
                continue
            series.append(
                dict(
                    label=f"{stem}/{dk}",
                    path=path,
                    demo_key=dk,
                    joint_pos=d["joint_pos"],
                    hand_left=d.get("hand_left_qpos"),
                    hand_right=d.get("hand_right_qpos"),
                )
            )
    return series


def main():
    parser = argparse.ArgumentParser(
        description="Overlay all EEF trajectories from every demo in a data folder"
    )
    parser.add_argument("data_dir", type=str, help="Folder containing .hdf5 files")
    parser.add_argument("--xml", type=str, default=None)
    parser.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="Sample every N-th frame — higher = fewer geoms when many demos (default 6)",
    )
    parser.add_argument("--robot_alpha", type=float, default=1.0)
    parser.add_argument("--no_interactive", action="store_true")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="PNG output dir (default: <data_dir>/traj_viz_overlay/)",
    )
    parser.add_argument(
        "--max_geom",
        type=int,
        default=80000,
        help="Offscreen renderer scene capacity (many overlaid EEF paths; default 80000)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"Not a directory: {data_dir}")
        return

    xml_path = Path(args.xml) if args.xml else _XML
    if not xml_path.exists():
        print(f"XML not found: {xml_path}")
        return

    print(f"Scanning {data_dir} …")
    entries = collect_eef_series(data_dir)
    if not entries:
        print("No demos found.")
        return
    print(f"Found {len(entries)} demo(s) across HDF5 files.")

    model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
    data = mujoco.MjData(model)

    # FK per demo → keep only EEF
    eef_runs = []
    for i, e in enumerate(entries):
        tr = compute_fk_trajectories(
            model, data, e["joint_pos"], e.get("hand_left"), e.get("hand_right"),
        )
        if "right_eef" not in tr or "left_eef" not in tr:
            print(f"  skip {e['label']}: missing EEF bodies")
            continue
        eef_runs.append(
            dict(
                label=e["label"],
                right_eef=tr["right_eef"],
                left_eef=tr["left_eef"],
            )
        )
        print(f"  [{i+1}/{len(entries)}] {e['label']}: {tr['right_eef'].shape[0]} frames")

    if not eef_runs:
        print("No valid EEF trajectories.")
        return

    # Hands for static pose: first entry’s first frame
    e0 = entries[0]
    hl0 = e0["hand_left"][0] if e0.get("hand_left") is not None else None
    hr0 = e0["hand_right"][0] if e0.get("hand_right") is not None else None
    set_robot_pose(model, data, SAFE_POSE_ROBOT0_JOINT_26, hl0, hr0)

    sr = EEF_SPHERE_RADIUS * _OVERLAY_RADIUS_SCALE
    lr = EEF_LINE_RADIUS * _OVERLAY_RADIUS_SCALE
    mr = EEF_MARKER_RADIUS * _OVERLAY_RADIUS_SCALE * 0.75

    def populate_scene(scn):
        for run in eef_runs:
            add_trajectory(
                scn,
                run["right_eef"],
                sr,
                lr,
                RIGHT_EEF_START,
                RIGHT_EEF_END,
                subsample=args.subsample,
                marker_radius=mr,
            )
            add_trajectory(
                scn,
                run["left_eef"],
                sr,
                lr,
                LEFT_EEF_START,
                LEFT_EEF_END,
                subsample=args.subsample,
                marker_radius=mr,
            )

    out_dir = Path(args.output_dir) if args.output_dir else data_dir / "traj_viz_overlay"
    rgba_backup = push_robot_alpha(model, args.robot_alpha)
    try:
        print(f"Saving PNGs → {out_dir}/")
        save_all_views(
            model, data, populate_scene, out_dir,
            prefix="all_eef_overlay", max_geom=args.max_geom,
        )

        if not args.no_interactive:
            print("\nInteractive viewer (close to exit). All EEF paths overlaid.")
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
