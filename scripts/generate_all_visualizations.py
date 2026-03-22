"""
Generate offline visualizations for all HDF5 demos in a directory.

Usage:
    python scripts/generate_all_visualizations.py /path/to/data/rby_mustard
    python scripts/generate_all_visualizations.py /path/to/data/rby_mustard --demo_key demo_0

Output tree:
    <input_dir>/visualization/<hdf5_stem>/<demo_key>/
        01_aria_video.mp4
        02_joint_cmd_vs_obs_<group>.png
        03_cartesian_cmd_vs_obs_<side>.png
        04_overview_all_joint_actions.png
        05_overview_all_cartesian_actions.png
        06_overview_all_cartesian_proprio.png
        07_overview_all_joint_proprio.png
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.hdf5_loader import list_demos, load_demo
from utils.plotting import (
    BODY_GROUPS, HAND_GROUPS, BASE_LABELS, _is_all_zero,
    plot_joint_comparison, plot_cartesian_comparison,
    plot_joint_overview, plot_cartesian_overview, save_video,
)

import numpy as np


def generate_demo_viz(hdf5_path, demo_key, out_dir):
    """Generate all visualizations for one demo."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = load_demo(hdf5_path, demo_key)
    t = d["t"]
    acts = d.get("actions_joint")
    acts_eef = d.get("actions_eef")

    print(f"  {demo_key}: {d['N']} frames, {t[-1]:.1f}s")

    # ── 1. Aria video ─────────────────────────────────────────────────────
    if d["aria_image"] is not None and not _is_all_zero(d["aria_image"]):
        dt = np.median(np.diff(d["robot_ts"]))
        fps = max(1, int(round(1.0 / dt))) if dt > 0 else 20
        save_video(d["aria_image"], fps, out_dir / "01_aria_video.mp4")
        print("    ✓ 01 aria video")

    # ── 2. Per-group joint: Command vs Observation ────────────────────────
    for name, key, obs_sl, act_sl, labels in BODY_GROUPS:
        obs_data = d["joint_pos"][:, obs_sl]
        if acts is not None:
            cmd_data = acts[:, act_sl]
            plot_joint_comparison(
                t, obs_data, cmd_data, labels,
                f"{name}: Command vs. Observation",
                out_dir / f"02_joint_cmd_vs_obs_{key}.png",
            )
            print(f"    ✓ 02 joint cmd vs obs — {name}")

    for name, key, obs_key, cmd_key, act_sl, labels in HAND_GROUPS:
        obs_data = d.get(obs_key)
        cmd_data = d.get(cmd_key)
        if obs_data is not None and cmd_data is not None and not _is_all_zero(obs_data):
            plot_joint_comparison(
                t, obs_data, cmd_data, labels,
                f"{name}: Command vs. Observation",
                out_dir / f"02_joint_cmd_vs_obs_{key}.png",
            )
            print(f"    ✓ 02 joint cmd vs obs — {name}")

    # ── 3. Cartesian: Command vs Observation ──────────────────────────────
    if acts_eef is not None:
        for side, obs_key, act_sl in [
            ("Right", "eef_right_proprio", slice(0, 16)),
            ("Left",  "eef_left_proprio",  slice(16, 32)),
        ]:
            obs_eef = d.get(obs_key)
            if obs_eef is not None and not _is_all_zero(obs_eef):
                plot_cartesian_comparison(
                    t, obs_eef, acts_eef[:, act_sl],
                    side,
                    out_dir / f"03_cartesian_cmd_vs_obs_{side.lower()}_eef.png",
                )
                print(f"    ✓ 03 cartesian cmd vs obs — {side} EEF")

    # ── 4. All joint actions overview ─────────────────────────────────────
    if acts is not None:
        groups = []
        for name, key, _, act_sl, labels in BODY_GROUPS:
            groups.append((name, acts[:, act_sl], labels))
        groups.append(("Base Cmd", acts[:, 22:25], BASE_LABELS))
        for name, key, _, _, act_sl, labels in HAND_GROUPS:
            if not _is_all_zero(acts[:, act_sl]):
                groups.append((f"{name} Cmd", acts[:, act_sl], labels))
        plot_joint_overview(t, groups, "All Joint Actions", out_dir / "04_overview_all_joint_actions.png")
        print("    ✓ 04 all joint actions overview")

    # ── 5. All Cartesian actions overview ─────────────────────────────────
    if acts_eef is not None:
        eefs = []
        for side, sl in [("Right EEF", slice(0, 16)), ("Left EEF", slice(16, 32))]:
            if not _is_all_zero(acts_eef[:, sl]):
                eefs.append((side, acts_eef[:, sl]))
        if eefs:
            plot_cartesian_overview(t, eefs, "All Cartesian Actions", out_dir / "05_overview_all_cartesian_actions.png")
            print("    ✓ 05 all cartesian actions overview")

    # ── 6. All Cartesian proprio overview ─────────────────────────────────
    eefs = []
    for side, key in [("Right EEF", "eef_right_proprio"), ("Left EEF", "eef_left_proprio")]:
        eef = d.get(key)
        if eef is not None and not _is_all_zero(eef):
            eefs.append((side, eef))
    if eefs:
        plot_cartesian_overview(t, eefs, "All Cartesian Proprio", out_dir / "06_overview_all_cartesian_proprio.png")
        print("    ✓ 06 all cartesian proprio overview")

    # ── 7. All joint proprio overview ─────────────────────────────────────
    groups = []
    for name, key, obs_sl, _, labels in BODY_GROUPS:
        groups.append((name, d["joint_pos"][:, obs_sl], labels))
    for name, key, obs_key, _, _, labels in HAND_GROUPS:
        hand = d.get(obs_key)
        if hand is not None and not _is_all_zero(hand):
            groups.append((name, hand, labels))
    plot_joint_overview(t, groups, "All Joint Proprio", out_dir / "07_overview_all_joint_proprio.png")
    print("    ✓ 07 all joint proprio overview")


def main():
    parser = argparse.ArgumentParser(description="Generate HDF5 visualizations")
    parser.add_argument("input_dir", type=str, help="Directory containing .hdf5 files")
    parser.add_argument("--demo_key", type=str, default=None,
                        help="Process only this demo key (default: all demos)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    hdf5_files = sorted(input_dir.glob("*.hdf5"))
    if not hdf5_files:
        print(f"No .hdf5 files found in {input_dir}")
        return

    viz_root = input_dir / "visualization"
    print(f"Found {len(hdf5_files)} HDF5 file(s). Output → {viz_root}/\n")

    for hdf5_path in hdf5_files:
        stem = hdf5_path.stem
        demos = [args.demo_key] if args.demo_key else list_demos(hdf5_path)
        print(f"[{stem}] — {len(demos)} demo(s)")

        for dk in demos:
            out_dir = viz_root / stem / dk
            try:
                generate_demo_viz(hdf5_path, dk, out_dir)
            except Exception as e:
                print(f"  ✗ {dk}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
