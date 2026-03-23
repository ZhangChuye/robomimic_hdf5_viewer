"""
Interactive 3D EEF trajectory visualization as a single self-contained HTML file.

Scans every HDF5 demo under a folder, runs FK via MuJoCo to extract EEF world
positions, then renders all paths in a Plotly 3D scatter. The result is a
drag-to-rotate, scroll-to-zoom webpage with per-demo toggling in the legend.

Uses the **same** time-gradient color scheme as ``overplot_all_eef_traj.py``:

  - Right EEF: green (start) → red (end)
  - Left  EEF: cyan  (start) → purple (end)
  - Start / End indicated by the color gradient along every path
  - Hover: file, demo, frame index, XYZ

Usage::

    python scripts/html_traj_plot.py /path/to/data
    python scripts/html_traj_plot.py /path/to/data -o my_viz.html
    python scripts/html_traj_plot.py /path/to/data --subsample 1  # every frame
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
    RIGHT_EEF_START,
    RIGHT_EEF_END,
    LEFT_EEF_START,
    LEFT_EEF_END,
    compute_fk_trajectories,
)

_ASSETS = Path(__file__).parent.parent / "assets"
_XML = _ASSETS / "rby1_with_xhand" / "model_v1.3_xhand_act.xml"


def _rgba_to_rgb_str(rgba):
    """Convert [r,g,b,a] (0-1 floats) to 'rgb(R,G,B)' string."""
    return f"rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})"


# Plotly colorscales matching the MuJoCo overlay color scheme
_RIGHT_COLORSCALE = [
    [0.0, _rgba_to_rgb_str(RIGHT_EEF_START)],
    [1.0, _rgba_to_rgb_str(RIGHT_EEF_END)],
]
_LEFT_COLORSCALE = [
    [0.0, _rgba_to_rgb_str(LEFT_EEF_START)],
    [1.0, _rgba_to_rgb_str(LEFT_EEF_END)],
]

_RIGHT_START_COLOR = _rgba_to_rgb_str(RIGHT_EEF_START)
_RIGHT_END_COLOR = _rgba_to_rgb_str(RIGHT_EEF_END)
_LEFT_START_COLOR = _rgba_to_rgb_str(LEFT_EEF_START)
_LEFT_END_COLOR = _rgba_to_rgb_str(LEFT_EEF_END)


def collect_eef_trajectories(data_dir: Path, model, data, subsample: int = 3):
    """Run FK on every demo in *data_dir* and return list of EEF path dicts."""
    hdf5_files = sorted(data_dir.glob("*.hdf5"))
    runs: list[dict] = []
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
            tr = compute_fk_trajectories(
                model, data, d["joint_pos"],
                d.get("hand_left_qpos"), d.get("hand_right_qpos"),
            )
            if "right_eef" not in tr or "left_eef" not in tr:
                print(f"  skip {stem}/{dk}: missing EEF bodies")
                continue
            # subsample
            idx = np.arange(0, tr["right_eef"].shape[0], max(1, subsample))
            if idx[-1] != tr["right_eef"].shape[0] - 1:
                idx = np.append(idx, tr["right_eef"].shape[0] - 1)
            runs.append(dict(
                label=f"{stem}/{dk}",
                stem=stem,
                demo_key=dk,
                right_eef=tr["right_eef"][idx],
                left_eef=tr["left_eef"][idx],
                frame_idx=idx,
            ))
            print(f"  [{len(runs)}] {stem}/{dk}: {tr['right_eef'].shape[0]} frames")
    return runs


def build_figure(runs: list[dict], title: str):
    import plotly.graph_objects as go

    fig = go.Figure()

    for i, run in enumerate(runs):
        label = run["label"]
        fidx = run["frame_idx"]

        for side, cscale, c_start, c_end, positions in [
            ("R", _RIGHT_COLORSCALE, _RIGHT_START_COLOR, _RIGHT_END_COLOR, run["right_eef"]),
            ("L", _LEFT_COLORSCALE,  _LEFT_START_COLOR,  _LEFT_END_COLOR,  run["left_eef"]),
        ]:
            N = positions.shape[0]
            t_norm = np.linspace(0.0, 1.0, N)
            tag = f"{label} ({side})"
            hover = [
                f"{label}<br>{side} EEF frame {fidx[k]}<br>"
                f"x={positions[k,0]:.4f}  y={positions[k,1]:.4f}  z={positions[k,2]:.4f}"
                for k in range(N)
            ]

            # Gradient-colored trajectory line
            fig.add_trace(go.Scatter3d(
                x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
                mode="lines",
                line=dict(
                    color=t_norm,
                    colorscale=cscale,
                    width=4,
                    showscale=False,
                ),
                opacity=0.55,
                name=tag,
                legendgroup=label,
                showlegend=(side == "R"),
                hovertext=hover,
                hoverinfo="text",
            ))

            # Start marker (start color)
            fig.add_trace(go.Scatter3d(
                x=[positions[0, 0]], y=[positions[0, 1]], z=[positions[0, 2]],
                mode="markers",
                marker=dict(size=5, color=c_start, symbol="circle"),
                legendgroup=label,
                showlegend=False,
                hovertext=[f"{tag} START<br>frame {fidx[0]}"],
                hoverinfo="text",
            ))

            # End marker (end color)
            fig.add_trace(go.Scatter3d(
                x=[positions[-1, 0]], y=[positions[-1, 1]], z=[positions[-1, 2]],
                mode="markers",
                marker=dict(size=6, color=c_end, symbol="diamond"),
                legendgroup=label,
                showlegend=False,
                hovertext=[f"{tag} END<br>frame {fidx[-1]}"],
                hoverinfo="text",
            ))

    # Axis bounds with padding
    all_pts = np.vstack(
        [r["right_eef"] for r in runs] + [r["left_eef"] for r in runs]
    )
    lo = all_pts.min(axis=0)
    hi = all_pts.max(axis=0)
    pad = (hi - lo).max() * 0.08 + 0.02

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        scene=dict(
            xaxis=dict(title="X (m)", range=[lo[0] - pad, hi[0] + pad],
                       backgroundcolor="rgb(240,240,240)", gridcolor="white", showbackground=True),
            yaxis=dict(title="Y (m)", range=[lo[1] - pad, hi[1] + pad],
                       backgroundcolor="rgb(240,240,240)", gridcolor="white", showbackground=True),
            zaxis=dict(title="Z (m)", range=[lo[2] - pad, hi[2] + pad],
                       backgroundcolor="rgb(240,240,240)", gridcolor="white", showbackground=True),
            aspectmode="data",
            camera=dict(eye=dict(x=1.3, y=1.3, z=0.9)),
        ),
        width=1400,
        height=900,
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    font=dict(size=10), itemsizing="constant"),
    )
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Interactive 3D HTML visualization of all EEF trajectories"
    )
    parser.add_argument("data_dir", type=str, help="Folder containing .hdf5 files")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output HTML path (default: <data_dir>/eef_trajectories_3d.html)")
    parser.add_argument("--xml", type=str, default=None)
    parser.add_argument("--subsample", type=int, default=3,
                        help="Keep every N-th frame for plotting (default 3)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"Not a directory: {data_dir}")
        return

    xml_path = Path(args.xml) if args.xml else _XML
    if not xml_path.exists():
        print(f"XML not found: {xml_path}")
        return

    print(f"Loading model: {xml_path.name}")
    model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
    data = mujoco.MjData(model)

    print(f"Scanning {data_dir} …")
    runs = collect_eef_trajectories(data_dir, model, data, subsample=args.subsample)
    if not runs:
        print("No EEF trajectories found.")
        return
    print(f"Collected {len(runs)} demo(s).")

    title = f"EEF Trajectories — {data_dir.name}  ({len(runs)} demos)"
    fig = build_figure(runs, title)

    out_path = Path(args.output) if args.output else data_dir / "eef_trajectories_3d.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import plotly.offline as pyo
    pyo.plot(fig, filename=str(out_path), auto_open=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
