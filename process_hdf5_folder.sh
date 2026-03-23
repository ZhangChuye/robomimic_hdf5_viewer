#!/usr/bin/env bash
#
# Generate ALL offline visualizations for every HDF5 file in a folder.
#
# Output structure (all inside DATA_DIR):
#   visualization/<hdf5_stem>/<demo_key>/   ← joint/cartesian plots, aria video
#   traj_viz/<hdf5_stem>/<demo_key>/          ← per-demo EEF trajectory PNGs
#   traj_viz_overlay/                         ← all-demo EEF overlay PNGs
#   replay_mp4/<hdf5_stem>/<demo_key>/       ← per-demo replay MP4s (EEF trail)
#   eef_trajectories_3d.html                 ← interactive 3D Plotly HTML
#
# Usage:
#   bash process_hdf5_folder.sh /path/to/data/rby_mustard
#
# Requirements: conda env "rby1" with h5py, numpy, matplotlib, mujoco, cv2, plotly.

set -euo pipefail

DATA_DIR="${1:?Usage: $0 <data_dir>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)/scripts"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$DATA_DIR" ]; then
  echo "Error: $DATA_DIR is not a directory"
  exit 1
fi

export MUJOCO_GL=egl

echo "=========================================="
echo "  RBY1 HDF5 Visualization Pipeline"
echo "  Data dir: $DATA_DIR"
echo "=========================================="
echo ""

# ── Step 1: Joint / Cartesian / Aria plots ────────────────────────────────────
echo "── Step 1/5: Joint & Cartesian plots + Aria video ──"
conda run -n rby1 python "$SCRIPT_DIR/generate_all_visualizations.py" "$DATA_DIR"
echo ""

# ── Step 2: Per-demo EEF trajectory PNGs ──────────────────────────────────────
echo "── Step 2/5: Per-demo 3D EEF trajectory renders ──"

# List demo keys from each HDF5 via a tiny inline Python
for HDF5 in "$DATA_DIR"/*.hdf5; do
  [ -f "$HDF5" ] || continue
  STEM="$(basename "$HDF5" .hdf5)"

  DEMOS=$(conda run -n rby1 python -c "
import h5py, sys
with h5py.File(sys.argv[1], 'r') as f:
    for k in sorted(f['data'].keys()):
        print(k)
" "$HDF5")

  for DK in $DEMOS; do
    OUT="$DATA_DIR/traj_viz/$STEM/$DK"
    echo "  $STEM / $DK → $OUT/"
    conda run -n rby1 python "$SCRIPT_DIR/plot_traj_1_demo.py" \
      "$HDF5" \
      --demo_key "$DK" \
      --output_dir "$OUT" \
      --no_interactive
  done
done
echo ""

# ── Step 3: All-demo EEF overlay ─────────────────────────────────────────────
echo "── Step 3/5: All-demo EEF overlay ──"
conda run -n rby1 python "$SCRIPT_DIR/overplot_all_eef_traj.py" \
  "$DATA_DIR" \
  --no_interactive
echo ""

# ── Step 4: Headless replay MP4s (EEF trajectory) ────────────────────────────
echo "── Step 4/5: Replay MP4s (MuJoCo + EEF trail) ──"
conda run -n rby1 python "$SCRIPT_DIR/render_traj_mp4_for_all.py" "$DATA_DIR"
echo ""

# ── Step 5: Interactive 3D HTML (Plotly) ──────────────────────────────────────
echo "── Step 5/5: Interactive 3D HTML (Plotly) ──"
conda run -n rby1 python "$SCRIPT_DIR/html_traj_plot.py" "$DATA_DIR"
echo ""

echo "=========================================="
echo "  All done.  Output in:"
echo "    $DATA_DIR/visualization/"
echo "    $DATA_DIR/traj_viz/"
echo "    $DATA_DIR/traj_viz_overlay/"
echo "    $DATA_DIR/replay_mp4/"
echo "    $DATA_DIR/eef_trajectories_3d.html"
echo "=========================================="
