#!/usr/bin/env bash
#
# Generate ALL offline visualizations for every HDF5 file in a folder.
#
# Output structure (all inside DATA_DIR):
#   visualization/<hdf5_stem>/<demo_key>/   ← joint/cartesian plots, aria video
#   traj_viz/<hdf5_stem>/<demo_key>/        ← per-demo EEF trajectory PNGs
#   traj_viz_overlay/                       ← all-demo EEF overlay PNGs
#
# Usage:
#   bash process_hdf5_folder.sh /path/to/data/rby_mustard
#
# Requirements: conda env "rby1" with h5py, numpy, matplotlib, mujoco, cv2.

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
echo "── Step 1/3: Joint & Cartesian plots + Aria video ──"
conda run --no-banner -n rby1 python "$SCRIPT_DIR/generate_all_visualizations.py" "$DATA_DIR"
echo ""

# ── Step 2: Per-demo EEF trajectory PNGs ──────────────────────────────────────
echo "── Step 2/3: Per-demo 3D EEF trajectory renders ──"

# List demo keys from each HDF5 via a tiny inline Python
for HDF5 in "$DATA_DIR"/*.hdf5; do
  [ -f "$HDF5" ] || continue
  STEM="$(basename "$HDF5" .hdf5)"

  DEMOS=$(conda run --no-banner -n rby1 python -c "
import h5py, sys
with h5py.File(sys.argv[1], 'r') as f:
    for k in sorted(f['data'].keys()):
        print(k)
" "$HDF5")

  for DK in $DEMOS; do
    OUT="$DATA_DIR/traj_viz/$STEM/$DK"
    echo "  $STEM / $DK → $OUT/"
    conda run --no-banner -n rby1 python "$SCRIPT_DIR/plot_traj_1_demo.py" \
      "$HDF5" \
      --demo_key "$DK" \
      --output_dir "$OUT" \
      --no_interactive
  done
done
echo ""

# ── Step 3: All-demo EEF overlay ─────────────────────────────────────────────
echo "── Step 3/3: All-demo EEF overlay ──"
conda run --no-banner -n rby1 python "$SCRIPT_DIR/overplot_all_eef_traj.py" \
  "$DATA_DIR" \
  --no_interactive
echo ""

echo "=========================================="
echo "  All done.  Output in:"
echo "    $DATA_DIR/visualization/"
echo "    $DATA_DIR/traj_viz/"
echo "    $DATA_DIR/traj_viz_overlay/"
echo "=========================================="
