#!/usr/bin/env bash
set -euo pipefail

MAX_JOBS=2
PYTHON_BIN="/coc/flash7/czhang883/miniconda3/envs/rby1/bin/python"
SCRIPT_DIR="/coc/flash7/czhang883/Documents/robomimic_hdf5_viewer/scripts"
export MUJOCO_GL=egl

usage() {
  echo "Usage: $0 <data_dir1> [data_dir2 ...]"
  exit 1
}

log() {
  echo "[$(date '+%F %T')] $*"
}

run_py() {
  "$PYTHON_BIN" "$@"
}

get_demo_keys() {
  local hdf5="$1"
  run_py -c "
import h5py, sys
with h5py.File(sys.argv[1], 'r') as f:
    for k in sorted(f['data'].keys()):
        print(k)
" "$hdf5"
}

process_folder() {
  local data_dir="$1"
  local start_ts end_ts
  local hdf5_count=0
  local demo_count=0

  if [[ ! -d "$data_dir" ]]; then
    log "[ERROR] Not a directory: $data_dir"
    return 1
  fi

  start_ts=$(date +%s)
  log "[START] $data_dir"

  log "[STEP 1/5] generate_all_visualizations.py"
  run_py "$SCRIPT_DIR/generate_all_visualizations.py" "$data_dir"

  log "[STEP 2/5] plot_traj_1_demo.py"
  shopt -s nullglob
  local hdf5_files=("$data_dir"/*.hdf5)
  shopt -u nullglob

  if [[ ${#hdf5_files[@]} -eq 0 ]]; then
    log "[WARN] No .hdf5 files found in: $data_dir"
  else
    for hdf5 in "${hdf5_files[@]}"; do
      local stem
      stem="$(basename "$hdf5" .hdf5)"
      ((hdf5_count += 1))

      local demos=()
      mapfile -t demos < <(get_demo_keys "$hdf5")
      log "[HDF5] $stem | demos=${#demos[@]}"

      for dk in "${demos[@]}"; do
        local out_dir="$data_dir/traj_viz/$stem/$dk"
        ((demo_count += 1))
        log "  [DEMO] $stem / $dk"
        run_py "$SCRIPT_DIR/plot_traj_1_demo.py" \
          "$hdf5" \
          --demo_key "$dk" \
          --output_dir "$out_dir" \
          --no_interactive
      done
    done
  fi

  log "[STEP 3/5] overplot_all_eef_traj.py"
  run_py "$SCRIPT_DIR/overplot_all_eef_traj.py" "$data_dir" --no_interactive

  log "[STEP 4/5] render_traj_mp4_for_all.py"
  run_py "$SCRIPT_DIR/render_traj_mp4_for_all.py" "$data_dir"

  log "[STEP 5/5] html_traj_plot.py"
  run_py "$SCRIPT_DIR/html_traj_plot.py" "$data_dir"

  end_ts=$(date +%s)
  log "[DONE] $data_dir | hdf5=$hdf5_count demos=$demo_count elapsed=$((end_ts - start_ts))s"
}

main() {
  [[ "$#" -ge 1 ]] || usage
  [[ -x "$PYTHON_BIN" ]] || { echo "Error: Python not executable: $PYTHON_BIN"; exit 1; }

  log "Pipeline start"
  log "Folders=$# | MAX_JOBS=$MAX_JOBS | PYTHON_BIN=$PYTHON_BIN | MUJOCO_GL=$MUJOCO_GL"

  local running=0
  local total=0

  for data_dir in "$@"; do
    process_folder "$data_dir" &
    ((running += 1))
    ((total += 1))
    log "[LAUNCHED] $data_dir | running=$running/$MAX_JOBS"

    if (( running >= MAX_JOBS )); then
      wait -n
      ((running -= 1))
      log "[SLOT FREED] running=$running/$MAX_JOBS"
    fi
  done

  wait
  log "All folders processed | total=$total"
}

main "$@"