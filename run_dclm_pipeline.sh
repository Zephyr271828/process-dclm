#!/usr/bin/env bash
set -euo pipefail

source $(conda info --base)/etc/profile.d/conda.sh
conda activate dclm

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

if [[ "${1:-}" != "--foreground" ]]; then
  nohup bash "$SCRIPT_DIR/run_dclm_pipeline.sh" --foreground "$@" >"$LOG_DIR/run_dclm_pipeline.launch.log" 2>&1 &
  pid=$!
  printf 'Started DCLM pipeline in background (pid=%s)\n' "$pid"
  printf 'Launcher log: %s\n' "$LOG_DIR/run_dclm_pipeline.launch.log"
  exit 0
fi

shift
exec python -u "$SCRIPT_DIR/dclm_pipeline.py" all "$@"
