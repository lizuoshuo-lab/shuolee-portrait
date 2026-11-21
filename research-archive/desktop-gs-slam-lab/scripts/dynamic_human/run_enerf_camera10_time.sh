#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
LAB=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
OUT="$LAB/outputs/dynamic-human-enerf-camera10-time-050"
for branch in same_frame camera10_previous_frame; do
  export ENERF_RUN_DIR="$OUT/$branch"
  python "$LAB/scripts/dynamic_human/train_enerf_camera10_time.py" --out "$ENERF_RUN_DIR" --steps 6000 > "$ENERF_RUN_DIR/training.log" 2>&1
  python "$LAB/scripts/dynamic_human/evaluate_enerf_camera10_time.py" > "$ENERF_RUN_DIR/evaluation.log" 2>&1
done
