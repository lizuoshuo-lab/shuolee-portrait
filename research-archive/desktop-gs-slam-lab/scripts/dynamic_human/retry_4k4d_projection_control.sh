#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/4k4d/env.sh
LAB=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
cd "$LAB/repos/4K4D"
export FOURKD_RUN_DIR="$LAB/outputs/dynamic-human-4k4d-projection-control-040/calibrated_projection"
test -f "$LAB/outputs/dynamic-human-4k4d-projection-control-040/mixed-device-recovery.json"
test ! -f "$FOURKD_RUN_DIR/model/latest.pt"
python "$LAB/scripts/dynamic_human/train_4k4d_projection_control.py" -c "$FOURKD_RUN_DIR/pilot.yaml" > "$FOURKD_RUN_DIR/training.log" 2>&1
export FOURKD_PROJECTION_OFFSETS="$LAB/outputs/dynamic-human-enerf-static-calibration-037/calibration.json"
python "$LAB/scripts/dynamic_human/evaluate_4k4d_pilot.py" -c "$FOURKD_RUN_DIR/pilot.yaml" > "$FOURKD_RUN_DIR/evaluation.log" 2>&1
