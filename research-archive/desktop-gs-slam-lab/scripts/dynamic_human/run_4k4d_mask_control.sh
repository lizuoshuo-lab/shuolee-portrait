#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/4k4d/env.sh
LAB=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
export FOURKD_RUN_DIR="$LAB/outputs/dynamic-human-4k4d-mask-control-039"
cd "$LAB/repos/4K4D"
python "$LAB/scripts/dynamic_human/probe_4k4d_data.py" -c "$FOURKD_RUN_DIR/pilot.yaml" > "$FOURKD_RUN_DIR/data-probe.log" 2>&1
python "$LAB/scripts/dynamic_human/train_4k4d_pilot.py" -c "$FOURKD_RUN_DIR/pilot.yaml" > "$FOURKD_RUN_DIR/training.log" 2>&1
python "$LAB/scripts/dynamic_human/evaluate_4k4d_pilot.py" -c "$FOURKD_RUN_DIR/pilot.yaml" > "$FOURKD_RUN_DIR/evaluation.log" 2>&1
