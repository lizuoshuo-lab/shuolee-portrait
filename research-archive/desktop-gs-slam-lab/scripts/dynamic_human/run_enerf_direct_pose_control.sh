#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
LAB=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
RESULT="$LAB/outputs/dynamic-human-enerf-direct-pose-065"
test -f "$RESULT/independent-audit.json"
python "$LAB/scripts/dynamic_human/train_enerf_articulated_control.py" --branch direct_reprojection --result-root "$RESULT" --motion-file "$RESULT/capture-affines.npz"
python "$LAB/scripts/dynamic_human/evaluate_enerf_articulated_control.py" --branch direct_reprojection --result-root "$RESULT" --motion-file "$RESULT/capture-affines.npz"
