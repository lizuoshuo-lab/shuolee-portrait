#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
LAB=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
RESULT="$LAB/outputs/dynamic-human-enerf-head-motion-066"
test -f "$RESULT/independent-audit.json"
test -f "$RESULT/head-only-invariance-audit.json"
python "$LAB/scripts/dynamic_human/train_enerf_articulated_control.py" --branch head_only --result-root "$RESULT" --motion-file "$RESULT/capture-affines.npz"
python "$LAB/scripts/dynamic_human/evaluate_enerf_articulated_control.py" --branch head_only --result-root "$RESULT" --motion-file "$RESULT/capture-affines.npz"
