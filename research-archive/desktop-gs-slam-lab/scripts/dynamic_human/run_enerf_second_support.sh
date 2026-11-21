#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
L=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
R="$L/outputs/dynamic-human-enerf-support-correction-081"
S="$L/outputs/dynamic-human-enerf-second-interval-079"
python "$L/scripts/dynamic_human/train_enerf_second_interval.py" --branch local_bspline --result-root "$R" --motion-file "$S/head/capture-affines.npz" --support-root "$R" > "$R/training.log" 2>&1
python "$L/scripts/dynamic_human/evaluate_enerf_second_interval.py" --branch local_bspline --result-root "$R" --motion-file "$S/heldout/capture-affines-estimated.npz" > "$R/evaluation.log" 2>&1
