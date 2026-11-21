#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
LAB=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
S="$LAB/scripts/dynamic_human"
R="$LAB/outputs/dynamic-human-enerf-rgb-pose-correction-074"
python "$S/train_enerf_graph_control.py" --branch residual_graph --result-root "$R" --motion-file "$R/head/capture-affines.npz"
python "$S/evaluate_enerf_graph_control.py" --branch residual_graph --result-root "$R" --motion-file "$R/head/capture-affines.npz"
