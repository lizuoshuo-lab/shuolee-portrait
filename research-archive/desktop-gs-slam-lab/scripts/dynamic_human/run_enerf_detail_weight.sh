#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
LAB=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
S="$LAB/scripts/dynamic_human"
R="$LAB/outputs/dynamic-human-enerf-detail-weight-076"
M="$LAB/outputs/dynamic-human-enerf-rgb-pose-correction-074/head/capture-affines.npz"
python "$S/train_enerf_detail_weight.py" --branch residual_graph --result-root "$R" --motion-file "$M"
python "$S/evaluate_enerf_graph_control.py" --branch residual_graph --result-root "$R" --motion-file "$M"
