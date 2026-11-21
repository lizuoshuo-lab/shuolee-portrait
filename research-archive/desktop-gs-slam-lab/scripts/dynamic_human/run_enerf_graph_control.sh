#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
LAB=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
RESULT="$LAB/outputs/dynamic-human-enerf-residual-graph-067"
MOTION="$LAB/outputs/dynamic-human-enerf-head-motion-066/capture-affines.npz"
python "$LAB/scripts/dynamic_human/train_enerf_graph_control.py" --branch residual_graph --result-root "$RESULT" --motion-file "$MOTION"
python "$LAB/scripts/dynamic_human/evaluate_enerf_graph_control.py" --branch residual_graph --result-root "$RESULT" --motion-file "$MOTION"
