#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
LAB=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
S="$LAB/scripts/dynamic_human"
RESULT="$LAB/outputs/dynamic-human-enerf-fine-graph-071"
MOTION="$LAB/outputs/dynamic-human-enerf-head-motion-066/capture-affines.npz"
python "$S/smoke_enerf_fine_graph.py"
python "$S/train_enerf_fine_graph.py" --branch residual_graph --result-root "$RESULT" --motion-file "$MOTION"
python "$S/evaluate_enerf_fine_graph.py" --branch residual_graph --result-root "$RESULT" --motion-file "$MOTION"
python "$S/render_enerf_fine_graph_heldout.py"
