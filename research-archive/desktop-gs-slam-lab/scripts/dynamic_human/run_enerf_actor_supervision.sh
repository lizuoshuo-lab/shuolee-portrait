#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
LAB=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
ROOT="$LAB/outputs/dynamic-human-enerf-actor-supervision-070"
MOTION="$LAB/outputs/dynamic-human-enerf-head-motion-066/capture-affines.npz"
python "$LAB/scripts/dynamic_human/train_enerf_actor_supervision.py" --branch residual_graph --result-root "$ROOT" --motion-file "$MOTION"
python "$LAB/scripts/dynamic_human/evaluate_enerf_graph_control.py" --branch residual_graph --result-root "$ROOT" --motion-file "$MOTION"
python "$LAB/scripts/dynamic_human/diagnose_enerf_training_owners.py" --candidate
