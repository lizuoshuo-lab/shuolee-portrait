#!/bin/bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
L=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
python "$L/scripts/dynamic_human/train_enerf_temporal_control.py" --branch residual_graph --result-root "$L/outputs/dynamic-human-enerf-temporal-basis-078" --motion-file "$L/outputs/dynamic-human-enerf-rgb-pose-correction-074/head/capture-affines.npz"
python "$L/scripts/dynamic_human/evaluate_enerf_temporal_control.py" --branch residual_graph --result-root "$L/outputs/dynamic-human-enerf-temporal-basis-078" --motion-file "$L/outputs/dynamic-human-enerf-rgb-pose-correction-074/head/capture-affines.npz"
