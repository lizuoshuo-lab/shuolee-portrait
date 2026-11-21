#!/bin/bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
python /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human/train_enerf_surface_control.py --out /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/dynamic-human-enerf-surface-gs-055/surface_only --steps 6000
export ENERF_RUN_DIR=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/dynamic-human-enerf-surface-gs-055/surface_only
python /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human/evaluate_enerf_surface_control.py
python /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human/train_enerf_surface_control.py --out /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/dynamic-human-enerf-surface-gs-055/surface_regularized --steps 6000 --surface-reg
export ENERF_RUN_DIR=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/dynamic-human-enerf-surface-gs-055/surface_regularized
python /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human/evaluate_enerf_surface_control.py
