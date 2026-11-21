#!/bin/bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
export ENERF_RUN_DIR=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/dynamic-human-enerf-surface-long-056
python /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human/train_enerf_surface_long.py --out "$ENERF_RUN_DIR" --steps 30000 --surface-reg
python /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human/evaluate_enerf_surface_long.py
