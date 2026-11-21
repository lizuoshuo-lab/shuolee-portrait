#!/bin/bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
python /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human/train_enerf_mask_uncertainty.py --out /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/dynamic-human-enerf-mask-uncertainty-053 --steps 6000
export ENERF_RUN_DIR=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/dynamic-human-enerf-mask-uncertainty-053
python /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human/evaluate_enerf_camera10_time.py
