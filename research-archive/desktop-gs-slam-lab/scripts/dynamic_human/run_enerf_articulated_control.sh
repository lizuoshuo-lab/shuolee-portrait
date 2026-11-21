#!/bin/bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
python /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human/train_enerf_articulated_control.py --branch fixed_offsets --steps 15000
python /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human/evaluate_enerf_articulated_control.py --branch fixed_offsets
python /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human/train_enerf_articulated_control.py --branch estimated_offsets --steps 15000
python /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human/evaluate_enerf_articulated_control.py --branch estimated_offsets
