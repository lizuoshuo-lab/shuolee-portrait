#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
L=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
python "$L/scripts/dynamic_human/train_enerf_hand_snapshot.py" --branch single_time
python "$L/scripts/dynamic_human/evaluate_enerf_hand_snapshot.py" --branch single_time
