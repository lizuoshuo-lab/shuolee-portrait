#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
L=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
for branch in photo_control normal_flow; do
 python "$L/scripts/dynamic_human/train_enerf_hand_flow_control.py" --branch "$branch" 2>&1 | tee -a "$L/logs/enerf-open-hand-flow-083-$branch.log"
done
