#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
L=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
for branch in joint_photo joint_flow; do
 python "$L/scripts/dynamic_human/evaluate_enerf_hand_joint.py" --branch "$branch" 2>&1 | tee -a "$L/logs/enerf-hand-joint-084-eval-$branch.log"
done
