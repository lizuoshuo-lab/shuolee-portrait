#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
L=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
for b in parent single_time multi_time; do
 python "$L/scripts/dynamic_human/evaluate_enerf_hand_time_scope.py" --branch "$b"
done
