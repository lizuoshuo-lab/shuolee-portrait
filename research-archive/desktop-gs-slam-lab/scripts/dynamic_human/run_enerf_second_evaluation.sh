#!/bin/bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
L=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
for branch in global_cubic local_bspline; do
 python "$L/scripts/dynamic_human/evaluate_enerf_second_interval.py" --branch "$branch" --result-root "$L/outputs/dynamic-human-enerf-second-interval-079" --motion-file "$L/outputs/dynamic-human-enerf-second-interval-079/heldout/capture-affines-estimated.npz" 2>&1 | tee -a "$L/logs/enerf-second-interval-079-$branch-evaluation.log"
done
