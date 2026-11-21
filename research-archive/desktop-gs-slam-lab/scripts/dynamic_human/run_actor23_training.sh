#!/bin/bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
L=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
for branch in static dynamic_nominal dynamic_clock; do
  python "$L/scripts/dynamic_human/train_actor23_control.py" --mode train --branch "$branch" > "$L/outputs/actor23-independent-106-${branch}-training.log" 2>&1
done
