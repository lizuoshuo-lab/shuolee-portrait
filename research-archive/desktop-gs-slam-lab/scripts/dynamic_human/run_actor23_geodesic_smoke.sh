#!/bin/bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
L=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
for branch in path_rigidity path_rigidity_binding; do
  python "$L/scripts/dynamic_human/train_actor23_geodesic.py" --mode smoke --branch "$branch" > "$L/outputs/actor23-geodesic-107-${branch}-smoke.log" 2>&1
done
