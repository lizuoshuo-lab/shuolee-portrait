#!/bin/bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
L=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
branches=("${@}")
if [ ${#branches[@]} -eq 0 ]; then branches=(path_rigidity path_rigidity_binding); fi
for branch in "${branches[@]}"; do
  case "$branch" in path_rigidity|path_rigidity_binding) ;; *) exit 2;; esac
  for scope in main continuity; do
    python "$L/scripts/dynamic_human/evaluate_actor23_geodesic.py" --branch "$branch" --scope "$scope" > "$L/outputs/actor23-geodesic-107-${branch}-${scope}-evaluation.log" 2>&1
  done
done
