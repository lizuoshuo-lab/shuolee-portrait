#!/bin/bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
L=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
for branch in static dynamic_nominal dynamic_clock; do
  test -f "$L/outputs/dynamic-human-actor23-independent-106/$branch/completion.json"
done
for branch in static dynamic_nominal dynamic_clock; do
  python "$L/scripts/dynamic_human/evaluate_actor23_control.py" --branch "$branch" --preview > "$L/outputs/actor23-independent-106-${branch}-preview.log" 2>&1
done
for branch in static dynamic_nominal dynamic_clock; do
  python "$L/scripts/dynamic_human/evaluate_actor23_control.py" --branch "$branch" > "$L/outputs/actor23-independent-106-${branch}-evaluation.log" 2>&1
  python "$L/scripts/dynamic_human/evaluate_actor23_control.py" --branch "$branch" --scope continuity > "$L/outputs/actor23-independent-106-${branch}-continuity.log" 2>&1
done
