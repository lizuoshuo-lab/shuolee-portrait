#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/4k4d/env.sh
LAB=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
cd "$LAB/repos/4K4D"
unset FOURKD_PROJECTION_OFFSETS FOURKD_SOURCE_FOREGROUND_GATE
for branch in original_blending source_support_blending; do
  export FOURKD_RUN_DIR="$LAB/outputs/dynamic-human-4k4d-source-gate-training-044/$branch"
  python "$LAB/scripts/dynamic_human/train_4k4d_source_gate.py" -c "$FOURKD_RUN_DIR/pilot.yaml" > "$FOURKD_RUN_DIR/training.log" 2>&1
  unset FOURKD_SOURCE_FOREGROUND_GATE
  if [ "$branch" = source_support_blending ]; then export FOURKD_SOURCE_FOREGROUND_GATE=1; fi
  python "$LAB/scripts/dynamic_human/evaluate_4k4d_pilot.py" -c "$FOURKD_RUN_DIR/pilot.yaml" > "$FOURKD_RUN_DIR/evaluation.log" 2>&1
done
