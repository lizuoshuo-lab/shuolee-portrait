#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
R=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/dynamic-human-enerf-forearm-matte-repair-072
for i in $(seq 1 360); do
  if [ -f "$R/residual_graph/completion.json" ]; then
    python /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human/audit_trained_enerf_forearm.py
    exit 0
  fi
  sleep 10
done
exit 2
