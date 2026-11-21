#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
L=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
python "$L/scripts/dynamic_human/diagnose_enerf_hand_spatial_support.py" 2>&1 | tee "$L/logs/enerf-hand-joint-084-spatial.log"
