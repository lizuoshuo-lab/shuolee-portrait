#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
L=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
python "$L/scripts/dynamic_human/diagnose_enerf_hand_flow_visibility.py" 2>&1 | tee "$L/logs/enerf-open-hand-flow-083-visibility.log"
