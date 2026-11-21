#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
L=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
python "$L/scripts/dynamic_human/prepare_enerf_hand_fine.py" 2>&1 | tee "$L/logs/enerf-hand-fine-085-preparation.log"
