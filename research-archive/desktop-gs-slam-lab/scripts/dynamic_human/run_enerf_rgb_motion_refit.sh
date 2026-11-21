#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
S=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human
python "$S/fit_enerf_rgb_body.py"
python "$S/fit_enerf_rgb_head.py"
