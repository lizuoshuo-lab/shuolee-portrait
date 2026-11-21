#!/bin/bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
S=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human
python "$S/render_enerf_detail_heldout.py"
python "$S/audit_trained_enerf_detail.py"
