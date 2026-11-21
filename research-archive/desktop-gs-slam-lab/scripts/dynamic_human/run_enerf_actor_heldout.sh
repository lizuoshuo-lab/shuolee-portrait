#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
LAB=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
python "$LAB/scripts/dynamic_human/render_enerf_actor_heldout.py"
python "$LAB/scripts/dynamic_human/diagnose_enerf_supervised_layers.py"
