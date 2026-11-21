#!/bin/bash
set -euo pipefail
mkdir -p /opt/gs-slam-lab/surface-gs
cp /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human/surface_gs_env.sh /opt/gs-slam-lab/surface-gs/env.sh
source /opt/gs-slam-lab/surface-gs/env.sh
python /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human/smoke_enerf_surface_gs.py
