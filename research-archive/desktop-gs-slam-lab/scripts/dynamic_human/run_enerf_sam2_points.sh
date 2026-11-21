#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/sam2/env.sh
cd /opt/gs-slam-lab/sam2
python /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human/probe_enerf_sam2_points.py > /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/dynamic-human-enerf-mask-quality-038/point-prompt.log 2>&1
