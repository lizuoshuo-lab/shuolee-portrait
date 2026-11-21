#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/sam2/env.sh
export CUDA_VISIBLE_DEVICES=0
python /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human/segment_enerf_open_hand.py
