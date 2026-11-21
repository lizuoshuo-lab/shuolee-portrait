#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
cd /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/PGSR-shuolee
/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python -u verify_pixel_center_remedy_091.py
