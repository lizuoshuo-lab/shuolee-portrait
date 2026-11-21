#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
cd /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/PGSR-shuolee
/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python -u measure_pixel_center_subject_091.py
