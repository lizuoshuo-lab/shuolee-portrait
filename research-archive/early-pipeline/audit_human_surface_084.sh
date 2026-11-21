#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
task_lab=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
cd "$task_lab/repos/PGSR-shuolee"
/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python -u audit_human_surface_084.py
