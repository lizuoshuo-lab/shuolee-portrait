#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
cd /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/GaussianAvatars_SMPLX-shuolee
/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python -u prepare_static_surface_085.py
