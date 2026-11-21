#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
cd '/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/GaussianAvatars_SMPLX-shuolee'
python=/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python
"$python" -u prepare_barycentric_arap_086.py
"$python" -u check_linear_arap_086.py
"$python" -u fit_barycentric_arap_086.py
