#!/bin/bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
cd /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/GaussianAvatars_SMPLX-shuolee
py=/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python
out=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/human-head-surface-initialization-082
"$py" -u seal_head_surface_082.py
timeout 600 "$py" -u fit_head_surface_082.py rigid_neutral 2>&1 | tee "$out/rigid_neutral.log"
timeout 600 "$py" -u fit_head_surface_082.py shared_shape100 2>&1 | tee "$out/shared_shape100.log"
"$py" -u review_head_surface_082.py rigid_neutral
"$py" -u review_head_surface_082.py shared_shape100
