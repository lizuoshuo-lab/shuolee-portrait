#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
cd /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/GaussianAvatars_SMPLX-shuolee
/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python -u prepare_observed_correspondence_077.py
