#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/msv2-clean-runtime-20261105/runtime.env
export CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=2
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$MSV2_PYTHON" -s '/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/review_msv2_face_composite.py' \
 --baseline-job /opt/gs-slam-lab/msv2-video15-20261105/runs/video15/job.json \
 --old-head-job /opt/gs-slam-lab/msv2-video15-20261105/runs/video15-head4x/job.json \
 --new-job /opt/gs-slam-lab/msv2-video15-20261105/runs/video15-parent1-child4/job.json \
 --old-record '/mnt/d/00/记录/2026-11-05_MSV2_video15_头部框超分' \
 --record '/mnt/d/00/记录/2026-11-06_MSV2_video15_子母图联合训练' "$@"
