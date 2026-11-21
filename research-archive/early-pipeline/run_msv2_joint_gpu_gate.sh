#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/msv2-clean-runtime-20261105/runtime.env
export PYTHONPATH=/mnt/d/Projects/msv2-paired
export MSV2_ROI_BACKEND_DIR=/opt/gs-slam-lab/msv2-roi-backend-20261106
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec /opt/gs-slam-lab/msv2-clean-runtime-20261105/.venv/bin/python '/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/probe_msv2_joint_gpu.py' --execute "$@"
