#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export MSV2_EXPERIMENT_CONFIG='C:/Users/Administrator/Documents/New project/shuolee-slide-update/msv2_demo59.json'
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
ulimit -n 16384
exec python '/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/run_standard_msv2_job.py' "$@"
