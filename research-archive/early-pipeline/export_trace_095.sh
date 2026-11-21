#!/bin/bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
task_lab=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
task_run="$task_lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/whole-body-early-trace-fresh-095"
cd "$task_lab/repos/GaussianAvatars_SMPLX-shuolee"
/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python -u audit_pgsr_static_delivery.py --run "$task_run" --early-trace --reference "$task_lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/static-ncc-paired-fresh-093/control/gaussian-run/point_cloud.ply" --reference-label '093 final18k comparison only' --label '095 fresh1800 diagnostic' > "$task_run/export.log" 2>&1
tail -n 3 "$task_run/export.log"
