#!/bin/bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
task_py=/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python
task_lab=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
task_base="$task_lab/outputs/mini-splatting2-mip-depth-001/shuolee/current"
task_run="$task_base/static-ncc-paired-fresh-093/control"
cd "$task_lab/repos/GaussianAvatars_SMPLX-shuolee"
"$task_py" -u audit_pgsr_static_delivery.py --run "$task_run" --reference "$task_base/person-static-human-surface-init-fresh-083/gaussian-run/point_cloud.ply" --reference-label '083 legacy reference' --label '093 control' > "$task_run/export-review.log" 2>&1
