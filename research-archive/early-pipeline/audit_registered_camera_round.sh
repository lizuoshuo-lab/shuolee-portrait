#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export PYTHONDONTWRITEBYTECODE=1
task_py=/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python
task_lab=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
task_base="$task_lab/outputs/mini-splatting2-mip-depth-001/shuolee/current"
task_run="$task_base/person-static-registered-camera-fresh-072"
cd "$task_lab/repos/GaussianAvatars_SMPLX-shuolee"
"$task_py" -u audit_pgsr_static_delivery.py --run "$task_run" --reference "$task_base/person-static-mask-foreground-fresh-070/gaussian-run/point_cloud.ply" --reference-label '070 prior camera training' --label '072 registered camera training' > "$task_run/export-audit.log" 2>&1
"$task_py" -u review_registered_training.py > "$task_run/same-camera-review.log" 2>&1
tail -n 1 "$task_run/same-camera-review.log"
