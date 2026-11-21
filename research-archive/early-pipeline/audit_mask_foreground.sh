#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export PYTHONDONTWRITEBYTECODE=1
task_py=/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python
task_lab=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
task_base="$task_lab/outputs/mini-splatting2-mip-depth-001/shuolee/current"
task_run="$task_base/person-static-mask-foreground-fresh-070"
cd "$task_lab/repos/GaussianAvatars_SMPLX-shuolee"
"$task_py" audit_pgsr_static_delivery.py --run "$task_run" --reference "$task_base/person-static-pgsr-fresh-064/gaussian-run/point_cloud.ply" --label '070 foreground-only color' > "$task_run/export-audit.log" 2>&1
"$task_py" review_static_alpha.py --runs "$task_base/person-static-pgsr-fresh-064" "$task_run" --out "$task_run/review/backgrounds" > "$task_run/alpha-review.log" 2>&1
"$task_py" audit_static_boundary.py --run "$task_run" --out "$task_run/review/coverage" > "$task_run/boundary-review.log" 2>&1
tail -n 1 "$task_run/alpha-review.log"
