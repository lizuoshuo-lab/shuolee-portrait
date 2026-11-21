#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export PYTHONDONTWRITEBYTECODE=1
task_py=/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python
task_lab=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
task_base="$task_lab/outputs/mini-splatting2-mip-depth-001/shuolee/current"
task_run="$task_base/person-static-alpha-fresh-069"
cd "$task_lab/repos/GaussianAvatars_SMPLX-shuolee"
"$task_py" audit_pgsr_static_delivery.py --run "$task_run" --reference "$task_base/person-static-pgsr-fresh-064/gaussian-run/point_cloud.ply" --label '069 boundary alpha supervision' > "$task_run/export-audit.log" 2>&1
"$task_py" review_static_alpha.py --runs "$task_base/person-static-pgsr-fresh-064" "$task_run" --out "$task_base/static-white-edge-audit-068/after" > "$task_run/alpha-review.log" 2>&1
"$task_py" audit_static_boundary.py --run "$task_run" --out "$task_base/static-white-edge-audit-068/boundary-after" > "$task_run/boundary-review.log" 2>&1
tail -n 1 "$task_run/alpha-review.log"
