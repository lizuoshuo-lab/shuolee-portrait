#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export PYTHONDONTWRITEBYTECODE=1
task_py=/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python
task_lab=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
task_base="$task_lab/outputs/mini-splatting2-mip-depth-001/shuolee/current"
cd "$task_lab/repos/PGSR-shuolee"
"$task_py" audit_boundary_objective.py --run "$task_base/person-static-pgsr-fresh-064" --out "$task_base/static-white-edge-audit-068/objective-diagnosis"
