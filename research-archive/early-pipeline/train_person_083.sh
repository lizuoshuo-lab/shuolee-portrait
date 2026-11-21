#!/bin/bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
task_py=/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python
task_lab=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
task_base="$task_lab/outputs/mini-splatting2-mip-depth-001/shuolee/current"
task_run="$task_base/person-static-human-surface-init-fresh-083"
"$task_py" -c 'import json,sys;assert json.load(open(sys.argv[1]))["allow_training"]' "$task_run/initialization-review.json"
cd "$task_lab/repos/PGSR-shuolee"
"$task_py" -u shuolee_static.py --run "$task_run" --mask-foreground-only --observed-correspondence "$task_base/fullscene-person-constraint-audit-076/observed-bank/pairs.npz" --observed-weight 0.002 > "$task_run/gaussian.log" 2>&1
tail -n 1 "$task_run/gaussian.log"
