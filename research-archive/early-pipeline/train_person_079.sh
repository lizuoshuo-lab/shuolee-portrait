#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
task_py=/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python
task_lab=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
task_base="$task_lab/outputs/mini-splatting2-mip-depth-001/shuolee/current"
task_run="$task_base/person-static-surface-consistency-fresh-079"
test -f "$task_run/initialization-review.json"
cd "$task_lab/repos/PGSR-shuolee"
"$task_py" -u shuolee_static.py --run "$task_run" --mask-foreground-only --observed-correspondence "$task_base/fullscene-person-constraint-audit-076/observed-bank/pairs.npz" --observed-weight 0.002 --surface-weight 0.01 > "$task_run/gaussian.log" 2>&1
tail -n 1 "$task_run/gaussian.log"
