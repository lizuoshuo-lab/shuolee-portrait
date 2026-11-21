#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export PYTHONDONTWRITEBYTECODE=1
task_py=/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python
task_lab=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
task_run="$task_lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/person-static-registered-camera-fresh-072"
task_checks='/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/check_registered_camera_round.py'
"$task_py" "$task_checks" >> "$task_run/controlled-check.log" 2>&1
test -f "$task_run/initialization-review.json"
cd "$task_lab/repos/PGSR-shuolee"
"$task_py" -u shuolee_static.py --run "$task_run" --mask-foreground-only > "$task_run/gaussian.log" 2>&1
"$task_py" "$task_checks" >> "$task_run/controlled-check.log" 2>&1
tail -n 1 "$task_run/gaussian.log"
