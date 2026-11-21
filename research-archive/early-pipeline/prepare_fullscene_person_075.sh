#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/GVHMR-shuolee
task_py=/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python
task_lab=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
task_run="$task_lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/person-static-fullscene-camera-fresh-075"
cd "$task_lab/repos/GVHMR-shuolee"
"$task_py" -u shuolee/fit_fresh_sequence.py --run "$task_run" --steps 1200 > "$task_run/fit.log" 2>&1
cd "$task_lab/repos/GaussianAvatars_SMPLX-shuolee"
"$task_py" -u prepare_fullscene_clothing_075.py > "$task_run/clothing-support.log" 2>&1
"$task_py" -u initialize_static_clothing_cloud.py --run "$task_run" > "$task_run/initialization.log" 2>&1
tail -n 1 "$task_run/initialization.log"
