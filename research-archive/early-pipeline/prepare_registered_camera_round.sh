#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/GVHMR-shuolee
task_py=/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python
task_lab=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
task_run="$task_lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/person-static-registered-camera-fresh-072"
cd "$task_lab/repos/GVHMR-shuolee"
"$task_py" -u shuolee/fit_fresh_sequence.py --run "$task_run" --steps 1200 > "$task_run/fit.log" 2>&1
"$task_py" -u shuolee/clothing_surface_support.py --run "$task_run" > "$task_run/clothing-support.log" 2>&1
cd "$task_lab/repos/GaussianAvatars_SMPLX-shuolee"
"$task_py" -u initialize_static_clothing_cloud.py --run "$task_run" > "$task_run/initialization.log" 2>&1
"$task_py" '/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/check_registered_camera_round.py' > "$task_run/controlled-check.log" 2>&1
tail -n 1 "$task_run/controlled-check.log"
