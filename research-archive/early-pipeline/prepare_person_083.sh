#!/bin/bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/GVHMR-shuolee
task_py=/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python
task_lab=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
task_run="$task_lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/person-static-human-surface-init-fresh-083"
cd "$task_lab/repos/GaussianAvatars_SMPLX-shuolee"
"$task_py" -u prepare_human_surface_083.py
"$task_py" -u initialize_human_surface_083.py head > "$task_run/head-fit.log" 2>&1
cd "$task_lab/repos/GVHMR-shuolee"
"$task_py" -u shuolee/fit_fresh_sequence.py --run "$task_run" --steps 1200 > "$task_run/fit.log" 2>&1
cd "$task_lab/repos/GaussianAvatars_SMPLX-shuolee"
"$task_py" -u -c 'import prepare_fullscene_clothing_075 as m; from pathlib import Path; import sys; m.R=Path(sys.argv[1]); m.main()' "$task_run" > "$task_run/clothing-support.log" 2>&1
"$task_py" -u initialize_static_clothing_cloud.py --run "$task_run" > "$task_run/initialization.log" 2>&1
"$task_py" -u initialize_human_surface_083.py patch > "$task_run/face-patch.log" 2>&1
tail -n 1 "$task_run/face-patch.log"
