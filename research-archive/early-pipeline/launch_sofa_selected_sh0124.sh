#!/usr/bin/env bash
set -uo pipefail
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
task_dir='/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124/sofa-selected-sh0-pipeline124'
test ! -e "$task_dir/gaussian-run" || exit 73
/opt/gs-slam-lab/shuolee-portrait-clean-001/env/bin/python -B '/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/train_sofa_selected_sh0124.py' train > "$task_dir/train.log" 2>&1
task_rc=$?
printf '{"exit_code":%s,"finished_utc":"%s"}\n' "$task_rc" "$(date -u +%FT%TZ)" > "$task_dir/process-exit.json"
tail -n 12 "$task_dir/train.log"
exit "$task_rc"
