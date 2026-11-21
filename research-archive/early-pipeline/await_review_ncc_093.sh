#!/bin/bash
set -euo pipefail
task_pair='/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/static-ncc-paired-fresh-093'
for task_attempt in $(seq 1 160); do
  if [[ -f "$task_pair/no_ncc/gaussian-run/result.json" ]]; then
    exec bash '/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/review_ncc_pair_093.sh'
  fi
  if [[ -f "$task_pair/no_ncc/gaussian-run/STOP" ]]; then
    echo 'Training was stopped; do not start acceptance review.'
    exit 2
  fi
  sleep 15
done
echo 'Training result unavailable after bounded wait; no review ran.'
exit 3
