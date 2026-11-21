#!/usr/bin/env bash
set -euo pipefail
task_pid=396
task_run=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/person-static-observed-correspondence-fresh-077
while test -r "/proc/$task_pid/cmdline"; do
  if ! tr '\0' ' ' < "/proc/$task_pid/cmdline" | grep -q 'shuolee_static.py.*person-static-observed-correspondence-fresh-077'; then break; fi
  sleep 5
done
test -f "$task_run/gaussian-run/result.json"
exec bash '/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/review_person_077.sh'
