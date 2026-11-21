#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export PYTHONDONTWRITEBYTECODE=1
task_lab=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
task_py=/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python
case "$1" in
  065) task_name=person-static-no-scale-fresh-065; task_label='065 no min-scale penalty';;
  066) task_name=person-static-no-multiview-fresh-066; task_label='066 no multi-view losses';;
  *) exit 2;;
esac
task_run="$task_lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/$task_name"
cd "$task_lab/repos/GaussianAvatars_SMPLX-shuolee"
"$task_py" audit_pgsr_static_delivery.py --run "$task_run" --reference "$task_lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/person-static-pgsr-fresh-064/gaussian-run/point_cloud.ply" --label "$task_label" > "$task_run/export-audit.log" 2>&1
tail -n 1 "$task_run/export-audit.log"
