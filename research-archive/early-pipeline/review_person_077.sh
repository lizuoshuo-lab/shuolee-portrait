#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
task_py=/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python
task_lab=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
task_base="$task_lab/outputs/mini-splatting2-mip-depth-001/shuolee/current"
task_run="$task_base/person-static-observed-correspondence-fresh-077"
"$task_py" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["steps"]==18000 and r["frames_used"]==175 and r["from_scratch"]' "$task_run/gaussian-run/result.json"
cd "$task_lab/repos/GaussianAvatars_SMPLX-shuolee"
"$task_py" -u audit_pgsr_static_delivery.py --run "$task_run" --reference "$task_base/person-static-fullscene-camera-fresh-075/gaussian-run/point_cloud.ply" --reference-label '075 fullscene baseline' --label '077 observed correspondence' > "$task_run/export-review.log" 2>&1
"$task_py" -u review_observed_person_077.py > "$task_run/same-camera-review.log" 2>&1
cd "$task_lab/repos/PGSR-shuolee"
"$task_py" -u review_observed_transfer_077.py > "$task_run/transfer-review.log" 2>&1
tail -n 1 "$task_run/transfer-review.log"
