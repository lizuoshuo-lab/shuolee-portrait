#!/bin/bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
task_py=/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python
task_lab=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
task_pair="$task_lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/static-ncc-paired-fresh-093"
cd "$task_lab/repos/PGSR-shuolee"
"$task_py" -u review_ncc_pair_093.py > "$task_pair/paired-review.log" 2>&1
cd "$task_lab/repos/GaussianAvatars_SMPLX-shuolee"
for task_arm in control no_ncc; do
  task_run="$task_pair/$task_arm"
  if [[ ! -f "$task_run/gaussian-run/export-audit/audit.json" ]]; then
    "$task_py" -u audit_pgsr_static_delivery.py --run "$task_run" --reference "$task_lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/person-static-human-surface-init-fresh-083/gaussian-run/point_cloud.ply" --reference-label '083 legacy reference' --label "093 $task_arm" > "$task_run/export-review.log" 2>&1
  fi
done
"$task_py" -u -c 'import review_fullscene_person_075 as m;from pathlib import Path;import sys;m.A=Path(sys.argv[1])/"control";m.R=Path(sys.argv[1])/"no_ncc";m.LABELS=("NCC0.15","NCC0");m.PROTECT_ONLY=True;m.main()' "$task_pair" > "$task_pair/continuous-review.log" 2>&1
cd "$task_lab/repos/PGSR-shuolee"
for task_arm in control no_ncc; do
  "$task_py" -u -c 'import audit_fullscene_correspondence_076 as m;from pathlib import Path;import sys;m.R=Path(sys.argv[1]);m.O=m.R/"review/correspondence";m.main()' "$task_pair/$task_arm" > "$task_pair/$task_arm/transfer-review.log" 2>&1
  "$task_py" -u -c 'import review_surface_depth_079 as m;from pathlib import Path;import sys;m.R=Path(sys.argv[1]);m.main()' "$task_pair/$task_arm" > "$task_pair/$task_arm/depth-review.log" 2>&1
done
