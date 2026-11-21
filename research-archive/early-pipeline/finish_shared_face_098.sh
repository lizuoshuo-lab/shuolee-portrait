#!/bin/bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
task_lab=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
task_run="$task_lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/person-static-shared-face-fresh-098"
task_ref="$task_lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/static-ncc-paired-fresh-093/control"
task_py=/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python
cd "$task_lab/repos/PGSR-shuolee"
"$task_py" -u evaluate_shared_face_098.py > "$task_run/evaluation.log" 2>&1
tail -n 1 "$task_run/evaluation.log"
cd "$task_lab/repos/GaussianAvatars_SMPLX-shuolee"
"$task_py" -u audit_pgsr_static_delivery.py --run "$task_run" --reference "$task_ref/gaussian-run/point_cloud.ply" --reference-label '093 control18k' --label '098 shared XYZ18k' > "$task_run/export.log" 2>&1
"$task_py" -u -c 'import review_fullscene_person_075 as m;from pathlib import Path;import sys;out=Path(sys.argv[1])/"evaluation";models={};sources={};
for label,root in [("093control",Path(sys.argv[2])),("098sharedXYZ",Path(sys.argv[1]))]:
 p=root/"gaussian-run/point_cloud.ply";sources[str(p)]=m.c.sha(p);model=m.GaussianModel(3);model.load_ply(str(p));models[label]=model
m.continuous(models,out)
assert all(m.c.sha(Path(p))==h for p,h in sources.items())
print("Continuous comparison rendered; visual review pending",flush=True)' "$task_run" "$task_ref" > "$task_run/continuous.log" 2>&1
tail -n 1 "$task_run/continuous.log"
