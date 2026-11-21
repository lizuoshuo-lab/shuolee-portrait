#!/bin/bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
task_lab=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
task_run="$task_lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/person-static-native-align-fresh-113"
task_ref="$task_lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/person-static-shared-face-fresh-098"
task_py=/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python
"$task_py" -u - "$task_run" <<'PY'
from pathlib import Path
import json,time,sys,os
r=Path(sys.argv[1]);end=time.time()+3*3600
while not (r/'gaussian-run/result.json').exists():
    if (r/'gaussian-run/failure.json').exists():raise RuntimeError('Training stopped; no automatic export or restart')
    if time.time()>end:raise RuntimeError('Bounded wait expired; inspect existing run, do not restart')
    time.sleep(15)
d=json.loads((r/'gaussian-run/result.json').read_text());assert d['steps']==18000 and d['from_scratch'] and d['frames_used']==175
print('113 completed18k; begin read-only evaluation and compatible export',flush=True)
PY
cd "$task_lab/repos/PGSR-shuolee"
"$task_py" -u evaluate_native_person_113.py > "$task_run/evaluation.log" 2>&1
cd "$task_lab/repos/GaussianAvatars_SMPLX-shuolee"
"$task_py" -u audit_pgsr_static_delivery.py --run "$task_run" --reference "$task_ref/gaussian-run/point_cloud.ply" --reference-label '098 sharedXYZ18k' --label '113 nativealign18k' > "$task_run/export.log" 2>&1
"$task_py" -u -c 'import review_fullscene_person_075 as m;from pathlib import Path;import sys;out=Path(sys.argv[1])/"evaluation";models={};sources={};
for label,root in [("098sharedXYZ",Path(sys.argv[2])),("113nativeAlign",Path(sys.argv[1]))]:
 p=root/"gaussian-run/point_cloud.ply";sources[str(p)]=m.c.sha(p);model=m.GaussianModel(3);model.load_ply(str(p));models[label]=model
m.continuous(models,out)
assert all(m.c.sha(Path(p))==h for p,h in sources.items())
print("113 continuous comparison rendered; quality and actualOpenFile review still pending",flush=True)' "$task_run" "$task_ref" > "$task_run/continuous.log" 2>&1
"$task_py" -u - "$task_run" <<'PY'
from pathlib import Path
import json,sys,time
r=Path(sys.argv[1]);(r/'postprocess-complete.json').write_text(json.dumps(dict(completed=time.time(),evaluation=True,export=True,continuous=True,actual_viewer_open_verified=False,quality_accepted=False,pc00_deployed=False),indent=2))
print('113 postprocess complete; not a quality acceptance',flush=True)
PY
