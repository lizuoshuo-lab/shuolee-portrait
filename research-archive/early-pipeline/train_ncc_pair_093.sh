#!/bin/bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
task_py=/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python
task_lab=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
task_base="$task_lab/outputs/mini-splatting2-mip-depth-001/shuolee/current"
task_arm="$1"
case "$task_arm" in control) task_factor=1 ;; no_ncc) task_factor=0 ;; *) exit 2 ;; esac
task_run="$task_base/static-ncc-paired-fresh-093/$task_arm"
"$task_py" -c 'import json,sys;assert json.load(open(sys.argv[1]))["allow_training"]' "$task_run/initialization-review.json"
"$task_py" - "$task_run" "$task_base" "$task_arm" <<'PY'
import hashlib,json,sys
from pathlib import Path
r,b,arm=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
for source in json.loads((r/'source-snapshot.json').read_text()):
    assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256'],source['path']
p=b.parent/'work-status.json';s=json.loads(p.read_text());s['latest_local_execution']=dict(round='093',arm=arm,run=str(r),state='native_preflight',running=True,gaussian_training=False,steps=18000)
p.write_text(json.dumps(s,ensure_ascii=False,indent=2))
PY
cd "$task_lab/repos/PGSR-shuolee"
"$task_py" -u shuolee_static.py --run "$task_run" --mask-foreground-only --observed-correspondence "$task_base/fullscene-person-constraint-audit-076/observed-bank/pairs.npz" --observed-weight 0.002 --pixel-centers-consistent --ncc-factor "$task_factor" --trace-surface > "$task_run/gaussian.log" 2>&1
tail -n 1 "$task_run/gaussian.log"
