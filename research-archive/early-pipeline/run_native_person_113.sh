#!/bin/bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
task_lab=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
task_run="$task_lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/person-static-native-align-fresh-113"
cd "$task_lab/repos/PGSR-shuolee"
/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python -u - "$task_run" <<'PY'
from pathlib import Path
import json,hashlib,shutil,time,ast
r=Path(__import__('sys').argv[1]);p=json.loads((r/'protocol.json').read_text());check=json.loads((r/'native-volume-preflight/result.json').read_text())
assert check['passed'] and check['optimizer_updates']==0
assert not (r/'gaussian-run/started.json').exists()
for f in ['shuolee_static.py','native_person_alignment_113.py','check_native_person_113.py']:
    ast.parse(Path(f).read_text());shutil.copyfile(f,r/'source-final'/f)
p['training']['admitted']=True;p['training']['admission_time']=time.time()
p['training']['geometry']='Native RGB contribution + finite3DGaussian conditional raymode; original plane version failed support and is preserved'
p['training']['support_preflight_valid_fraction']=check['valid_fraction'];p['training']['admission_file']='native-volume-preflight/result.json'
p['training']['support_safeguard']='At every500steps stop with checkpoint if validqueryfraction<90percent; never relax the threshold during training'
p['training']['quality_accepted']=False
(r/'protocol.json').write_text(json.dumps(p,ensure_ascii=False,indent=2))
PY
/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python -u shuolee_static.py \
  --run "$task_run" --steps 18000 --height 1536 --mask-foreground-only \
  --pixel-centers-consistent \
  --observed-correspondence "$task_lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/fullscene-person-constraint-audit-076/observed-bank/pairs.npz" \
  --native-person-alignment "$task_run/shared-targets/native-targets.npz" > "$task_run/gaussian.log" 2>&1
