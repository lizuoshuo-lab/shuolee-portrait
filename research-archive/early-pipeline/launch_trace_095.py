from pathlib import Path
import json, hashlib, shutil, datetime

L=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab')
B=L/'outputs/mini-splatting2-mip-depth-001/shuolee'
R=B/'current/whole-body-early-trace-fresh-095'
P=L/'repos/PGSR-shuolee'
G=L/'repos/GaussianAvatars_SMPLX-shuolee'
BUILD=Path(__file__).parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def linux(p):return '/mnt/c/'+str(p).replace('\\','/').split(':/',1)[1]
assert not (R/'gaussian-run/current.pt').exists()
assert not (R/'gaussian-run/loss.jsonl').exists()
check=read(R/'trace-noninterference.json');assert check['passed']
for path,digest in check['production_sources'].items():
    native=Path('C:/'+path.split('/mnt/c/',1)[1]);assert sha(native)==digest,path
save(R/'initialization-review.json',dict(reviewed_at=datetime.datetime.now().astimezone().isoformat(),
    actual_images=['clothing-init/initialization-review.jpg','clothing-init/patch-review-000070.jpg'],
    observations=['Six views of a shared clothed silhouette surface are present, with finite torso, limbs and head.',
                  'Neutral facial support projected from freshly fitted official template. This is not ground-truth surface or final geometry.'],
    approved_for_recording_diagnostic_only=True,geometry_accepted=False,quality_accepted=False,
    fresh_helper_provenance='095 recomputes fitting/hull using the093 initializer recipe; inherited fresh093 figure labels identify the helper, not reused learned arrays.'))
sources=[P/'shuolee_static.py',P/'trace_whole_body_095.py',P/'check_trace_noninterference_095.py',
         P/'scene/gaussian_model.py',P/'scene/gaussian_model_095.py',P/'shuolee_pixel_center.py',
         P/'observed_correspondence.py',P/'audit_whole_body_alignment_094.py',
         G/'prepare_whole_body_trace_095.py',G/'initialize_ncc_pair_093.py',
         G/'initialize_static_clothing_cloud.py',G/'prepare_fullscene_clothing_075.py',
         L/'repos/GVHMR-shuolee/shuolee/fit_fresh_sequence.py',
         BUILD/'prepare_whole_body_trace_095.sh',Path(__file__)]
snapshot=R/'source-final';snapshot.mkdir(exist_ok=True)
manifest=[]
for src in sources:
    dest=snapshot/src.name;shutil.copy2(src,dest)
    assert sha(src)==sha(dest);manifest.append(dict(path=str(src),copy=dest.name,sha256=sha(src)))
save(R/'source-final/manifest.json',manifest)
state=read(B/'work-status.json');state['local_training_running']=False
state['latest_local_execution']=dict(round='095',state='fresh_initialization_and_trace_verified',running=False,
    run=str(R),steps=1800,nominal_schedule_steps=18000,quality_accepted=False,
    diagnostic='Full-body early Adam updates and exact clone/split/prune ancestry; unchanged objective.')
save(B/'work-status.json',state)
cmd=f'''#!/bin/bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
cd "{linux(P)}"
/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python -u shuolee_static.py --run "{linux(R)}" --steps 18000 --height 1536 --mask-foreground-only --observed-correspondence "{linux(B)}/current/fullscene-person-constraint-audit-076/observed-bank/pairs.npz" --observed-weight 0.002 --pixel-centers-consistent --ncc-factor 1 --whole-body-trace --stop-after 1800 > "{linux(R)}/gaussian.log" 2>&1
tail -n 2 "{linux(R)}/gaussian.log"
'''
(BUILD/'train_whole_body_trace_095.sh').write_text(cmd,encoding='utf-8',newline='\n')
print(json.dumps(dict(ready=True,sources=len(sources),run=str(R),diagnostic_stop=1800)))
