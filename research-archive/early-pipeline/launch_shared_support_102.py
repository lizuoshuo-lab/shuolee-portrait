from pathlib import Path
import ast, json, hashlib, shutil, datetime

L=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab');B=L/'outputs/mini-splatting2-mip-depth-001/shuolee'
R=B/'current/person-static-shared-support-fresh-102';P=L/'repos/PGSR-shuolee';G=L/'repos/GaussianAvatars_SMPLX-shuolee'
BUILD=Path(__file__).parent
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def linux(p):return '/mnt/c/'+str(p).replace('\\','/').split(':/',1)[1]
assert not (R/'gaussian-run/current.pt').exists() and not (R/'gaussian-run/loss.jsonl').exists()
assert read(B/'current/shared-finite-support-preflight-102/result.json')['numerical_preflight_passed']
assert read(R/'clothing-init/audit.json')['from_scratch']
assert read(R/'shared-targets/result.json')['available_points']==50
save(R/'initialization-review.json',dict(reviewed_at=datetime.datetime.now().astimezone().isoformat(),
    actual_images=['clothing-init/initialization-review.jpg','shared-targets/shared-points-000100.jpg'],
    observations=['All six fixed views show a finite clothed whole-body hull and supported head initialization.',
                  'Raw100 shared-point projections have visible several-pixel residual around nose/lips and eyebrows; the estimates remain uncertain and are not camera or geometry ground truth.', 'The initial cloud has finite clothed outline in six fixed views. This is an initialization check only, not an acceptance of novel-view geometry.'],
    allow_single_preregistered_training=True,quality_accepted=False))
sources=[P/'shuolee_static.py',P/'shuolee_pixel_center.py',P/'shared_face_world.py',P/'surface_concentration.py',
         P/'gaussian_renderer/surface.py',P/'scene/gaussian_model.py',P/'observed_correspondence.py',
         P/'preflight_shared_support_102.py',P/'shared_face_support.py',P/'gaussian_renderer/shared_support.py',P/'evaluate_shared_support_102.py',G/'prepare_shared_support_102.py',G/'build_shared_face_points_097.py',
         G/'initialize_ncc_pair_093.py',G/'fit_head_surface_082.py',G/'initialize_static_clothing_cloud.py',
         G/'prepare_fullscene_clothing_075.py',L/'repos/GVHMR-shuolee/shuolee/fit_fresh_sequence.py',
         BUILD/'prepare_shared_support_102.sh',R/'protocol.json',Path(__file__)]
snapshot=R/'source-final';assert not (snapshot/'manifest.json').exists()
manifest=[]
for src in sources:
    if src.suffix=='.py':ast.parse(src.read_text(encoding='utf-8-sig'))
    dest=snapshot/src.name;assert not dest.exists();shutil.copy2(src,dest)
    manifest.append(dict(path=str(src),copy=dest.name,sha256=sha(src)))
save(snapshot/'manifest.json',manifest)
cmd=f'''#!/bin/bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
cd "{linux(P)}"
/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python -u shuolee_static.py --run "{linux(R)}" --steps 18000 --height 1536 --mask-foreground-only --observed-correspondence "{linux(B)}/current/fullscene-person-constraint-audit-076/observed-bank/pairs.npz" --observed-weight 0.002 --pixel-centers-consistent --ncc-factor 1 --shared-face-world "{linux(R)}/shared-targets/shared-face-points.npz" --shared-face-weight 0.002 --shared-face-mode finite_support > "{linux(R)}/gaussian.log" 2>&1
tail -n 2 "{linux(R)}/gaussian.log"
'''
(BUILD/'train_shared_support_102.sh').write_text(cmd,encoding='utf-8',newline='\n')
state=read(B/'work-status.json');state['shared_finite_support_102']=dict(available_points=50,observations=5815,cameras=175,
    numerical_preflight_passed=True,consumer=str(P/'shared_face_support.py'),support_mode='finite_support',gaussian_quality_accepted=False)
state['latest_local_execution'].update(round='102',run=str(R),state='fresh_initialization_reviewed_starting_training',running=True,steps=18000)
save(B/'work-status.json',state)
print(json.dumps(dict(ready=True,steps=18000,initial_points=200000,run=str(R))))
