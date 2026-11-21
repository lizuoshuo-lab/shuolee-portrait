from pathlib import Path
import json, shutil, hashlib
LAB=Path(__file__).resolve().parents[2]; S=LAB/'scripts/dynamic_human'
BASE=LAB/'outputs/dynamic-human-enerf-surface-gs-055'; OUT=LAB/'outputs/dynamic-human-enerf-surface-long-056'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,x:p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
assert not OUT.exists(), 'Preserve existing experiment'
OUT.mkdir(); source=BASE/'surface_regularized'
assert read(BASE/'paired-training-state-audit.json')['status'].startswith('paired6000')
for f in source.glob('*.npz'):shutil.copy2(f,OUT/f.name)
for name in ['geometry-audit.json','initialization.json','input-audit.json']:shutil.copy2(source/name,OUT/name)
save(OUT/'protocol.json',dict(status='fixed_before_training',method='2DGS surface regularization with longer training and density schedule',
    initialization='Same 52109 training-only hull surfels, original cameras/mattes/RGB and PCA orientations as055. Fresh optimizer, seed73.',
    updates=30000,densification_start=400,densification_stop_exclusive=15000,opacity_reset_every=3000,point_limit=800000,
    position_lr='Original initial rate times 0.01**(update/30000); other rates unchanged',
    regularization='Same .01 distortion from zero-based600 and .05 normal from1400, eroded foreground support',
    contrast='Training budget, position LR decay and density schedule jointly changed versus055. Not a one-factor iteration ablation.',
    evaluation='Fixed final30000 with unchanged six development cameras and17predeclared regions. Repeatedly inspected heldout cameras are development views, not fresh test evidence.',
    limits='Single frame750 foreground only. No full-scene or temporal clarity acceptance.',goal_complete=False))
t=(S/'train_enerf_surface_control.py').read_text(encoding='utf8')
first=t.index('from pathlib import Path')
t='"""Native-resolution 2DGS 30k surface training; protocol in experiment056."""\n'+t[first:]
t=t.replace('min(4000,a.steps-1000)','min(15000,a.steps-1000)').replace('reset_every=2000','reset_every=3000')
t=t.replace("point_limit=500000","point_limit=800000").replace("method='gsplat1.5.3 2DGaussian surface control'","method='gsplat1.5.3 2DGaussian surface 30000update schedule'")
t=t.replace('time.time()', 'time.monotonic()')
compile(t,'train_enerf_surface_long.py','exec');(S/'train_enerf_surface_long.py').write_text(t,encoding='utf8')
e=(S/'evaluate_enerf_surface_control.py').read_text(encoding='utf8').replace('6000','30000')
e=e.replace('paired050training-observation control, same model family and fixed target/background schedules.','056extended optimization schedule, compared to0556000update regularized endpoint; development cameras.')
e=e.replace('Six native views of048','Six native views of056').replace('NATIVE_3DGS','NATIVE_2DGS')
compile(e,'evaluate_enerf_surface_long.py','exec');(S/'evaluate_enerf_surface_long.py').write_text(e,encoding='utf8')
shell='''#!/bin/bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
export ENERF_RUN_DIR=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/dynamic-human-enerf-surface-long-056
python /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human/train_enerf_surface_long.py --out "$ENERF_RUN_DIR" --steps 30000 --surface-reg
python /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human/evaluate_enerf_surface_long.py
'''
(S/'run_enerf_surface_long.sh').write_text(shell,encoding='utf8',newline='\n')
snap=OUT/'source-snapshot';snap.mkdir()
for name in ['prepare_enerf_surface_long.py','train_enerf_surface_long.py','evaluate_enerf_surface_long.py','run_enerf_surface_long.sh']:shutil.copy2(S/name,snap/name)
save(OUT/'input-provenance.json',dict(source=str(source),files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.npz')}))
print('056 fresh 30000update surface training prepared')
