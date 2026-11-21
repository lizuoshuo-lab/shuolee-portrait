from pathlib import Path
import json,hashlib,shutil
import numpy as np,torch
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');OUT=LAB/'outputs/dynamic-human-enerf-forearm-control-061';BASE=LAB/'outputs/dynamic-human-enerf-surface-gs-055/surface_regularized'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cp=torch.load(OUT/'first-frame/resume.pt',weights_only=False,map_location='cpu');assert cp['step']==6000 and all(torch.isfinite(v).all() for v in cp['params'].values())
steps={k:[int(st['step']) for st in op['state'].values() if 'step' in st] for k,op in cp['optim'].items()};assert all(v==[6000] for v in steps.values())
assert read(BASE/'first-frame/initial-and-sampling-audit.json')==read(OUT/'first-frame/initial-and-sampling-audit.json')
a=np.load(BASE/'first-frame/sampling-schedule.npz');b=np.load(OUT/'first-frame/sampling-schedule.npz');assert all(np.array_equal(a[k],b[k]) for k in a.files)
for f in BASE.glob('*.npz'):assert sha(f)==sha(OUT/f.name)
grad=read(OUT/'first-frame/forearm-gradient-check.json');assert grad['means_gradient_l1']>0 and grad['constrained_points']>0
report=dict(status='6000updates_same_initial_inputs_draws_and_optimizer_counts_verified',step=6000,points=len(cp['params']['means']),checkpoint_sha256=sha(OUT/'first-frame/resume.pt'),optimizer_steps=steps,forearm_gradient_check=grad,baseline='055surface_regularized6000',only_training_change='soft forearm point-to-tangent penalty',goal_complete=False)
(OUT/'training-state-audit.json').write_text(json.dumps(report,indent=2)+'\n');shutil.copy2(__file__,OUT/'source-snapshot'/Path(__file__).name)
print(json.dumps(report))
