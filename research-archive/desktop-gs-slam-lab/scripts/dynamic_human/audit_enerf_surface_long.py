from pathlib import Path
import json,hashlib,torch,numpy as np,shutil
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');OUT=LAB/'outputs/dynamic-human-enerf-surface-long-056';BASE=LAB/'outputs/dynamic-human-enerf-surface-gs-055/surface_regularized'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cp=torch.load(OUT/'first-frame/resume.pt',weights_only=False,map_location='cpu')
assert cp['step']==30000 and all(torch.isfinite(v).all() for v in cp['params'].values())
steps={k:[int(st['step']) for st in op['state'].values() if 'step' in st] for k,op in cp['optim'].items()}
assert all(v==[30000] for v in steps.values())
a=read(BASE/'first-frame/initial-and-sampling-audit.json');b=read(OUT/'first-frame/initial-and-sampling-audit.json');assert a['initial_parameters']==b['initial_parameters']
x=np.load(BASE/'first-frame/sampling-schedule.npz');y=np.load(OUT/'first-frame/sampling-schedule.npz')
assert all(np.array_equal(x[k],y[k][:6000]) for k in x.files)
assert all(sha(f)==sha(BASE/f.name) for f in OUT.glob('*.npz'))
assert read(OUT/'first-frame/surface-regularizer-gradient-check.json')['means_gradient_l1']>0
report=dict(status='30000updates_same_initial_inputs_first6000draws_and_finite_Adam_state_verified',
 step=30000,points=len(cp['params']['means']),checkpoint_sha256=sha(OUT/'first-frame/resume.pt'),optimizer_steps=steps,
 final_progress=read(OUT/'first-frame/progress.json'),comparison_qualification='Longer budget plus modified density and LR schedules, not one-factor iteration ablation.',goal_complete=False)
(OUT/'training-state-audit.json').write_text(json.dumps(report,indent=2)+'\n')
shutil.copy2(__file__,OUT/'source-snapshot'/Path(__file__).name)
print(json.dumps(report))
