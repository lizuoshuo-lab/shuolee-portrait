from pathlib import Path
import json,hashlib,shutil
import torch,numpy as np
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');OUT=LAB/'outputs/dynamic-human-enerf-surface-gs-055'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
roots=[OUT/'surface_only',OUT/'surface_regularized'];rows=[]
assert read(OUT/'runtime-smoke.json')['status'].startswith('isolated_2dgs_native_CUDA')
for root in roots:
 run=root/'first-frame';cp=torch.load(run/'resume.pt',weights_only=False,map_location='cpu')
 assert cp['step']==6000 and all(torch.isfinite(v).all() for v in cp['params'].values())
 steps={k:[int(st['step']) for st in op['state'].values() if 'step' in st] for k,op in cp['optim'].items()}
 assert all(v==[6000] for v in steps.values())
 rows.append(dict(branch=root.name,step=6000,points=len(cp['params']['means']),checkpoint_sha256=sha(run/'resume.pt'),optimizer_steps=steps,
    final_progress=read(run/'progress.json')))
 del cp
assert read(roots[0]/'first-frame/initial-and-sampling-audit.json')==read(roots[1]/'first-frame/initial-and-sampling-audit.json')
a=np.load(roots[0]/'first-frame/sampling-schedule.npz');b=np.load(roots[1]/'first-frame/sampling-schedule.npz')
assert all(np.array_equal(a[k],b[k]) for k in a.files)
assert all(sha(f)==sha(roots[1]/f.name) for f in roots[0].glob('*.npz'))
reg=read(roots[1]/'first-frame/surface-regularizer-gradient-check.json');assert reg['means_gradient_l1']>0
report=dict(status='paired6000updates_exact_surface_initial_inputs_draws_and_nonzero_regularizer_gradient_verified',branches=rows,
    regularizer_check=reg,goal_complete=False)
(OUT/'paired-training-state-audit.json').write_text(json.dumps(report,indent=2)+'\n')
for name in ['audit_enerf_surface_control.py','compare_enerf_surface_control.py']:
 shutil.copy2(LAB/'scripts/dynamic_human'/name,OUT/'source-snapshot'/name)
print(json.dumps(rows))
