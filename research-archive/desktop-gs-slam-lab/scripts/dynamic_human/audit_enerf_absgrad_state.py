from pathlib import Path
import json,hashlib,shutil
import torch,numpy as np
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=LAB/'scripts/dynamic_human'
OUT=LAB/'outputs/dynamic-human-enerf-absgrad-052';BASE=LAB/'outputs/dynamic-human-enerf-camera10-time-050/same_frame'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
for name,root in [('standard',BASE),('absgrad',OUT)]:
 run=root/'first-frame';cp=torch.load(run/'resume.pt',map_location='cpu',weights_only=False)
 assert cp['step']==6000 and all(torch.isfinite(v).all() for v in cp['params'].values())
 steps={k:[int(st['step']) for st in op['state'].values() if 'step' in st] for k,op in cp['optim'].items()}
 assert all(v==[6000] for v in steps.values())
 scales=cp['params']['scales'].exp().numpy();live=cp['params']['opacities'].sigmoid().numpy()>.05
 rows.append(dict(branch=name,updates=6000,points=len(scales),optimizer_steps=steps,checkpoint_sha256=sha(run/'resume.pt'),
    opacity_above005=int(live.sum()),scale_max_quantiles=np.quantile(scales[live].max(1),[.5,.9,.95,.99,1]).tolist(),
    aspect_ratio_quantiles=np.quantile(scales[live].max(1)/scales[live].min(1),[.5,.9,.95,.99,1]).tolist()))
 del cp
assert read(BASE/'first-frame/initial-and-sampling-audit.json')==read(OUT/'first-frame/initial-and-sampling-audit.json')
a=np.load(BASE/'first-frame/sampling-schedule.npz');b=np.load(OUT/'first-frame/sampling-schedule.npz')
assert all(np.array_equal(a[k],b[k]) for k in a.files)
assert all(sha(p)==sha(BASE/p.name) for p in OUT.glob('*.npz'))
assert read(OUT/'first-frame/absolute-gradient-check.json')['status']=='CUDA_absolute_gradient_buffer_present_finite_nonzero'
report=dict(status='paired6000updates_exact_initial_inputs_and6000draws_absgrad_buffer_verified',branches=rows,
    limits='Scale aspect ratios alone do not identify visible artifacts: very thin surfel-like Gaussians can have large ratios.',goal_complete=False)
(OUT/'paired-training-state-audit.json').write_text(json.dumps(report,indent=2)+'\n')
for name in ['audit_enerf_absgrad_state.py','compare_enerf_absgrad.py']:shutil.copy2(S/name,OUT/'source-snapshot'/name)
print(json.dumps(rows))
