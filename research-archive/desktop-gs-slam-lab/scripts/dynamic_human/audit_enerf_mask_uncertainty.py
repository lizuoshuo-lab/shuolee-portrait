from pathlib import Path
import json,hashlib,shutil
import numpy as np,torch
from PIL import Image
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=LAB/'scripts/dynamic_human'
BASE=LAB/'outputs/dynamic-human-enerf-camera10-time-050/same_frame';OUT=LAB/'outputs/dynamic-human-enerf-mask-uncertainty-053'
MASK=LAB/'outputs/dynamic-human-enerf-mask-quality-038/train15'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inputrows=[]
for f in sorted(OUT.glob('train-camera-*.npz')):
 a=np.load(BASE/f.name);b=np.load(f);c=f.stem[-2:];sam=np.array(Image.open(MASK/f'camera{c}-candidate.png'))>0
 unknown=(a['matte']>.5)&~sam
 assert np.array_equal(b['valid'],a['valid']&~unknown)
 assert all(np.array_equal(a[k],b[k]) for k in a.files if k!='valid')
 inputrows.append(dict(camera=c,ignored_pixels=int((unknown&a['valid']).sum()),sha256=sha(f)))
assert sha(BASE/'train-visual-hull-init.npz')==sha(OUT/'train-visual-hull-init.npz')
assert read(BASE/'first-frame/initial-and-sampling-audit.json')==read(OUT/'first-frame/initial-and-sampling-audit.json')
a=np.load(BASE/'first-frame/sampling-schedule.npz');b=np.load(OUT/'first-frame/sampling-schedule.npz')
assert all(np.array_equal(a[k],b[k]) for k in a.files)
rows=[]
for name,root in [('baseline',BASE),('uncertain_ignored',OUT)]:
 cp=torch.load(root/'first-frame/resume.pt',map_location='cpu',weights_only=False)
 assert cp['step']==6000 and all(torch.isfinite(v).all() for v in cp['params'].values())
 steps={k:[int(s['step']) for s in op['state'].values() if 'step' in s] for k,op in cp['optim'].items()}
 assert all(v==[6000] for v in steps.values())
 rows.append(dict(branch=name,step=6000,points=len(cp['params']['means']),optimizer_steps=steps,sha256=sha(root/'first-frame/resume.pt')))
 del cp
r=dict(status='paired6000updates_exact_initial_sampling_only_disagreement_validity_changed',branches=rows,inputs=inputrows,goal_complete=False)
(OUT/'paired-training-state-audit.json').write_text(json.dumps(r,indent=2)+'\n')
for n in ['audit_enerf_mask_uncertainty.py','compare_enerf_mask_uncertainty.py','run_enerf_mask_uncertainty.sh']:shutil.copy2(S/n,OUT/'source-snapshot'/n)
print(json.dumps(rows))
