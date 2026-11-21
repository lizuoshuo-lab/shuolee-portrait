"""Check real checkpoint against the unchanged baseline and weak-label schedule."""
from pathlib import Path
import json,hashlib,torch,numpy as np,shutil
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');R=L/'outputs/dynamic-human-enerf-actor-supervision-070';B=L/'outputs/dynamic-human-enerf-residual-graph-067/residual_graph';A=R/'residual_graph'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert read(A/'completion.json')['step']==15000
for f in ['initial-and-sampling-audit.json']:
 assert read(A/f)==read(B/f)
aa=np.load(A/'sampling-schedule.npz');bb=np.load(B/'sampling-schedule.npz');assert aa.files==bb.files and all(np.array_equal(aa[k],bb[k]) for k in aa.files)
pa=read(A/'protocol.json');pb=read(B/'protocol.json')
for k in ['updates','learning_rates','input_count','input_manifest_sha256','motion_sha256','renderer','selection']:assert pa[k]==pb[k]
ga=read(A/'graph-protocol.json');gb=read(B/'graph-protocol.json');assert {k:v for k,v in ga.items() if k!='actor_supervision'}==gb
assert ga['actor_supervision']['weight']==.05 and ga['actor_supervision']['review_sha256']==sha(R/'mask-visual-review.json') and ga['actor_supervision']['mask_sha256']==sha(R/'mask-preparation.json')
cams=read(R/'mask-visual-review.json')['accepted_cameras'];assert ga['actor_supervision']['cameras']==cams and len(cams)==13
used=(aa['frame']==750)&np.isin(aa['camera'],cams);cp=torch.load(A/'resume.pt',map_location='cpu',weights_only=False);assert cp['step']==15000
assert all([int(v['step']) for v in o['state'].values() if 'step' in v]==[15000] for o in cp['optim'].values())
seen=[r for r in cp['history'] if r['actor_loss']>0];assert len(seen)>0
report=dict(status='actual15000update_checkpoint_same105images_initialization_sampling_and_motion_verified',updates=15000,ownership_supervised_updates=int(used.sum()),supervised_cameras=cams,weight=.05,positive_ownership_loss_logged_steps=[r['step'] for r in seen],
 baseline_points=read(B/'completion.json')['points'],candidate_points=len(cp['params']['means']),checkpoint_sha256=sha(A/'resume.pt'),
 qualification='Additional13training-only SAM2core labels and class-balanced ownership loss are the intervention. Same update count is not same compute or final point count. Loggedloss evidence is sampled every100updates, while supervised update count is derived from verified executed schedule.',goal_complete=False)
(R/'training-protocol-audit.json').write_text(json.dumps(report,indent=2)+'\n');shutil.copy2(__file__,R/Path(__file__).name);print(json.dumps(report))
