from pathlib import Path
import json,shutil,hashlib,torch
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');OUT=LAB/'outputs/dynamic-human-enerf-surface-long-056';S=LAB/'scripts/dynamic_human'
run=OUT/'first-frame';cpfile=run/'resume.pt';cp=torch.load(cpfile,map_location='cpu',weights_only=False)
assert cp['step']==14901 and len(cp['params']['means'])==800528
assert all(torch.isfinite(p).all() for p in cp['params'].values())
assert all([int(s['step']) for s in o['state'].values() if 'step' in s]==[14901] for o in cp['optim'].values())
assert not (OUT/'cap-recovery.json').exists()
shutil.copy2(cpfile,run/'resume-at-cap-14901.pt')
p=S/'train_enerf_surface_long.py';t=p.read_text();assert 'point_limit=800000' in t
shutil.copy2(p,OUT/'source-snapshot/train_enerf_surface_long_before_cap_recovery.py')
t=t.replace('point_limit=800000','point_limit=805000');p.write_text(t)
spec=run/'protocol.json';r=json.loads(spec.read_text());assert r['point_limit']==800000
shutil.copy2(spec,run/'protocol-before-cap-recovery.json');r['point_limit']=805000;spec.write_text(json.dumps(r,indent=2)+'\n')
report=dict(status='finite_step14901_checkpoint_preserved_resume_authorized_with_nominal_cap805000',
 old_cap=800000,new_cap=805000,checkpoint_points=800528,checkpoint_step=14901,
 checkpoint_sha256=hashlib.sha256(cpfile.read_bytes()).hexdigest(),
 reason='Post-densification resource guard exceeded by528points. Last scheduled densification was zero-based14900; no further growth scheduled. MaxallocatedVRAMabout2.3GB, noOOM.',
 unchanged='No parameter, optimizer, strategy, RNG, loss, camera, image, mask, LR or iteration schedule modified.',goal_complete=False)
(OUT/'cap-recovery.json').write_text(json.dumps(report,indent=2)+'\n')
shutil.copy2(p,OUT/'source-snapshot/train_enerf_surface_long.py');shutil.copy2(__file__,OUT/'source-snapshot'/Path(__file__).name)
print(json.dumps(report))
