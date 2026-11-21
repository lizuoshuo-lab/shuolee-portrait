from pathlib import Path
import json,hashlib,torch,numpy as np,shutil
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');OUT=LAB/'outputs/dynamic-human-enerf-camera10-time-050'
read=lambda p:json.loads(p.read_text())
save=lambda p,x:p.write_text(json.dumps(x,indent=2)+'\n')
rows=[];initial=[];schedules=[]
for branch in ['same_frame','camera10_previous_frame']:
 root=OUT/branch;run=root/'first-frame';cp=torch.load(run/'resume.pt',map_location='cpu',weights_only=False)
 assert cp['step']==6000 and read(run/'completion.json')['step']==6000
 assert all(torch.isfinite(v).all() for v in cp['params'].values())
 steps={k:[int(s['step']) for s in v['state'].values() if 'step' in s] for k,v in cp['optim'].items()}
 assert all(v==[6000] for v in steps.values()),steps
 assert len(cp['params']['means'])==read(run/'completion.json')['points']
 initial.append(read(run/'initial-and-sampling-audit.json'));z=np.load(run/'sampling-schedule.npz')
 assert not set(z['camera'])&{'00','06','12'} and len(z['camera'])==6000
 schedules.append({k:z[k] for k in z.files})
 rows.append(dict(branch=branch,updates=6000,points=len(cp['params']['means']),optimizer_steps=steps,
  checkpoint_sha256=hashlib.sha256((run/'resume.pt').read_bytes()).hexdigest()))
 del cp
assert initial[0]==initial[1]
assert all(np.array_equal(schedules[0][k],schedules[1][k]) for k in schedules[0])
assert read(OUT/'paired-input-audit.json')['status'].startswith('only_camera10_RGB_and_mask_changed')
save(OUT/'independent-state-audit.json',dict(status='paired6000updates_exact_initial_parameters_and_all6000target_background_draws_verified',
  branches=rows,initial=initial[0],goal_complete=False))
for n in ['run_enerf_camera10_time.sh','evaluate_enerf_camera10_time.py','audit_enerf_camera10_time_state.py','prepare_enerf_camera10_regions.py']:
 shutil.copy2(LAB/'scripts/dynamic_human'/n,OUT/'source-snapshot'/n)
print(json.dumps(rows))
