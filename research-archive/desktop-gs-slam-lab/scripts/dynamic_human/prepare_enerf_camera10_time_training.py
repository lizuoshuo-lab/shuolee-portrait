"""Create two complete training inputs, paired sampling, and frozen source copies."""
from pathlib import Path
import json,hashlib,shutil,numpy as np
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=LAB/'scripts/dynamic_human'
BASE=LAB/'outputs/dynamic-human-enerf-representation-048';OUT=LAB/'outputs/dynamic-human-enerf-camera10-time-050'
assert json.loads((OUT/'input-mask/summary.json').read_text())['status'].startswith('exact750SAM_mask_RGB_reproduced')
assert (OUT/'input-mask/visual-review.json').exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
save=lambda p,x:p.write_text(json.dumps(x,indent=2)+'\n')
for branch in ['same_frame','camera10_previous_frame']:
 root=OUT/branch;root.mkdir(exist_ok=True)
 assert not (root/'first-frame/completion.json').exists()
 for name in ['geometry-audit.json','train-visual-hull-init.npz','initialization.json']:
  shutil.copy2(BASE/name,root/name)
 for src in BASE.glob('train-camera-*.npz'):
  chosen=OUT/'input-mask/frame749-training.npz' if branch=='camera10_previous_frame' and src.name=='train-camera-10.npz' else src
  shutil.copy2(chosen,root/src.name)
 assert sha(root/'train-visual-hull-init.npz')==sha(BASE/'train-visual-hull-init.npz')
 save(root/'input-audit.json',dict(records=[dict(camera=p.stem[-2:],sha256=sha(p)) for p in sorted(root.glob('train-camera-*.npz'))]))
records=[]
for source in sorted((OUT/'same_frame').glob('train-camera-*.npz')):
 other=OUT/'camera10_previous_frame'/source.name;a=np.load(source);b=np.load(other)
 for key in ['K','w2c','valid']:assert np.array_equal(a[key],b[key])
 changed=[key for key in a.files if not np.array_equal(a[key],b[key])]
 assert changed==(['rgb','matte'] if source.name=='train-camera-10.npz' else [])
 records.append(dict(camera=source.stem[-2:],changed_arrays=changed,before_sha256=sha(source),after_sha256=sha(other)))
save(OUT/'paired-input-audit.json',dict(status='only_camera10_RGB_and_mask_changed_all_camera_matrices_validity_and_initialization_identical',
 initial_shell_sha256=sha(BASE/'train-visual-hull-init.npz'),records=records,goal_complete=False))

t=(S/'train_enerf_first_frame.py').read_text()
t=t.replace('import argparse,json,time,os,random,sys','import argparse,json,time,os,random,sys,hashlib')
needle="    start=0;history=[];elapsed=0"
insert="""    # Precompute independent observation/background schedules so a change in
    # densification RNG consumption cannot change the paired observations.
    view_rng=random.Random(73)
    view_schedule=[view_rng.randrange(len(obs)) for _ in range(a.steps)]
    bg_schedule=torch.rand(a.steps,3,generator=torch.Generator(device='cpu').manual_seed(9073))
    np.savez_compressed(run/'sampling-schedule.npz',camera=np.array([obs[j]['id'] for j in view_schedule]),background=bg_schedule.numpy())
    initial={k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in params.items()}
    save(run/'initial-and-sampling-audit.json',dict(initial_parameters=initial,initial_gaussians=len(params['means']),
        target_schedule_sha256=hashlib.sha256(np.array(view_schedule,dtype=np.int32).tobytes()).hexdigest(),
        background_schedule_sha256=hashlib.sha256(bg_schedule.numpy().tobytes()).hexdigest(),
        qualification='Same initial tensors and fixed target/background draws. Densification can legitimately diverge after changed input.',goal_complete=False))
    bg_schedule=bg_schedule.cuda()
    start=0;history=[];elapsed=0"""
assert needle in t;t=t.replace(needle,insert)
t=t.replace("o=obs[random.randrange(len(obs))];x0,y0,x1,y1=o['box'];bg=torch.rand(3,device='cuda');degree=min(2,i//1000)",
 "o=obs[view_schedule[i]];x0,y0,x1,y1=o['box'];bg=bg_schedule[i];degree=min(2,i//1000)")
t=t.replace("method='gsplat 1.3.0 DefaultStrategy 3DGS foreground first-frame diagnostic'", "method='gsplat1.3DefaultStrategy paired camera10 observation control with fixed target/background schedules'")
compile(t,'train_enerf_camera10_time.py','exec');(S/'train_enerf_camera10_time.py').write_text(t)
evaluation=(S/'evaluate_enerf_representation.py').read_text()
evaluation=evaluation.replace('import json,hashlib,torch,numpy as np','import json,hashlib,torch,numpy as np,os')
evaluation=evaluation.replace("OUT=LAB/'outputs/dynamic-human-enerf-representation-048'", "OUT=Path(os.environ['ENERF_RUN_DIR'])")
evaluation=evaluation.replace('differs in representation and training pipeline, not a one-factor ablation.','paired050training-observation control, same model family and fixed target/background schedules.')
compile(evaluation,'evaluate_enerf_camera10_time.py','exec');(S/'evaluate_enerf_camera10_time.py').write_text(evaluation)
snap=OUT/'source-snapshot';snap.mkdir(exist_ok=True)
for n in ['prepare_enerf_camera10_time_mask.py','prepare_enerf_camera10_time_training.py','train_enerf_camera10_time.py']:
 shutil.copy2(S/n,snap/n)
print('050paired inputs verified; exact common initialization and isolated sampling prepared')
