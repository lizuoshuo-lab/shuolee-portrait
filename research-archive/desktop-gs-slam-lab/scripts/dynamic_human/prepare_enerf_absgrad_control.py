from pathlib import Path
import json,shutil,hashlib
LAB=Path(__file__).resolve().parents[2];S=LAB/'scripts/dynamic_human'
OUT=LAB/'outputs/dynamic-human-enerf-absgrad-052';BASE=LAB/'outputs/dynamic-human-enerf-camera10-time-050/same_frame'
OUT.mkdir(exist_ok=True);snap=OUT/'source-snapshot';snap.mkdir(exist_ok=True)
assert not (OUT/'first-frame').exists()
for name in ['geometry-audit.json','initialization.json','train-visual-hull-init.npz','input-audit.json']:
 shutil.copy2(BASE/name,OUT/name)
for p in BASE.glob('train-camera-*.npz'):shutil.copy2(p,OUT/p.name)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(p)==sha(BASE/p.name) for p in OUT.glob('*.npz'))
protocol=dict(status='fixed_before_training',baseline=str(BASE),candidate='gsplat1.3DefaultStrategy AbsGS absolute screen-space gradient densification',
    references=['https://arxiv.org/abs/2404.10484','Installed gsplat.strategy.DefaultStrategy documentation'],
    changes={'strategy_absgrad':True,'rasterization_absgrad':True,'grow_grad2d':.0008},
    baseline_changes={'absgrad':False,'grow_grad2d':.0002},
    fixed='Same initial tensors, seed73,6000camera/background draws, optimizer/loss, native images, masks, originalK/R/T,6000steps, refinement400-4000/100 and reset2000.',
    evaluation='Original00/06/12photos750heldout;01/08/17trainingdiagnostic; same17predeclared050detailROIs.',
    scope='Local gsplat implementation of the documented AbsGS densification option, not complete author benchmark reproduction.',
    acceptance='No adoption fromPSNRalone; hands, movinglimbs, clothing, heldobjects and occlusion must be viewed.',goal_complete=False)
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
t=(S/'train_enerf_camera10_time.py').read_text()
t=t.replace('refine_every=100,verbose=True)','refine_every=100,verbose=True,absgrad=True,grow_grad2d=.0008)')
t=t.replace('near_plane=.1,far_plane=20,eps2d=.3)','near_plane=.1,far_plane=20,eps2d=.3,absgrad=True)')
t=t.replace("method='gsplat1.3DefaultStrategy paired camera10 observation control with fixed target/background schedules'", "method='gsplat1.3DefaultStrategy AbsGS absgrad=true grow_grad2d=.0008 matched6000update control'")
# Confirm CUDA kernel supplies absolute per-pixel gradient accumulation.
t=t.replace("strategy.step_pre_backward(params,optim,state,i,info);loss.backward()", "strategy.step_pre_backward(params,optim,state,i,info);loss.backward()\n        if i==0:\n            assert hasattr(info['means2d'],'absgrad') and torch.isfinite(info['means2d'].absgrad).all()\n            assert info['means2d'].absgrad.sum()>0\n            save(run/'absolute-gradient-check.json',dict(status='CUDA_absolute_gradient_buffer_present_finite_nonzero',absolute_sum=float(info['means2d'].absgrad.sum()),signed_absolute_sum=float(info['means2d'].grad.abs().sum())))")
compile(t,'train_enerf_absgrad.py','exec');(S/'train_enerf_absgrad.py').write_text(t)
shell='''#!/bin/bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
python /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human/train_enerf_absgrad.py --out /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/dynamic-human-enerf-absgrad-052 --steps 6000
export ENERF_RUN_DIR=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/dynamic-human-enerf-absgrad-052
python /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human/evaluate_enerf_camera10_time.py
'''
(S/'run_enerf_absgrad.sh').write_text(shell,newline='\n')
for n in ['train_enerf_absgrad.py','run_enerf_absgrad.sh','prepare_enerf_absgrad_control.py','evaluate_enerf_camera10_time.py']:shutil.copy2(S/n,snap/n)
print('052same inputs prepared; absolute-gradient strategy fixed before training')
