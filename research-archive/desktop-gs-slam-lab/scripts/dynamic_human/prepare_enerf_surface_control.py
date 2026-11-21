from pathlib import Path
import json,shutil,hashlib
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
LAB=Path(__file__).resolve().parents[2];S=LAB/'scripts/dynamic_human';BASE=LAB/'outputs/dynamic-human-enerf-camera10-time-050/same_frame'
OUT=LAB/'outputs/dynamic-human-enerf-surface-gs-055';OUT.mkdir(exist_ok=True)
assert not (OUT/'protocol.json').exists()
save=lambda p,x:p.write_text(json.dumps(x,indent=2)+'\n')
init=np.load(BASE/'train-visual-hull-init.npz');pts=init['points'];_,idx=cKDTree(pts).query(pts,k=16,workers=4)
local=pts[idx];centered=local-local.mean(1,keepdims=True);cov=np.einsum('nki,nkj->nij',centered,centered)/16
values,vectors=np.linalg.eigh(cov);R=vectors[:,:,[1,2,0]].copy();R[np.linalg.det(R)<0,:,0]*=-1
qxyzw=Rotation.from_matrix(R).as_quat();q=np.c_[qxyzw[:,3],qxyzw[:,:3]].astype(np.float32)
assert np.max(abs(R.transpose(0,2,1)@R-np.eye(3)))<1e-5 and np.all(np.linalg.det(R)>.999)
np.savez_compressed(OUT/'surface-orientation.npz',quaternions_wxyz=q,normals=vectors[:,:,0],eigenvalues=values)
save(OUT/'protocol.json',dict(status='fixed_before_training',
 method='gsplat1.5.3 two-dimensional Gaussian surfels, paired surface-only versus surface regularization.',
 reference='https://surfsplatting.github.io/',
 initialization='Exact050training-only hull shell and colors; local16-neighbor PCA tangent planes, wxyz quaternions, no author surfaces or heldout observations.',
 fixed='Native750observations15trainingcameras, originalK/R/T/masks, same6000camera/backgrounddraws and optimizer/density settings across2Dbranches.',
 change='Regularizedbranch adds depth distortion and normal consistency. Surface-only computes same rendering outputs but loss weights remainzero.',
 regularization={'distortion_weight':.01,'distortion_start_update':600,'normal_weight':.05,'normal_start_update':1400,
  'support':'5x5eroded trainingforeground AND originalvalid; excludes silhouette derivatives',
  'implementation':'gsplat L1distortion variant, not original paper L2; normal formula from example, opacity-weighted depthnormals.'},
 iterations=6000,scope='Foreground single-frame method comparison; no temporal continuity or full-scene acceptance claimed.',goal_complete=False))
for branch in ['surface_only','surface_regularized']:
 root=OUT/branch;root.mkdir(exist_ok=True)
 for n in ['geometry-audit.json','initialization.json','train-visual-hull-init.npz','input-audit.json']:shutil.copy2(BASE/n,root/n)
 for f in BASE.glob('train-camera-*.npz'):shutil.copy2(f,root/f.name)
 shutil.copy2(OUT/'surface-orientation.npz',root/'surface-orientation.npz')
t=(S/'train_enerf_camera10_time.py').read_text()
t=t.replace('from gsplat import rasterization','from gsplat.rendering import rasterization_2dgs')
t=t.replace("ap.add_argument('--run',default='first-frame');a=ap.parse_args()", "ap.add_argument('--run',default='first-frame');ap.add_argument('--surface-reg',action='store_true');a=ap.parse_args()")
t=t.replace("dilated=tensor(cv2.dilate(m,np.ones((7,7),np.uint8)))", "dilated=tensor(cv2.dilate(m,np.ones((7,7),np.uint8))),surface_valid=tensor(cv2.erode(m,np.ones((5,5),np.uint8))*d['valid'])")
t=t.replace("tensor(np.tile([1,0,0,0],(N,1)))", "tensor(np.load(out/'surface-orientation.npz')['quaternions_wxyz'])")
t=t.replace('refine_every=100,verbose=True)', "refine_every=100,verbose=True,key_for_gradient='gradient_2dgs')")
t=t.replace("method='gsplat1.3DefaultStrategy paired camera10 observation control with fixed target/background schedules'", "method='gsplat1.5.3 2DGaussian surface control',surface_regularization=a.surface_reg")
old="""        return rasterization(params['means'],params['quats'],params['scales'].exp(),params['opacities'].sigmoid(),torch.cat([params['sh0'],params['shN']],1),o['w2c'][None],K[None],w,h,sh_degree=degree,packed=False,near_plane=.1,far_plane=20,eps2d=.3)"""
new="""        r,alpha,normals,depthnormals,distortion,median,info=rasterization_2dgs(params['means'],params['quats'],params['scales'].exp(),params['opacities'].sigmoid(),torch.cat([params['sh0'],params['shN']],1),o['w2c'][None],K[None],w,h,sh_degree=degree,packed=False,near_plane=.1,far_plane=20,eps2d=.3,render_mode='RGB+ED',distloss=True)
        if depthnormals.ndim==3:depthnormals=depthnormals[None]
        info['surface_normals']=normals;info['depth_normals']=depthnormals;info['distortion']=distortion
        return r[...,:3],alpha,info"""
assert old in t;t=t.replace(old,new)
needle="        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')"
add="""        if a.surface_reg:
            support=o['surface_valid'][y0:y1,x0:x1];denom=support.sum().clamp_min(1)
            normal_error=1-(info['surface_normals'][0]*(info['depth_normals'][0]*alpha[0].detach())).sum(-1)
            nl=(normal_error*support).sum()/denom
            dl=(info['distortion'][0,...,0]*support).sum()/denom
            if i>=600:loss=loss+.01*dl
            if i>=1400:loss=loss+.05*nl
            if i==1400:
                reggrad=torch.autograd.grad(.05*nl+.01*dl,params['means'],retain_graph=True)[0]
                assert torch.isfinite(reggrad).all() and reggrad.abs().sum()>0
                save(run/'surface-regularizer-gradient-check.json',dict(update_index=i,normal_weight=.05,distortion_weight=.01,means_gradient_l1=float(reggrad.abs().sum()),status='surface_loss_has_finite_nonzero_geometry_gradient'))
        else:nl=loss.new_zeros(());dl=loss.new_zeros(())
"""+needle
assert needle in t;t=t.replace(needle,add)
t=t.replace('row=dict(step=i+1,loss=float(loss),points=', 'row=dict(step=i+1,loss=float(loss),normal_term=float(nl),distortion_term=float(dl),points=')
compile(t,'train_enerf_surface_control.py','exec');(S/'train_enerf_surface_control.py').write_text(t)
e=(S/'evaluate_enerf_camera10_time.py').read_text()
e=e.replace('from gsplat import rasterization',"from gsplat.rendering import rasterization_2dgs\ndef rasterization(*args,**kwargs):\n r,a,n,nd,d,m,info=rasterization_2dgs(*args,**kwargs,render_mode='RGB+ED')\n return r[...,:3],a,info")
e=e.replace("method='Direct anisotropic3dGS, SH2,6000updates'", "method='2DGaussian surface control, SH2,6000updates'")
e=e.replace('Direct anisotropic3DGS','2DGaussian surface control').replace('direct3dgs','2dgs').replace('Direct3DGS6000','2DGS6000')
compile(e,'evaluate_enerf_surface_control.py','exec');(S/'evaluate_enerf_surface_control.py').write_text(e)
shell='''#!/bin/bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
python /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human/train_enerf_surface_control.py --out /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/dynamic-human-enerf-surface-gs-055/surface_only --steps 6000
export ENERF_RUN_DIR=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/dynamic-human-enerf-surface-gs-055/surface_only
python /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human/evaluate_enerf_surface_control.py
python /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human/train_enerf_surface_control.py --out /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/dynamic-human-enerf-surface-gs-055/surface_regularized --steps 6000 --surface-reg
export ENERF_RUN_DIR=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/dynamic-human-enerf-surface-gs-055/surface_regularized
python /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human/evaluate_enerf_surface_control.py
'''
(S/'run_enerf_surface_control.sh').write_text(shell,newline='\n')
snap=OUT/'source-snapshot';snap.mkdir(exist_ok=True)
for n in ['prepare_enerf_surface_control.py','train_enerf_surface_control.py','evaluate_enerf_surface_control.py','run_enerf_surface_control.sh','surface_gs_env.sh','smoke_enerf_surface_gs.py']:shutil.copy2(S/n,snap/n)
print('055paired2Dsurface inputs/PCAorientation/trainer/evaluator prepared')
