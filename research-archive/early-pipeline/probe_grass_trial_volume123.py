"""One non-mutating feasibility check with the already registered surface settings.

Does not relax a threshold, invent a fit gate, or publish/train points.
"""
import fcntl
import time
import numpy as np
from PIL import Image
from trial_grass_surface123 import ROOT, OUT, ASSETS, read, save, sha, now, verify_runtime

lock=open('/tmp/shuolee-native122-gpu.lock','a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
verify_runtime()
from shuolee_portrait import runtime as rt
rt.configure(ASSETS,ROOT)
import torch, cv2
cfg=read(ROOT/'config.json');fit=dict(np.load(ROOT/'fit/fitted.npz'))
rows=read(ROOT/'data/training-inputs.json')['records']
scale=float(fit['world_scale']);R=fit['root_rotation'][0].astype(float);t=fit['world_translation'][0].astype(float)
local=(fit['vertices_world'][0]-t)@R/scale
spacing=cfg['surface_spacing_m'];padding=cfg['surface_padding_m']
axes=[np.arange(local[:,j].min()-padding,local[:,j].max()+padding,spacing) for j in range(3)]
shape=tuple(len(a) for a in axes)
assert np.prod(shape)<=5000000
grid=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3)
points=torch.tensor(grid@R.T*scale+t,device='cuda',dtype=torch.float32)
votes=torch.zeros(len(points),device='cuda',dtype=torch.int16)
tolerance=int(cfg['surface_mask_tolerance_px']);kernel=np.ones((2*tolerance+1,)*2,np.uint8)
start=time.monotonic()
with torch.no_grad():
    for i,row in enumerate(rows):
        assert rt.sha(row['mask'])==row['mask_sha256']
        mask=cv2.dilate((np.array(Image.open(row['mask']).convert('L'))>127).astype(np.uint8),kernel)
        im=torch.tensor(mask,device='cuda',dtype=torch.float32)[None,None]
        E=torch.tensor(row['w2c'],device='cuda',dtype=torch.float32)
        K=torch.tensor(row['K'],device='cuda',dtype=torch.float32)
        wh=torch.tensor(row['wh'],device='cuda',dtype=torch.float32)
        for begin in range(0,len(points),250000):
            chunk=points[begin:begin+250000];cam=chunk@E[:3,:3].T+E[:3,3];q=cam@K.T
            uv=q[:,:2]/q[:,2:].clamp_min(.001)-.5
            normalized=2*uv/(wh-1)-1
            inside=torch.nn.functional.grid_sample(im,normalized[None,:,None],align_corners=True,padding_mode='zeros')[0,0,:,0]>.5
            votes[begin:begin+len(chunk)]+=(inside&(cam[:,2]>.001)).to(torch.int16)
        if i%16==0:print(f'VOLUME_PROBE {i+1}/{len(rows)}',flush=True)
values=votes.cpu().numpy().reshape(shape)
occupied=values/len(rows)>cfg['surface_vote_fraction']
edge=any(np.any(np.take(occupied,[0,-1],axis=j)) for j in range(3))
result=dict(updated=now(),read_only_feasibility=True,coarse_fit_gate_passed=False,
    settings=dict(spacing_model_m=spacing,padding_model_m=padding,mask_tolerance_native_px=tolerance,vote_fraction=cfg['surface_vote_fraction']),
    grid_shape=shape,grid_voxels=int(values.size),max_views_in_common=int(values.max()),frames=len(rows),
    occupied_voxels=int(occupied.sum()),minimum_required=1000,reaches_boundary=edge,
    surface_feasible=bool(occupied.sum()>=1000 and not edge),seconds=time.monotonic()-start,
    fit_sha256=sha(ROOT/'fit/fitted.npz'),cameras_sha256=sha(ROOT/'data/training-inputs.json'),
    gaussian_steps=0,point_cloud_created=False,threshold_changed=False,quality_accepted=False,
    limitation='Finite grid in the registered body proposal bounds. No emptiness claim for all possible world locations.')
save(OUT/'surface-feasibility.json',result)
print(result,flush=True)
