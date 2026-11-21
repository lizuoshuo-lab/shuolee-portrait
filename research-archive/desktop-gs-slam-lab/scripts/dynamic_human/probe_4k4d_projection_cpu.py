"""Check coordinate conversion and mask-crop invariance on actual point samples."""
from pathlib import Path
import json,torch,numpy as np
from pytorch3d.renderer import PerspectiveCameras
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.fcds_utils import get_pytorch3d_camera_params
from audit_enerf_input import read_cameras
torch.set_num_threads(4)
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
OUT=LAB/'outputs/dynamic-human-4k4d-official-035'
source=Path('/mnt/d/GS-SLAM-Data/enerf_outdoor/extracted/actor5_6')
cameras=read_cameras(source/'optimized')
checkpoint=torch.load(OUT/'model/latest.pt',map_location='cpu',weights_only=False)
points=checkpoint['model']['sampler.pcds.0'][::64].float()[None]
del checkpoint
crops={str(r['target']).zfill(2):r for r in json.loads((OUT/'official-data-probe.json').read_text())['records']}
records=[]
for name,cam in cameras.items():
    K=torch.tensor(cam['K'],dtype=torch.float32)[None]
    w2c=torch.tensor(cam['w2c'],dtype=torch.float32)[None]
    R=w2c[:,:3,:3];T=w2c[:,:3,3:4]
    cv=points@R.mT+T.mT
    uv=(cv@K.mT);uv=uv[...,:2]/uv[...,2:]
    assert bool((cv[...,2]>0).all())
    cases=[('native',1080,1920,K)]
    if name in crops:
        row=crops[name];cases.append(('crop',row['H'],row['W'],torch.tensor(row['K'])[None]))
    for label,H,W,k in cases:
        batch=dotdict(K=k,R=R,T=T,meta=dotdict(H=torch.tensor([H]),W=torch.tensor([W])))
        _,_,nk,nr,nt,_=get_pytorch3d_camera_params(batch)
        camera=PerspectiveCameras(K=nk,R=nr,T=nt,device='cpu')
        screen=camera.transform_points_screen(points,image_size=((H,W),))[...,:2]
        offset=K[0,:2,2]-k[0,:2,2]
        err=torch.linalg.vector_norm(screen+offset-uv,dim=-1)
        record=dict(camera=name,mode=label,points=points.shape[1],max_error_px=float(err.max()),median_error_px=float(err.median()))
        records.append(record)
report=dict(status='measured',checks=records,max_error_px=max(x['max_error_px'] for x in records),
    scope='Official CPU PyTorch3D screen projection vs direct released-camera projection; native and actual mask crops. Does not validate physical calibration, synchronization, or raster pixel-center sampling.')
report['status']='coordinate_projection_and_crop_invariance_pass' if report['max_error_px']<.02 else 'projection_discrepancy_requires_review'
(OUT/'projection-cpu-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
