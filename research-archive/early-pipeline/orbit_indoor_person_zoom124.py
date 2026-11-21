"""Read-only, same 36 frozen diagnostic cameras for both 1000-step PLYs."""
import copy
import fcntl
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
sys.path[:0]=['/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src',
             '/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src/avatar_core']
from shuolee_portrait import runtime as rt
from run_official_gaussians123 import ASSETS

ROOT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')
Z=ROOT/'indoor-person-zoom2x124';B=ROOT/'indoor-selected-sh0-pipeline124'

def main():
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert rt.read(Z/'gaussian-run/result.json')['state']=='bounded_control_completed'
    dest=Z/'gaussian-run/step-001000/orbit-comparison';dest.mkdir(exist_ok=False)
    rt.configure(ASSETS)
    import torch
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from shuolee_portrait.training import camera,image
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    row=rt.read(B/'data/training-inputs.json')['records'][0]
    other=rt.read(Z/'data/training-inputs.json')['records'][0]
    assert row['frame']==other['frame']=='frame_000105'
    assert np.array_equal(row['w2c'],other['w2c'])
    G=np.eye(4);G[:3,:3]=np.diag([1.,-1.,-1.])@np.asarray(row['w2c'])[:3,:3]
    assert np.max(np.abs(G[:3,:3].T@G[:3,:3]-np.eye(3)))<1e-6
    original_cameras=B/'delivery/step-018000/novel/cameras.json'
    cameras=rt.read(original_cameras)['records'];assert len(cameras)==36
    models=[];hashes={};started=time.monotonic()
    for folder in (B,Z):
        path=folder/'gaussian-run/step-001000/world.ply'
        assert rt.sha(path)==rt.read(path.parent/'result.json')['world_ply_sha256']
        hashes[str(path)]=rt.sha(path)
        model=GaussianModel(0);model.load_ply(str(path));model.active_sh_degree=0;models.append(model)
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    bg=torch.zeros(3,device='cuda');records=[];pairs=[]
    with torch.no_grad():
        for i,original in enumerate(cameras):
            row=copy.deepcopy(original)
            row['w2c']=(np.asarray(original['w2c'])@G).tolist()
            cam,_=camera(row,i)
            pics=[]
            for name,model in zip(('baseline','zoom2x'),models):
                rgb=render(cam,model,pipe,bg)['render'];assert torch.isfinite(rgb).all()
                pic=image(rgb);path=dest/f'{i:02d}-{name}.png';pic.save(path);pics.append(pic)
            pair=Image.new('RGB',(pics[0].width*2,pics[0].height),'black')
            pair.paste(pics[0],(0,0));pair.paste(pics[1],(pics[0].width,0))
            pairs.append(pair);records.append(row)
        for start in (0,12,24):
            sheet=Image.new('RGB',(3072,1536),'black')
            for j,pic in enumerate(pairs[start:start+12]):sheet.paste(pic,((j%4)*768,(j//4)*512))
            sheet.save(dest/f'orbit-{start:02d}-{start+11:02d}.png')
        quadrants=Image.new('RGB',(1536,1024),'black')
        for j,index in enumerate((0,9,18,27)):quadrants.paste(pairs[index],((j%2)*768,(j//2)*512))
        quadrants.save(dest/'four-uniform-directions.png')
    assert all(rt.sha(path)==digest for path,digest in hashes.items())
    rt.save(dest/'result.json',dict(updated=rt.now(),records=records,source_hashes=hashes,
        frozen_orbit_cameras_sha256=rt.sha(original_cameras),views=36,ground_truth_available=False,
        same_reference_pose=True,same_camera_for_both=True,rotation_of_camera_only=True,
        model_parameters_edited=False,optimization_updates=0,seconds=time.monotonic()-started,
        panel_order=['baseline1000','zoom2x1000'],actual_visual_review=False,quality_accepted=False))
    print('ZOOM_READONLY_ORBIT_READY',len(records),flush=True)

if __name__=='__main__':main()
