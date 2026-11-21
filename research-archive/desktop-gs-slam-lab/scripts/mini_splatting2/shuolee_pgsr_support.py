"""Project adaptation: static-pixel masks and compact PGSR execution evidence."""
from collections import OrderedDict
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

_cache=OrderedDict()
_began=time.time()


def valid_mask(camera,erode=0):
    key=(camera.image_name,camera.image_height,camera.image_width,erode)
    if key in _cache:
        value=_cache.pop(key);_cache[key]=value;return value
    path=Path(os.environ['SHUOLEE_MOTION_MASKS'])/(camera.image_name+'.png')
    raw=np.asarray(Image.open(path).convert('L'))
    mask=torch.from_numpy((raw==0).astype('float32'))[None,None].cuda()
    mask=F.interpolate(mask,size=(camera.image_height,camera.image_width),mode='nearest')
    if erode:
        bad=F.pad(1-mask,(erode,erode,erode,erode),value=1)
        mask=1-F.max_pool2d(bad,2*erode+1,stride=1)
    value=mask[0];_cache[key]=value
    while len(_cache)>4:_cache.popitem(last=False)
    return value


def multiview_valid(source,target,points_in_target,patch_size):
    source_valid=valid_mask(source,patch_size)[0].reshape(-1)>.999
    z=points_in_target[:,2].clamp_min(1e-5)
    x=points_in_target[:,0]/z*target.Fx+target.Cx
    y=points_in_target[:,1]/z*target.Fy+target.Cy
    grid=torch.stack([2*x/(target.image_width-1)-1,2*y/(target.image_height-1)-1],-1).reshape(1,-1,1,2)
    target_valid=F.grid_sample(valid_mask(target,patch_size)[None],grid,align_corners=True,padding_mode='zeros')[0,0,:,0]>.999
    return source_valid&target_valid


def patch_static_valid(target,grid):
    samples=F.grid_sample(valid_mask(target)[None],grid.reshape(1,-1,1,2),align_corners=True,padding_mode='zeros')
    return (samples.reshape(grid.shape[0],grid.shape[1])>.999).all(-1)


def audit(scene,iteration,image_loss,normal_loss,geo_loss,ncc_loss):
    if iteration!=1 and iteration%200!=0:return
    def value(v):return None if v is None else float(v.detach())
    row=dict(iteration=iteration,gaussians=len(scene.gaussians.get_xyz),rgb=value(image_loss),
        normal=value(normal_loss),multiview_geometry=value(geo_loss),multiview_ncc=value(ncc_loss),
        elapsed_seconds=time.time()-_began,allocated_mib=torch.cuda.memory_allocated()/2**20,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        neighbor_counts={c.image_name:len(c.nearest_id) for c in scene.getTrainCameras()} if iteration==1 else None)
    assert all(v is None or np.isfinite(v) for v in [row[k] for k in ['rgb','normal','multiview_geometry','multiview_ncc']])
    root=Path(scene.model_path)
    with (root/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    (root/'progress.json').write_text(json.dumps(dict(status='training',**row),indent=2))
    print('SHUOLEE '+json.dumps(row),flush=True)


def verify_scene(scene):
    train=scene.getTrainCameras();test=scene.getTestCameras()
    assert len(train)==51 and len(test)==8
    assert all(int(c.image_name)%8 for c in train)
    assert all(int(c.image_name)%8==0 for c in test)
    assert all(c.nearest_id for c in train),'No valid multi-view partner for a training camera'
    assert all(len(c.nearest_id)==len(c.nearest_names) for c in train)
    assert all((c.image_width,c.image_height)==(1514,2694) for c in train+test)
    assert all(torch.isfinite(c.world_view_transform).all() for c in train+test)
    torch.cuda.set_per_process_memory_fraction(.82)
