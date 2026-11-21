"""Residual-camera training consumer, with frozen per-frame K/E/D and native pixels.

This component supplies no person-data admission or adaptive density policy.
"""
import copy
import hashlib
import json
import numpy as np
import torch
from native_training122 import NativeTrainingView,foreground_objective
from native_training122_v2 import FreshNativeGaussians
from native_portrait_renderer122_v3 import render_cameras
from native_residual_camera122 import validate_camera,base_camera,sampled_domain_check
from native_scene_package122_v4 import write_scene

def camera_digest(camera):
    return hashlib.sha256(json.dumps(camera,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def assert_frozen_view(view):
    if not hasattr(view,'_recorded_camera_digest') or camera_digest(view.camera)!=view._recorded_camera_digest:
        raise ValueError('Recorded K/E/D changed after view creation')

def create_view(camera,rgb,alpha,rgb_valid,observation_valid,*,device):
    validate_camera(camera);sampled_domain_check(camera)
    v=NativeTrainingView.create(base_camera(camera),rgb,alpha,rgb_valid,observation_valid,device=device)
    v.camera=copy.deepcopy(camera)
    v._recorded_camera_digest=camera_digest(v.camera)
    return v

def render_view(model,view,contract):
    assert_frozen_view(view)
    validate_camera(view.camera)
    for value,key in [(view.K,'K'),(view.w2c,'w2c')]:
        recorded=torch.as_tensor(view.camera[key],dtype=value.dtype,device=value.device)[None]
        if value.requires_grad or not torch.equal(value,recorded):
            raise ValueError('Training tensor differs from the frozen recorded camera')
    rgb,alpha,meta=render_cameras(model.parameters_linear(),[view.camera],contract=contract,sh_degree=3)
    return rgb[0],alpha[0],meta

def accumulate_update(model,views,contract,optimizer):
    if not isinstance(model,FreshNativeGaussians): raise ValueError('Fresh free-world Gaussian model required')
    if not views or len({v.camera['id'] for v in views})!=len(views): raise ValueError('Distinct recorded views required')
    originals=copy.deepcopy([v.camera for v in views])
    optimizer.zero_grad(set_to_none=True);rows=[]
    for v in views:
        rgb,alpha,meta=render_view(model,v,contract)
        loss,terms=foreground_objective(rgb,alpha,v)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite native residual loss')
        (loss/len(views)).backward()
        rows.append(dict(frame_id=v.camera['id'],loss=float(loss.detach()),visible=int((meta['radii']>0).all(-1).sum())))
    for name,p in model.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all(): raise RuntimeError('Missing/nonfinite gradient '+name)
    optimizer.step()
    if originals!=[v.camera for v in views]: raise RuntimeError('Frozen K/E/D changed')
    return dict(per_view=rows,optimizer_steps=1,model_growth='fixed count',physical_training_admission=False)

def export_native_checkpoint(path,model,views,contract,*,scope):
    for view in views: assert_frozen_view(view)
    return write_scene(path,model.numpy_linear(),[v.camera for v in views],contract=contract,scope=scope)
