"""Mip-Splatting filtering for the isolated native Mini-Splatting2 experiment.

3D filter is a detached sampling estimate from training cameras. 2D determinant
compensation is differentiated in PyTorch; both native rasterizers use a matched
0.1-pixel-squared covariance floor in forward AND backward. Raw model scales and
opacity remain the parameters used by native cloning/pruning/simplification.
Reference: autonomousvision/mip-splatting @ dda02ab5ecf45d6edb8c540d9bb65c7e451345a9.
"""
import json
import math
from pathlib import Path

import torch
from plyfile import PlyData

KERNEL_SIZE = 0.1


def rotation_matrix(q):
    q = torch.nn.functional.normalize(q, dim=-1)
    r, x, y, z = q.unbind(-1)
    return torch.stack([
        1-2*(y*y+z*z), 2*(x*y-r*z), 2*(x*z+r*y),
        2*(x*y+r*z), 1-2*(x*x+z*z), 2*(y*z-r*x),
        2*(x*z-r*y), 2*(y*z+r*x), 1-2*(x*x+y*y)
    ], dim=-1).reshape(-1,3,3)


@torch.no_grad()
def refresh_filter(pc, force=False):
    key = (id(pc._xyz), pc.get_xyz.shape[0], getattr(pc, '_mip_iteration', 0)//100)
    if not force and getattr(pc, '_mip_cache_key', None) == key:
        return pc._mip_filter
    if not getattr(pc, '_mip_cameras', None):
        raise RuntimeError('3D filtering requires the TRAINING camera list')
    xyz = pc.get_xyz
    distance = torch.full_like(xyz[:,0], float('inf'))
    valid_any = torch.zeros(len(xyz), device=xyz.device, dtype=torch.bool)
    focal = 0.0
    for cam in pc._mip_cameras:
        v = cam.world_view_transform
        p = xyz @ v[:3,:3] + v[3,:3]
        z = p[:,2].clamp_min(.001)
        fx = cam.image_width/(2*math.tan(cam.FoVx/2))
        fy = cam.image_height/(2*math.tan(cam.FoVy/2))
        x = p[:,0]/z*fx + cam.image_width/2
        y = p[:,1]/z*fy + cam.image_height/2
        valid = (p[:,2]>.2)&(x>=-.15*cam.image_width)&(x<=1.15*cam.image_width)&(y>=-.15*cam.image_height)&(y<=1.15*cam.image_height)
        distance = torch.where(valid, torch.minimum(distance,z),distance)
        valid_any |= valid
        focal = max(focal,fx)
    if not valid_any.any():
        raise RuntimeError('No Gaussian is visible to the training cameras')
    distance[~valid_any] = distance[valid_any].max()
    pc._mip_filter = (distance/focal*math.sqrt(.2))[:,None]
    pc._mip_cache_key = key
    pc._mip_refreshes = getattr(pc, '_mip_refreshes', 0)+1
    return pc._mip_filter


def filtered_3d(scales, opacity, sigma):
    variance = scales.square()
    filtered_variance = variance + sigma.square()
    # Product of ratios avoids multiplying very small determinants.
    compensation = (variance/filtered_variance).prod(-1,keepdim=True).clamp_min(0).sqrt()
    return filtered_variance.sqrt(), opacity*compensation


def projected_covariance(means, scales, quats, camera, scaling_modifier=1.0):
    view = camera.world_view_transform.to(dtype=means.dtype,device=means.device)
    p = means @ view[:3,:3]+view[3,:3]
    # Invisible points never contribute; keep their intermediate arithmetic finite.
    z = p[:,2].clamp_min(.001)
    tx,ty = math.tan(camera.FoVx/2),math.tan(camera.FoVy/2)
    fx,fy = camera.image_width/(2*tx),camera.image_height/(2*ty)
    ux=(p[:,0]/z).clamp(-1.3*tx,1.3*tx)
    uy=(p[:,1]/z).clamp(-1.3*ty,1.3*ty)
    zero=torch.zeros_like(z)
    J=torch.stack([fx/z,zero,-fx*ux/z,zero,fy/z,-fy*uy/z],-1).reshape(-1,2,3)
    R=rotation_matrix(quats)
    # J * R_world_to_camera * R_gaussian * S
    A=(J @ view[:3,:3].T @ R)*(scales*scaling_modifier)[:,None,:]
    return A @ A.transpose(-1,-2)


def compensate_2d(means,scales,quats,opacity,camera,scaling_modifier=1.0):
    cov=projected_covariance(means,scales,quats,camera,scaling_modifier)
    a,b,c=cov[:,0,0],cov[:,0,1],cov[:,1,1]
    det0=a*c-b*b
    det1=(a+KERNEL_SIZE)*(c+KERNEL_SIZE)-b*b
    # Match the reference Mip-Splatting numerical convention.
    coef=(det0.clamp_min(0)/(det1.clamp_min(0)+1e-6)+1e-6).sqrt()
    coef=torch.where((det0>1e-6)&(det1>1e-6),coef,torch.zeros_like(coef))
    return opacity*coef[:,None]


def apply_filter(pc,camera,scales,rotations,opacity,scaling_modifier=1.0):
    sigma=refresh_filter(pc)
    scales,opacity=filtered_3d(scales,opacity,sigma)
    opacity=compensate_2d(pc.get_xyz,scales,rotations,opacity,camera,scaling_modifier)
    return scales,opacity


@torch.no_grad()
def save_baked(pc,model_path,iteration):
    sigma=refresh_filter(pc,force=True)
    scales,opacity=filtered_3d(pc.get_scaling,pc.get_opacity,sigma)
    raw=Path(model_path)/'point_cloud'/f'iteration_{iteration}'/'point_cloud.ply'
    p=PlyData.read(str(raw));v=p['vertex'].data
    logs=scales.log().cpu().numpy()
    logits=torch.logit(opacity.clamp(1e-9,1-1e-7)).cpu().numpy()[:,0]
    for i in range(3):v[f'scale_{i}']=logs[:,i]
    v['opacity']=logits
    baked=raw.with_name('point_cloud_mip3d_baked.ply');p.write(str(baked))
    torch.save(sigma.cpu(),raw.with_name('filter_3d.pt'))
    (Path(model_path)/'mip-filter.json').write_text(json.dumps(dict(
        schema=1,kernel_variance_2d=KERNEL_SIZE,gaussians=len(v),
        train_cameras=len(pc._mip_cameras),refreshes=pc._mip_refreshes,
        refresh_policy='every100steps and each parameter/topology replacement; final forced',
        raw_ply=str(raw),baked_ply=str(baked),
        bake_scope='3D filter only; view-dependent 2D compensation must still be applied',
        raw_native_densification_parameters_preserved=True
    ),indent=2))
