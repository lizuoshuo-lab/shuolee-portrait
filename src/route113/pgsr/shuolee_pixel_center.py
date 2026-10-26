"""Opt-in consistent pixel centers for the existing PGSR static adapter.

RGB CUDA uses integer sample centers with principal point (W-1)/2.
PGSR's compiled plane-depth output instead uses W/2. Recompute plane depth
from its differentiable normal/distance maps, keeping RGB exactly unchanged.
Only the existing scale=1 geometry path has been audited. No default training
or current result is changed by importing this module.
"""
import copy
import torch
import shuolee_static as legacy
from gaussian_renderer import render as native_render, render_normal
_native_cameras = legacy.cameras

def corrected_camera(cam):
    assert cam.ncc_scale == 1, 'This adapter is validated only at NCC scale 1'
    result=copy.copy(cam)
    result.Cx=(cam.image_width-1)*.5
    result.Cy=(cam.image_height-1)*.5
    result.shuolee_pixel_center_consistent=True
    return result

def render(cam, model, pipe, bg, scaling_modifier=1., override_color=None,
           app_model=None, return_plane=True, return_depth_normal=True):
    assert getattr(cam,'shuolee_pixel_center_consistent',False)
    assert abs(cam.Cx-(cam.image_width-1)*.5)<1e-7
    assert abs(cam.Cy-(cam.image_height-1)*.5)<1e-7
    pkg=native_render(cam,model,pipe,bg,scaling_modifier,override_color,app_model,
                      return_plane=return_plane,return_depth_normal=False)
    if not return_plane:
        return pkg
    normal=pkg['rendered_normal']
    rays=cam.get_rays().permute(2,0,1)
    denominator=-(normal*rays).sum(dim=0,keepdim=True)-1e-8
    pkg['plane_depth']=pkg['rendered_distance']/denominator
    if return_depth_normal:
        pkg['depth_normal']=render_normal(cam,pkg['plane_depth'].squeeze())*pkg['rendered_alpha'].detach()
    return pkg

def cameras(rows,height,alpha_composite=False):
    cams,records=_native_cameras(rows,height,alpha_composite)
    result=[corrected_camera(cam) for cam in cams]
    for cam,r in zip(result,records):
        r['RGB_K_corner_convention']=r['K']
        r['geometry_K_integer_pixel_centers']=cam.get_k().cpu().tolist()
        r['pixel_center_correction']='Geometry rays and plane depth aligned to unchanged CUDA RGB'
    return result,records
