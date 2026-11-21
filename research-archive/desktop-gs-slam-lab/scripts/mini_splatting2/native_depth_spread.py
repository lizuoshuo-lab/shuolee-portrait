"""Experimental center-depth variance regularization; not a 2DGS reproduction.

The existing expected-depth and RGB passes stay unchanged. This auxiliary pass
has separate screen-space tensors, so it does not directly alter RGB gradient
statistics for densification. Training/evaluation acceptance remains separate.
"""
import json
import math
import os
from pathlib import Path

import torch
from native_depth_guidance import DepthGuidance
from native_mip_filter import apply_filter


def center_depth_variance(camera,pc,normalizer,culling=None):
    from diff_gaussian_rasterization_ms import GaussianRasterizationSettings,GaussianRasterizer
    xyz=pc.get_xyz
    z=(xyz@camera.world_view_transform[:3,:3]+camera.world_view_transform[3,:3])[:,2]
    q=z*normalizer.detach()
    channels=torch.stack([q,q.square(),torch.ones_like(q)],-1)
    scales,opacity=apply_filter(pc,camera,pc.get_scaling,pc.get_rotation,pc.get_opacity)
    settings=GaussianRasterizationSettings(image_height=camera.image_height,image_width=camera.image_width,
        tanfovx=math.tan(camera.FoVx/2),tanfovy=math.tan(camera.FoVy/2),
        bg=torch.zeros(3,device=xyz.device,dtype=xyz.dtype),scale_modifier=1.,
        viewmatrix=camera.world_view_transform,projmatrix=camera.full_proj_transform,
        sh_degree=0,campos=camera.camera_center,prefiltered=False,debug=False)
    if culling is None:culling=torch.zeros(len(xyz),device=xyz.device,dtype=torch.bool)
    result,_,_=GaussianRasterizer(settings)(means3D=xyz,means2D=torch.zeros_like(xyz,requires_grad=True),
        colors_precomp=channels,scales=scales,rotations=pc.get_rotation,opacities=opacity,culling=culling,flag_max_count=False)
    alpha=result[2];mean=result[0]/alpha.clamp_min(1e-8)
    raw_variance=result[1]/alpha.clamp_min(1e-8)-mean.square()
    return raw_variance.clamp_min(0),alpha


def spread_loss(variance,alpha,inverse,confidence,valid,normalizer,minimum_alpha=.95):
    weight=(confidence*valid.squeeze(0)*(alpha.detach()>minimum_alpha)).detach()
    # variance is in normalized z units. Convert to variance / prior_depth^2.
    relative=variance*(inverse/normalizer.detach().clamp_min(1e-5)).square()
    value=(torch.log1p(relative)*weight).sum()/weight.sum().clamp_min(1)
    return value,weight


class DepthSpread(DepthGuidance):
    def __init__(self,prior_root,cameras,output_root,coefficient):
        super().__init__(prior_root,cameras,output_root,coefficient)
        self.start=int(os.environ.get('MINIGS_SPREAD_START','4000'))
        self.ramp_steps=int(os.environ.get('MINIGS_SPREAD_RAMP','1000'))

    def __call__(self,camera,pc,iteration,valid,culling):
        self.calls+=1
        if self.coefficient==0 or iteration<=self.start:return pc.get_xyz.new_zeros(())
        inverse,confidence,normalizer=self.prior(camera)
        variance,alpha=center_depth_variance(camera,pc,normalizer,culling)
        loss,weight=spread_loss(variance,alpha,inverse,confidence,valid,normalizer)
        ramp=min(1.,(iteration-self.start)/self.ramp_steps)
        weighted=self.coefficient*ramp*loss
        self.active_calls+=1;self.usage[camera.image_name]=self.usage.get(camera.image_name,0)+1
        if iteration%100==0:
            record=dict(iteration=iteration,frame=camera.image_name,coefficient=self.coefficient*ramp,
                raw_loss=float(loss.detach()),weighted_loss=float(weighted.detach()),
                supervised_fraction=float((weight>0).float().mean()),alpha_mean=float(alpha.detach().mean()))
            assert math.isfinite(record['raw_loss']);self.records.append(record)
            with (self.output_root/'spread-training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        return weighted

    def finish(self):
        data=dict(coefficient=self.coefficient,start_iteration=self.start,ramp_steps=self.ramp_steps,
            calls=self.calls,active_calls=self.active_calls,training_frames_used=sorted(self.usage),usage=self.usage,
            loss='confidence weighted mean log(1 + center-z-variance / prior-depth^2)',
            mean_depth_loss='unchanged existing expected-depth pass, weight0.003',
            gaussian_internal_variance_included=False,not_2dgs_intersection_depth=True,
            minimum_alpha=.95,motion_excluded=True,heldout_used=False,
            densification_statistics='RGB screen-space gradient only; geometry and topology may change indirectly',
            prior='training-only aligned monocular prior, not independent GT',last_records=self.records[-5:])
        (self.output_root/'spread-guidance-usage.json').write_text(json.dumps(data,indent=2))
