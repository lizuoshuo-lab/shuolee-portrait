"""Training-only inverse expected camera-z supervision through native Mip kernels.

This differentiates alpha-weighted Gaussian-center depth, not native geometric
ray/ellipsoid intersection depth. The auxiliary pass uses the RGB pass's same
culling and Mip filtering. It does not alter RGB densification statistics.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from native_mip_filter import apply_filter


def expected_depth(camera, pc, culling=None):
    from diff_gaussian_rasterization_ms import GaussianRasterizationSettings, GaussianRasterizer
    means=pc.get_xyz
    z=(means@camera.world_view_transform[:3,:3]+camera.world_view_transform[3,:3])[:,2]
    # Precomputed channels allow z > 1 and retain dz/dmean in autograd.
    channels=torch.stack((z,torch.ones_like(z),torch.zeros_like(z)),dim=-1)
    scales,opacity=apply_filter(pc,camera,pc.get_scaling,pc.get_rotation,pc.get_opacity)
    settings=GaussianRasterizationSettings(image_height=camera.image_height,image_width=camera.image_width,
        tanfovx=math.tan(camera.FoVx/2),tanfovy=math.tan(camera.FoVy/2),
        bg=torch.zeros(3,device=means.device,dtype=means.dtype),scale_modifier=1.,
        viewmatrix=camera.world_view_transform,projmatrix=camera.full_proj_transform,
        sh_degree=0,campos=camera.camera_center,prefiltered=False,debug=False)
    if culling is None:culling=torch.zeros(len(means),device=means.device,dtype=torch.bool)
    rendered,_,_=GaussianRasterizer(settings)(means3D=means,
        means2D=torch.zeros_like(means,requires_grad=True),colors_precomp=channels,
        scales=scales,rotations=pc.get_rotation,opacities=opacity,culling=culling,flag_max_count=False)
    weighted_z,alpha=rendered[0],rendered[1]
    return weighted_z/alpha.clamp_min(1e-8),alpha


def inverse_depth_loss(z,alpha,inverse,confidence,valid,normalizer,minimum_alpha=.95):
    # The mask is not allowed to learn to suppress its own supervision.
    weight=(confidence*valid.squeeze(0)*(alpha.detach()>minimum_alpha)).detach()
    prediction=z.clamp_min(.2).reciprocal()
    residual=(prediction-inverse).abs()/normalizer.clamp_min(1e-5)
    value=(residual*weight).sum()/weight.sum().clamp_min(1)
    return value,weight


class DepthGuidance:
    def __init__(self,prior_root,cameras,output_root,coefficient):
        self.prior_root=Path(prior_root);self.output_root=Path(output_root)
        self.names={c.image_name for c in cameras}
        self.coefficient=float(coefficient);self.cache={};self.usage={};self.records=[]
        self.calls=0;self.active_calls=0;self.start=500;self.ramp_steps=1000
        meta=json.loads((self.prior_root/'manifest.json').read_text())
        assert meta['status']=='complete' and meta['appearance_holdout_excluded']
        assert {r['frame'] for r in meta['records']}==self.names
        assert len(self.names)==51 and all(int(n)%8!=0 for n in self.names)

    def prior(self,camera):
        assert camera.image_name in self.names,'Non-training camera requested depth guidance'
        key=(camera.image_name,camera.image_height,camera.image_width)
        if key not in self.cache:
            with np.load(self.prior_root/(camera.image_name+'.npz')) as n:
                inverse=torch.tensor(n['inverse_depth'].astype('float32'),device='cuda')[None,None]
                confidence=torch.tensor(n['weight'].astype('float32'),device='cuda')[None,None]
            assert torch.isfinite(inverse).all() and torch.isfinite(confidence).all()
            size=(camera.image_height,camera.image_width)
            if inverse.shape[-2:]!=size:
                weighted=F.interpolate(inverse*confidence,size=size,mode='area')
                confidence=F.interpolate(confidence,size=size,mode='area')
                inverse=weighted/confidence.clamp_min(1e-8)
            inverse=inverse[0,0];confidence=confidence[0,0]
            confidence=confidence*((inverse>0)&(confidence>.25))
            normalizer=inverse[confidence>0].median() if (confidence>0).any() else inverse.new_tensor(1.)
            self.cache[key]=(inverse,confidence,normalizer)
        return self.cache[key]

    def __call__(self,camera,pc,iteration,valid,culling):
        self.calls+=1
        if self.coefficient==0 or iteration<=self.start:return pc.get_xyz.new_zeros(())
        inverse,confidence,normalizer=self.prior(camera)
        z,alpha=expected_depth(camera,pc,culling)
        loss,weight=inverse_depth_loss(z,alpha,inverse,confidence,valid,normalizer)
        ramp=min(1.,(iteration-self.start)/self.ramp_steps)
        weighted=self.coefficient*ramp*loss
        self.active_calls+=1;self.usage[camera.image_name]=self.usage.get(camera.image_name,0)+1
        if iteration%100==0:
            record=dict(iteration=iteration,frame=camera.image_name,coefficient=self.coefficient*ramp,
                raw_loss=float(loss.detach()),weighted_loss=float(weighted.detach()),
                confidence_fraction=float((confidence>0).float().mean()),
                supervised_fraction=float((weight>0).float().mean()),alpha_mean=float(alpha.detach().mean()))
            assert math.isfinite(record['raw_loss'])
            self.records.append(record)
            with (self.output_root/'depth-training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        return weighted

    def finish(self):
        data=dict(coefficient=self.coefficient,start_iteration=self.start,ramp_steps=self.ramp_steps,
            calls=self.calls,active_calls=self.active_calls,training_frames_used=sorted(self.usage),usage=self.usage,
            loss='confidence-weighted L1 of inverse expected camera z, normalized by accepted prior median',
            minimum_alpha=.95,depth_definition='alpha-normalized expected Gaussian-center z; not surface intersection',
            motion_excluded=True,culling='same view-dependent native culling as RGB',
            densification_statistics='unchanged RGB pass only; depth changes optimized geometry and topology indirectly',
            prior='Depth Anything V2 Small aligned to SfM; flip, boundary and training multi-view checks; not GT',
            last_records=self.records[-5:])
        (self.output_root/'depth-guidance-usage.json').write_text(json.dumps(data,indent=2))
