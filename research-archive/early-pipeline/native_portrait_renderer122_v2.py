"""Versioned UT/ray-3D adapter with geometry-only UT bounds.

The projection kernel receives no opacity. The ray-response kernel receives the
original opacity. This removes the demonstrated blur-compensation false cull.
UT sigma bounds remain an approximation, not a proof of arbitrary-lens support.
No legacy module or live viewer is modified by importing this implementation.
"""
from dataclasses import dataclass
import math
from native_portrait_renderer122 import NativeCameraContract

REVISION='122-ray3d-geometry-bounds-v2'
BOUNDS='ut-3.33-sigma-with-eps2d-no-opacity-cull'
UT_SETTINGS=dict(alpha=.1,beta=2.,kappa=0.,in_image_margin_factor=.1,require_all_sigma_points_valid=True)

@dataclass(frozen=True)
class NativeCameraContractV2(NativeCameraContract):
    bounds: str = BOUNDS

    def validate(self):
        super().validate()
        if self.bounds != BOUNDS:raise ValueError('Unsupported v2 bounds; no fallback')

    def manifest(self):
        value=super().manifest()
        value.update(schema='shuolee-native-renderer-v2',adapterRevision=REVISION,
                     bounds=self.bounds,opacityCompensationApplied=False)
        return value


def render_native(means,quats,scales,opacities,colors,viewmats,Ks,width,height,*,contract,radial_coeffs=None,sh_degree=None):
    import torch
    from gsplat.cuda._wrapper import (fully_fused_projection_with_ut,isect_tiles,isect_offset_encode,
        rasterize_to_pixels_eval3d,spherical_harmonics,UnscentedTransformParameters,RollingShutterType)
    if not isinstance(contract,NativeCameraContractV2):raise ValueError('Explicit v2 contract required')
    contract.validate()
    if type(width)is not int or type(height)is not int or min(width,height)<=0:raise ValueError('Positive integer image size required')
    if means.ndim!=2 or means.shape[1]!=3 or len(means)==0:raise ValueError('Nonempty unbatched world XYZ required')
    n=len(means)
    if quats.shape!=(n,4) or scales.shape!=(n,3) or opacities.shape!=(n,):raise ValueError('Gaussian attribute shape mismatch')
    if viewmats.ndim!=3 or viewmats.shape[1:]!=(4,4) or len(viewmats)==0 or Ks.shape!=(len(viewmats),3,3):raise ValueError('Explicit complete camera batch required')
    tensors=[means,quats,scales,opacities,colors,viewmats,Ks]+([] if radial_coeffs is None else [radial_coeffs])
    if any(x.device!=means.device or x.dtype!=torch.float32 or not x.is_cuda for x in tensors):raise ValueError('All inputs must share one CUDA device and float32 dtype')
    if any(not torch.isfinite(x).all() for x in tensors):raise ValueError('Nonfinite input')
    if Ks.requires_grad or viewmats.requires_grad:raise ValueError('This adapter freezes K and extrinsics; no camera optimization')
    if torch.any(scales<=0) or torch.any((opacities<0)|(opacities>1)) or torch.any(quats.norm(dim=-1)<=1e-12):raise ValueError('Invalid scale, opacity or quaternion')
    rot=viewmats[:,:3,:3]
    if (not torch.allclose(viewmats[:,3],viewmats.new_tensor([0,0,0,1]).expand_as(viewmats[:,3]),atol=1e-6,rtol=0)
            or not torch.allclose(rot@rot.transpose(-2,-1),torch.eye(3,device=rot.device).expand_as(rot),atol=1e-5,rtol=0)
            or torch.any((torch.linalg.det(rot)-1).abs()>1e-5)):raise ValueError('Rigid OpenCV w2c required')
    if (torch.any(Ks[:,0,0]<=0) or torch.any(Ks[:,1,1]<=0) or torch.any(Ks[:,0,1]!=0) or torch.any(Ks[:,1,0]!=0)
            or torch.any(Ks[:,2]!=Ks.new_tensor([0,0,1]))):raise ValueError('Full zero-skew K required')
    if sh_degree is None:
        if colors.shape!=(n,3):raise ValueError('Explicit RGB must have shape N,3')
    elif type(sh_degree)is not int or sh_degree not in range(4) or colors.shape!=(n,(sh_degree+1)**2,3):raise ValueError('Complete SH degree0..3 RGB array required')
    c=len(viewmats)
    if contract.projection=='pinhole':
        if radial_coeffs is not None:raise ValueError('Rectified pinhole accepts no distortion')
        camera_model='pinhole'
    else:
        if radial_coeffs is None or radial_coeffs.shape!=(c,4):raise ValueError('Exactly four calibrated KB coefficients per camera required')
        camera_model='fisheye'
    ut=UnscentedTransformParameters(**UT_SETTINGS)
    geometry=dict(camera_model=camera_model,ut_params=ut,radial_coeffs=radial_coeffs,rolling_shutter=RollingShutterType.GLOBAL)
    radii,means2d,depths,conics,_=fully_fused_projection_with_ut(means,quats,scales,None,viewmats,Ks,width,height,
        eps2d=.3,near_plane=.01,far_plane=1e10,radius_clip=0.,calc_compensations=False,**geometry)
    tile=16;tw=math.ceil(width/tile);th=math.ceil(height/tile)
    counts,ids,flat=isect_tiles(means2d,radii,depths,tile,tw,th,packed=False,sort=True,segmented=False)
    offsets=isect_offset_encode(ids,c,tw,th).reshape(c,th,tw)
    if sh_degree is None:rgb=colors[None].expand(c,-1,-1)
    else:
        centers=torch.inverse(viewmats)[:,:3,3]
        dirs=means[None]-centers[:,None]
        rgb=(spherical_harmonics(sh_degree,dirs,colors[None].expand(c,-1,-1,-1),masks=(radii>0).all(-1))+.5).clamp_min(0)
    # Intentionally no blur compensation: classic 3D response uses original opacity.
    alpha=opacities[None].expand(c,-1)
    output,coverage=rasterize_to_pixels_eval3d(means,quats,scales,rgb,alpha,viewmats,Ks,width,height,tile,offsets,flat,**geometry)
    return output,coverage,dict(radii=radii,means2d=means2d,depths=depths,conics=conics,opacities=alpha,
        tiles_per_gauss=counts,isect_ids=ids,flatten_ids=flat,isect_offsets=offsets,
        width=width,height=height,tile_size=tile,adapter_revision=REVISION,bounds=BOUNDS)
