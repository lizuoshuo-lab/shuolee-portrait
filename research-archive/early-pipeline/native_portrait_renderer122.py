"""Explicit native 3DGUT-style backend adapter; this is not the Spark preview.

Uses an existing gsplat installation. It does not estimate cameras, resample input
images, infer a vendor lens model, optimize a person, or silently switch backends.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class NativeCameraContract:
    projection: str
    input_domain: str
    pixel_convention: str = 'pixel_corners'
    evaluation: str = 'ut-projection-and-3d-response'

    def validate(self):
        if self.projection not in ('pinhole', 'opencv_fisheye4'):
            raise ValueError('Only explicit pinhole or OpenCV/KB four-coefficient fisheye is supported; vendor FISHEYE_8 is not interchangeable')
        if self.input_domain not in ('rectified', 'native_fisheye', 'synthetic_known_camera'):
            raise ValueError('Input pixel domain must be explicit')
        if self.input_domain == 'rectified' and self.projection != 'pinhole':
            raise ValueError('Already-rectified images must not receive a second fisheye projection')
        if self.input_domain == 'native_fisheye' and self.projection != 'opencv_fisheye4':
            raise ValueError('Raw fisheye input requires its calibrated native projection')
        if self.pixel_convention != 'pixel_corners' or self.evaluation != 'ut-projection-and-3d-response':
            raise ValueError('Convert pixel convention explicitly before this native renderer; no evaluation fallback')

    def manifest(self):
        self.validate()
        return {'schema':'shuolee-native-renderer-v1','backend':'gsplat','cameraProjection':self.projection,
                'inputDomain':self.input_domain,'pixelConvention':self.pixel_convention,
                'gaussianEvaluation':'ray-3d','projectionApproximation':'unscented-transform',
                'sparkNativeEquivalent':False,'physicalCalibrationValidated':False}

def render_native(means, quats, scales, opacities, colors, viewmats, Ks, width, height,
                  *, contract, radial_coeffs=None, sh_degree=None):
    import torch
    from gsplat import rasterization
    contract.validate()
    if min(width,height) <= 0 or not isinstance(width,int) or not isinstance(height,int):
        raise ValueError('Positive integer image dimensions required')
    if Ks.shape[-2:] != (3,3) or viewmats.shape[-2:] != (4,4):
        raise ValueError('Full K and w2c matrices required')
    if not torch.isfinite(Ks).all() or not torch.isfinite(viewmats).all():
        raise ValueError('Nonfinite camera')
    if Ks.requires_grad:
        raise ValueError('This gsplat backend does not provide K gradients; calibrate separately, never silently optimize a frozen K')
    rotation=viewmats[...,:3,:3]
    if not torch.allclose(viewmats[...,3,:],viewmats.new_tensor([0,0,0,1]).expand_as(viewmats[...,3,:]),atol=1e-6,rtol=0):
        raise ValueError('Expected rigid OpenCV world-to-camera matrices')
    if not torch.allclose(rotation@rotation.transpose(-2,-1),torch.eye(3,device=rotation.device,dtype=rotation.dtype).expand_as(rotation),atol=1e-5,rtol=0) or torch.any((torch.linalg.det(rotation)-1).abs()>1e-5):
        raise ValueError('Camera rotation contains scale, reflection or non-rigid terms')
    if any(not torch.isfinite(x).all() for x in [means,quats,scales,opacities,colors]) or torch.any(scales<=0) or torch.any(torch.linalg.vector_norm(quats,dim=-1)<=1e-12) or torch.any((opacities<0)|(opacities>1)):
        raise ValueError('Expected finite Gaussian parameters, positive linear scales, nonzero wxyz quaternions and opacity probabilities')
    if torch.any(Ks[...,0,0] <= 0) or torch.any(Ks[...,1,1] <= 0):
        raise ValueError('Invalid focal length')
    if torch.any(Ks[...,0,1] != 0) or torch.any(Ks[...,1,0] != 0) or torch.any(Ks[...,2,:] != Ks.new_tensor([0,0,1])):
        raise ValueError('Skew/projective K is not supported; no silent removal')
    if contract.projection == 'pinhole':
        if radial_coeffs is not None:
            raise ValueError('Rectified pinhole path accepts no distortion argument')
        camera_model='pinhole'
    else:
        if radial_coeffs is None or radial_coeffs.shape != Ks.shape[:-2]+(4,) or not torch.isfinite(radial_coeffs).all():
            raise ValueError('OpenCV/KB fisheye requires exactly four finite radial coefficients per camera')
        camera_model='fisheye'
    return rasterization(means=means,quats=quats,scales=scales,opacities=opacities,colors=colors,
                         viewmats=viewmats,Ks=Ks,width=width,height=height,packed=False,
                         with_ut=True,with_eval3d=True,camera_model=camera_model,
                         radial_coeffs=radial_coeffs,sh_degree=sh_degree,render_mode='RGB',eps2d=.3)
