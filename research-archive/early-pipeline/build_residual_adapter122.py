"""One-time creation of a separate residual-output adapter; old modules untouched."""
from pathlib import Path
H=Path(__file__).parent
p=H/'native_portrait_renderer122_v3.py'
if p.exists(): raise FileExistsError(p)
s=(H/'native_portrait_renderer122_v2.py').read_text(encoding='utf-8-sig')
a=s.index('from dataclasses import dataclass'); b=s.index('\ndef render_native(')
s='''"""Versioned output-residual Brown projection and ray-3D renderer.

Uses the proven v2 geometry-only bounds; identical frozen6+2+4 coefficients enter
both UT projection and pixel-ray evaluation. No camera learning or image warp.
"""\nimport math\nfrom native_residual_camera122 import ResidualRenderContract\n
REVISION='122-ray3d-output-residual-v3'
BOUNDS='ut-3.33-sigma-with-eps2d-no-opacity-cull'
UT_SETTINGS=dict(alpha=.1,beta=2.,kappa=0.,in_image_margin_factor=.1,require_all_sigma_points_valid=True)
'''+s[b:]
s=s.replace('radial_coeffs=None,sh_degree=None','radial_coeffs,tangential_coeffs,thin_prism_coeffs,sh_degree=None')
s=s.replace("isinstance(contract,NativeCameraContractV2)","isinstance(contract,ResidualRenderContract)").replace('Explicit v2 contract required','Explicit output-residual contract required')
s=s.replace("+([] if radial_coeffs is None else [radial_coeffs])", "+[radial_coeffs,tangential_coeffs,thin_prism_coeffs]")
s=s.replace("if Ks.requires_grad or viewmats.requires_grad:","if any(x.requires_grad for x in [Ks,viewmats,radial_coeffs,tangential_coeffs,thin_prism_coeffs]):")
s=s.replace('This adapter freezes K and extrinsics; no camera optimization','This adapter freezes K, extrinsics and distortion; no camera optimization')
a=s.index("    if contract.projection=='pinhole':");b=s.index('    ut=UnscentedTransformParameters',a)
s=s[:a]+'''    if radial_coeffs.shape!=(c,6) or tangential_coeffs.shape!=(c,2) or thin_prism_coeffs.shape!=(c,4):
        raise ValueError('Explicit6+2+4 coefficients per camera required; no vendor FISHEYE_8 reinterpretation')
    camera_model='pinhole'
'''+s[b:]
s=s.replace('radial_coeffs=radial_coeffs,rolling_shutter=', 'radial_coeffs=radial_coeffs,tangential_coeffs=tangential_coeffs,thin_prism_coeffs=thin_prism_coeffs,rolling_shutter=')
s+='''
def render_cameras(parameters,cameras,*,contract,sh_degree=3):
    import torch
    from native_residual_camera122 import validate_camera
    if not cameras: raise ValueError('Explicit recorded cameras required')
    for c in cameras:
        validate_camera(c)
        if c['residual_projection']['input_domain']!=contract.input_domain:
            raise ValueError('Camera and renderer input domains differ')
    if len({(c['width'],c['height']) for c in cameras})!=1:
        raise ValueError('Batch requires equal native dimensions; no implicit image resize')
    t=lambda v:torch.tensor(v,device=parameters['means'].device,dtype=torch.float32)
    return render_native(parameters['means'],parameters['quats'],parameters['scales'],parameters['opacities'],parameters['sh'],
        t([c['w2c'] for c in cameras]),t([c['K'] for c in cameras]),cameras[0]['width'],cameras[0]['height'],
        contract=contract,sh_degree=sh_degree,
        radial_coeffs=t([c['residual_projection']['radial'] for c in cameras]),
        tangential_coeffs=t([c['residual_projection']['tangential'] for c in cameras]),
        thin_prism_coeffs=t([c['residual_projection']['thin_prism'] for c in cameras]))
'''
p.write_text(s,encoding='utf-8')
print(p)
