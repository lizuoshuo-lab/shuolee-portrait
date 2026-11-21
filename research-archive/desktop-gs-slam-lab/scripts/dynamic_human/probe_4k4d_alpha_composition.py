"""CPU semantic check of the actual installed PyTorch3D compositor."""
from pathlib import Path
import json,torch,hashlib
from pytorch3d.renderer import AlphaCompositor
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
OUT=LAB/'outputs/dynamic-human-4k4d-alpha-composition-041';OUT.mkdir(exist_ok=True)
idx=torch.tensor([[[[0]],[[1]]]],dtype=torch.int64)
opacity=torch.tensor([.4,.5],requires_grad=True)
alphas=opacity.view(1,2,1,1)
colors=torch.tensor([[.1,.8],[.6,.2],[.3,.4]])
legacy=AlphaCompositor()(idx,alphas,torch.cat([colors,opacity[None]],0))
standard=AlphaCompositor()(idx,alphas,torch.cat([colors,torch.ones_like(opacity)[None]],0))
expected=1-torch.prod(1-opacity)
assert torch.equal(legacy[:,:3],standard[:,:3])
assert torch.allclose(standard[0,3,0,0],expected,atol=1e-7)
assert abs(float(legacy[0,3,0,0])-.31)<1e-6
assert abs(float(standard[0,3,0,0])-.7)<1e-6
old_grad=torch.autograd.grad(legacy[0,3,0,0],opacity,retain_graph=True)[0]
new_grad=torch.autograd.grad(standard[0,3,0,0],opacity)[0]
source=LAB/'repos/4K4D/easyvolcap/models/samplers/point_planes_sampler.py'
gl=LAB/'repos/4K4D/easyvolcap/utils/gl_utils.py'
needle='rgb = torch.cat([rgb, occ, dpt], dim=-1)'
assert needle in source.read_text() and needle in gl.read_text()
report=dict(status='actual_CPU_compositor_semantic_difference_confirmed',
    opacities=opacity.detach().tolist(),legacy_acc=float(legacy[0,3,0,0]),standard_acc=float(standard[0,3,0,0]),
    legacy_alpha_gradient=old_grad.tolist(),standard_alpha_gradient=new_grad.tolist(),raw_rgb_unchanged=True,
    formulas=dict(legacy='sum_i T_i a_i o_i',standard='sum_i T_i a_i = 1 - product_i(1-a_i)',point_alpha='a_i = o_i * radial_falloff_i'),
    scope='Both official PyTorch3D and differentiable OpenGL code append point opacity as an accumulated feature. This is not unique to the local fallback. Actual quality impact not yet measured.',
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),gl_source_sha256=hashlib.sha256(gl.read_bytes()).hexdigest(),goal_complete=False)
(OUT/'cpu-semantic-check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
