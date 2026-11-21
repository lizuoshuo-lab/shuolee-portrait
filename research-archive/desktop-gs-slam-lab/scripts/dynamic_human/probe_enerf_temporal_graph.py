"""Independent B-spline arithmetic and an actual frozen-model GPU derivative."""
from pathlib import Path
import json,hashlib,shutil,math
import numpy as np
import torch
from scipy.interpolate import BSpline
from enerf_temporal_graph import cubic_bspline,GraphMotion,ResidualGraphRenderer
from enerf_graph_renderer import GraphMotion as OldGraph,ResidualGraphRenderer as OldRenderer

L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');R=L/'outputs/dynamic-human-enerf-temporal-basis-078';B=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
torch.set_num_threads(8)
x=np.linspace(-4,4,1601);sx=BSpline.basis_element([-2,-1,0,1,2],extrapolate=False);expected=np.nan_to_num(sx(x))
xx=torch.tensor(x,dtype=torch.float64,requires_grad=True);out=cubic_bspline(xx)
error=float(np.max(abs(out.detach().numpy()-expected)));assert error<2e-15
d1=torch.autograd.grad(out.sum(),xx,create_graph=True)[0];d2=torch.autograd.grad(d1.sum(),xx)[0]
assert np.max(abs(d1.detach().numpy()-np.nan_to_num(sx.derivative()(x))))<3e-15
assert np.max(abs(d2.detach().numpy()-np.nan_to_num(sx.derivative(2)(x))))<3e-15
cp_path=B/'residual_graph/resume.pt';before=sha(cp_path);cp=torch.load(cp_path,map_location='cpu',weights_only=False)
g=GraphMotion(R/'graph.npz',cp['scene_scale']);old=OldGraph(B/'graph.npz',cp['scene_scale'])
old.load_state_dict(cp['graph']);missing=g.load_state_dict(cp['graph'],strict=False)
assert set(missing.missing_keys)=={'local_centers','local_rotation_raw','local_translation_raw'} and not missing.unexpected_keys
taus=torch.linspace(-1.5,1.5,601,device='cuda')
bs=torch.stack([g.basis(t) for t in taus]);assert float(bs.abs().sum(-1).max())<=2.000001
for tau in [taus[0],taus[160],taus[300],taus[450],taus[-1]]:
    a,b=g(tau);oa,ob=old(tau);assert torch.equal(a,oa) and torch.equal(b,ob)
ref=torch.tensor(0.,device='cuda');a,b=g(ref)
assert torch.equal(a,torch.eye(3,device='cuda').expand_as(a)) and torch.count_nonzero(b)==0
model=ResidualGraphRenderer(B/'head/capture-affines.npz',g);original=OldRenderer(B/'head/capture-affines.npz',old)
z=np.load(L/'outputs/dynamic-human-enerf-articulated-control-064/native-input/camera01-frame000744.npz');tensor=lambda v:torch.tensor(v,dtype=torch.float32,device='cuda')
p={k:v.cuda() for k,v in cp['params'].items()};W,K=tensor(z['w2c']),tensor(z['K'])
with torch.no_grad():
    rgb,alpha,_=model.render(p,'01',744,W,K,1920,1080,2)
    org,orgalpha,_=original.render(p,'01',744,W,K,1920,1080,2)
    max_rgb_diff=float((rgb-org).abs().max());max_alpha_diff=float((alpha-orgalpha).abs().max())
    assert max_rgb_diff==0 and max_alpha_diff==0
del rgb,alpha,org,orgalpha,original,old
rgb,alpha,info=model.render(p,'01',744,W,K,1920,1080,2)
target=tensor(z['rgb']/255)*tensor(z['matte'])[...,None]+.5*(1-tensor(z['matte'])[...,None]);valid=tensor(z['valid'])[...,None]
loss=(((rgb[0]+.5*(1-alpha[0]))-target).abs()*valid).sum()/(3*valid.sum())
loss.backward()
grad={n:float(v.grad.abs().sum()) for n,v in g.named_parameters()}
assert all(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum()>0 for v in g.parameters())
# This pass never steps an optimizer and cannot alter the preserved baseline.
assert sha(cp_path)==before
for n in ['local_rotation_raw','local_translation_raw']:assert torch.count_nonzero(getattr(g,n))==0
report=dict(status='independent_Bspline_values_first_second_derivatives_identity_and_real_RGB_gradients_verified',
    scipy_grid=1601,bspline_max_value_difference=error,basis_max_l1=float(bs.abs().sum(-1).max()),
    native_rgb_zero_extension_max_difference=max_rgb_diff,native_alpha_zero_extension_max_difference=max_alpha_diff,
    real_source_camera='01',real_source_frame=744,resolution=[1920,1080],
    parameter_gradient_l1=grad,actual_parameter_count=sum(v.numel() for v in g.parameters()),
    max_rotation_bound_rad=model.max_rotation_bound,less_than_pi_over2=model.max_rotation_bound<math.pi/2,
    baseline_checkpoint_sha256=before,optimizer_updates=0,reference_identity_exact=True,
    scope='Discarded frozen074 checkpoint probe. Only original training photo744 used for derivative. No probe parameters used to initialize078training.',goal_complete=False)
(R/'runtime-probe.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
shutil.copy2(__file__,R/Path(__file__).name)
print(json.dumps(report))
