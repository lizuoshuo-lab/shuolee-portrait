"""Check differentiable expected-z semantics, native culling, and gradients."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from gsplat import rasterization
from validate_native_mip import camera
from native_mip_filter import filtered_3d
from native_depth_guidance import expected_depth,inverse_depth_loss

ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'

class Mock:
    def __init__(self,values,cam):
        self._xyz,self.logs,self.q,self.logits=values
        self._mip_iteration=0;self._mip_cameras=[cam]
        self._mip_filter=torch.full((len(self._xyz),1),.005,device='cuda')
        self._mip_cache_key=(id(self._xyz),len(self._xyz),0)
    @property
    def get_xyz(self):return self._xyz
    @property
    def get_scaling(self):return self.logs.exp()
    @property
    def get_rotation(self):return torch.nn.functional.normalize(self.q,dim=-1)
    @property
    def get_opacity(self):return self.logits.sigmoid()

def main():
    out=ROOT/'native-depth-runtime';out.mkdir(exist_ok=True)
    torch.manual_seed(31);cam=camera();n=12
    local=torch.rand(n,3,device='cuda');local[:,:2]=(local[:,:2]-.5)*.5;local[:,2]+=2.
    v=cam.world_view_transform;xyz=(local-v[3,:3])@v[:3,:3].T
    values=[xyz,torch.randn(n,3,device='cuda')*.1-1.8,
        torch.nn.functional.normalize(torch.randn(n,4,device='cuda'),dim=-1),
        torch.full((n,1),.3,device='cuda')]
    pc=Mock(values,cam);z,alpha=expected_depth(cam,pc)
    scales,opacity=filtered_3d(pc.get_scaling,pc.get_opacity,pc._mip_filter)
    k=torch.tensor([[95.,0,64],[0,80,48],[0,0,1]],device='cuda')
    ref,ar,_=rasterization(pc.get_xyz,pc.get_rotation,scales,opacity[:,0],torch.zeros(n,3,device='cuda'),
        v.T[None],k[None],128,96,packed=False,rasterize_mode='antialiased',eps2d=.1,render_mode='RGB+ED')
    mask=(alpha>.05)&(ar[0,:,:,0]>.05)
    mae=float((z[mask]-ref[0,:,:,3][mask]).abs().mean())
    alpha_mae=float((alpha-ar[0,:,:,0]).abs().mean())
    _,none=expected_depth(cam,pc,torch.ones(n,device='cuda',dtype=torch.bool))
    yy,xx=torch.meshgrid(torch.arange(96,device='cuda'),torch.arange(128,device='cuda'),indexing='ij')
    target=z.detach().clamp_min(.2).reciprocal()*(1.1+.04*torch.sin(xx*.13))
    # Restrict finite differences to the common interior: alpha cutoffs and tile
    # support are discrete decisions, not differentiable geometry operations.
    conf=(mask&(xx>60)&(xx<68)&(yy>44)&(yy<52)).float()
    assert int(conf.sum())==49
    valid=torch.ones(1,96,128,device='cuda')
    def objective(vals):
        d,a=expected_depth(cam,Mock(vals,cam))
        return inverse_depth_loss(d,a,target,conf,valid,target.new_tensor(.4),minimum_alpha=0)[0]
    args=[x.detach().clone().requires_grad_() for x in values]
    loss=objective(args);grads=torch.autograd.grad(loss,args);checks=[]
    independent=Mock(args,cam)
    sf,of=filtered_3d(independent.get_scaling,independent.get_opacity,independent._mip_filter)
    independent_rgb,independent_alpha,_=rasterization(independent.get_xyz,independent.get_rotation,sf,of[:,0],torch.zeros(n,3,device='cuda'),
        v.T[None],k[None],128,96,packed=False,rasterize_mode='antialiased',eps2d=.1,render_mode='RGB+ED')
    independent_loss=inverse_depth_loss(independent_rgb[0,:,:,3],independent_alpha[0,:,:,0],target,conf,valid,target.new_tensor(.4),minimum_alpha=0)[0]
    independent_grads=torch.autograd.grad(independent_loss,args)
    independent_max=max(float((a-b).abs().max()) for a,b in zip(grads,independent_grads))
    for j,k in [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,2),(3,0)]:
        # Largest useful derivative in each parameter group avoids vacuous zero checks.
        idx=int(grads[j][:,k].abs().argmax());eps=3e-4 if j==0 else 2e-3
        plus=[x.detach().clone() for x in args];minus=[x.detach().clone() for x in args]
        plus[j][idx,k]+=eps;minus[j][idx,k]-=eps
        fd=float((objective(plus)-objective(minus))/(2*eps));ad=float(grads[j][idx,k])
        rel=abs(fd-ad)/max(abs(fd),abs(ad),1e-4)
        checks.append(dict(parameter=['means','logscales','quaternion','opacity'][j],row=idx,column=k,
            autograd=ad,finite_difference=fd,relative_error=rel))
    # Exact no-gradient invariant for excluded observations.
    a=[x.detach().clone().requires_grad_() for x in values];d,al=expected_depth(cam,Mock(a,cam))
    zero,_=inverse_depth_loss(d,al,target,conf,torch.zeros_like(valid),target.new_tensor(.4),minimum_alpha=0)
    zero_grads=torch.autograd.grad(zero,a)
    excluded_max=max(float(g.abs().max()) for g in zero_grads)
    passed=mae<2e-4 and alpha_mae<1e-5 and float(none.abs().max())==0 and excluded_max==0 and independent_max<1e-4 and max(x['relative_error'] for x in checks)<.08
    result=dict(status='passed' if passed else 'failed',depth_mae=mae,alpha_mae=alpha_mae,
        all_culled_alpha_max=float(none.abs().max()),excluded_loss=float(zero),excluded_gradient_max=excluded_max,
        gradient_checks=checks,independent_gradient_max_absolute_error=independent_max,
        reference='gsplat alpha-normalized expected camera z; unequal fx/fy and rotated camera',
        note='Frozen visibility mask and frozen 3D sampling filter during finite differences; training acceptance remains separate')
    (out/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
