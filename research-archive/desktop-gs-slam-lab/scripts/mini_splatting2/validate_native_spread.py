"""Independent forward/backward and finite-difference checks before any training."""
import json
from pathlib import Path

import torch
from gsplat import rasterization
from validate_native_depth import Mock
from validate_native_mip import camera
from native_mip_filter import filtered_3d
from native_depth_spread import center_depth_variance,spread_loss

ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'


def main():
    out=ROOT/'native-spread-runtime';out.mkdir(exist_ok=True);target=out/'validation.json';assert not target.exists()
    torch.manual_seed(31);cam=camera();n=12
    local=torch.rand(n,3,device='cuda');local[:,:2]=(local[:,:2]-.5)*.5;local[:,2]+=2.
    v=cam.world_view_transform;xyz=(local-v[3,:3])@v[:3,:3].T
    values=[xyz,torch.randn(n,3,device='cuda')*.1-1.8,
        torch.nn.functional.normalize(torch.randn(n,4,device='cuda'),dim=-1),torch.full((n,1),.3,device='cuda')]
    normalizer=torch.tensor(.4,device='cuda');inverse=torch.full((96,128),.4,device='cuda')
    yy,xx=torch.meshgrid(torch.arange(96,device='cuda'),torch.arange(128,device='cuda'),indexing='ij')
    confidence=((xx>60)&(xx<68)&(yy>44)&(yy<52)).float();valid=torch.ones(1,96,128,device='cuda')
    def independent(vals):
        pc=Mock(vals,cam);sf,of=filtered_3d(pc.get_scaling,pc.get_opacity,pc._mip_filter)
        q=(pc.get_xyz@v[:3,:3]+v[3,:3])[:,2]*normalizer
        features=torch.stack([q,q*q,torch.ones_like(q)],-1)
        k=torch.tensor([[95.,0,64],[0,80,48],[0,0,1]],device='cuda')
        image,_,_=rasterization(pc.get_xyz,pc.get_rotation,sf,of[:,0],features,v.T[None],k[None],128,96,
            packed=False,rasterize_mode='antialiased',eps2d=.1)
        a=image[0,...,2];mean=image[0,...,0]/a.clamp_min(1e-8)
        return (image[0,...,1]/a.clamp_min(1e-8)-mean.square()).clamp_min(0),a
    def objective(vals,reference=False,mask=valid):
        var,a=independent(vals) if reference else center_depth_variance(cam,Mock(vals,cam),normalizer)
        return spread_loss(var,a,inverse,confidence,mask,normalizer,minimum_alpha=0)[0]
    args=[x.clone().detach().requires_grad_() for x in values]
    loss=objective(args);grads=torch.autograd.grad(loss,args)
    other=objective(args,reference=True);other_grads=torch.autograd.grad(other,args)
    var,a=center_depth_variance(cam,Mock(values,cam),normalizer);rv,ra=independent(values);mask=(a>.05)&(ra>.05)
    forward=float((var[mask]-rv[mask]).abs().mean());alpha_error=float((a-ra).abs().mean())
    gradient_error=max(float((x-y).abs().max()) for x,y in zip(grads,other_grads));checks=[]
    for j,k in [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,2),(3,0)]:
        index=int(grads[j][:,k].abs().argmax());eps=3e-4 if j==0 else 2e-3
        plus=[x.detach().clone() for x in args];minus=[x.detach().clone() for x in args]
        plus[j][index,k]+=eps;minus[j][index,k]-=eps
        fd=float((objective(plus)-objective(minus))/(2*eps));ad=float(grads[j][index,k]);relative=abs(fd-ad)/max(abs(fd),abs(ad),1e-4)
        checks.append(dict(group=j,column=k,row=index,autograd=ad,finite_difference=fd,relative_error=relative))
    excluded=objective(args,mask=torch.zeros_like(valid));zero_grads=torch.autograd.grad(excluded,args)
    zero_error=max(float(x.abs().max()) for x in zero_grads)
    _,culled=center_depth_variance(cam,Mock(values,cam),normalizer,torch.ones(n,device='cuda',dtype=torch.bool))
    passed=forward<2e-5 and alpha_error<1e-5 and gradient_error<1e-4 and zero_error==0 and float(culled.abs().max())==0 and max(r['relative_error'] for r in checks)<.08
    result=dict(status='passed' if passed else 'failed',variance_forward_mae=forward,alpha_mae=alpha_error,
        independent_gradient_max_absolute_error=gradient_error,finite_differences=checks,excluded_loss=float(excluded),
        excluded_gradient_max=zero_error,all_culled_alpha_max=float(culled.abs().max()),
        scope='numerical checks only; center-depth variance is not geometry GT or training acceptance')
    target.write_text(json.dumps(result,indent=2));print(json.dumps(result));return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
