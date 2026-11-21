"""Validate native Mip rendering against gsplat and finite differences."""
import importlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from gsplat import rasterization
from native_mip_filter import filtered_3d,compensate_2d,rotation_matrix
from utils.graphics_utils import getProjectionMatrix

LAB=Path(__file__).resolve().parents[2]
OUT=LAB/'outputs/mini-splatting2-mip-depth-001/native-mip-runtime'


def camera():
    W,H,fx,fy=128,96,95.,80.
    a=.23
    R=torch.tensor([[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]],device='cuda')
    center=torch.tensor([.13,-.2,.1],device='cuda')
    v=torch.eye(4,device='cuda');v[:3,:3]=R.T;v[3,:3]=-center@R.T
    cam=SimpleNamespace(image_width=W,image_height=H,FoVx=2*math.atan(W/(2*fx)),FoVy=2*math.atan(H/(2*fy)),world_view_transform=v,camera_center=center)
    proj=getProjectionMatrix(znear=.01,zfar=100,fovX=cam.FoVx,fovY=cam.FoVy).T.cuda()
    cam.full_proj_transform=v@proj
    return cam


def native(module,cam,means,logscales,quats,logits,sigma,colors):
    scales,opacity=filtered_3d(logscales.exp(),logits.sigmoid(),sigma)
    opacity=compensate_2d(means,scales,quats,opacity,cam)
    settings=module.GaussianRasterizationSettings(image_height=cam.image_height,image_width=cam.image_width,
        tanfovx=math.tan(cam.FoVx/2),tanfovy=math.tan(cam.FoVy/2),bg=torch.zeros(3,device='cuda'),scale_modifier=1.,
        viewmatrix=cam.world_view_transform,projmatrix=cam.full_proj_transform,sh_degree=0,
        campos=cam.camera_center,prefiltered=False,debug=False)
    args=dict(means3D=means,means2D=torch.zeros_like(means,requires_grad=True),opacities=opacity,
        colors_precomp=colors,scales=scales,rotations=torch.nn.functional.normalize(quats,dim=-1))
    if module.__name__.endswith('_ms'):args['culling']=torch.zeros(len(means),device='cuda',dtype=torch.bool)
    return module.GaussianRasterizer(settings)(**args)[0]


def main():
    torch.manual_seed(19);cam=camera();N=24
    xyzcam=torch.rand(N,3,device='cuda')
    xyzcam[:,:2]=(xyzcam[:,:2]-.5)*2;xyzcam[:,2]=2+xyzcam[:,2]
    v=cam.world_view_transform;means=(xyzcam-v[3,:3])@v[:3,:3].T
    logs=torch.rand(N,3,device='cuda')*.8-4.5
    q=torch.randn(N,4,device='cuda');q=torch.nn.functional.normalize(q,dim=-1)
    logits=torch.full((N,1),-.4,device='cuda');sigma=torch.full((N,1),.005,device='cuda')
    colors=torch.rand(N,3,device='cuda')*.7+.1
    scales,opacity=filtered_3d(logs.exp(),logits.sigmoid(),sigma)
    K=torch.tensor([[95,0,64],[0,80,48],[0,0,1.]],device='cuda')
    ref,_,_=rasterization(means,q,scales,opacity[:,0],colors,cam.world_view_transform.T[None],K[None],128,96,
        packed=False,rasterize_mode='antialiased',eps2d=.1,sh_degree=None)
    ref=ref[0].permute(2,0,1)
    results=[]
    for module_name in ['diff_gaussian_rasterization','diff_gaussian_rasterization_ms']:
        module=importlib.import_module(module_name)
        image=native(module,cam,means,logs,q,logits,sigma,colors)
        err=(image-ref).abs();row=dict(module=module_name,forward_mae=float(err.mean()),forward_max=float(err.max()))
        # One isolated Gaussian avoids visibility/order changes during finite differences.
        args=[means[0:1].detach().clone().requires_grad_(),logs[0:1].detach().clone().requires_grad_(),
              q[0:1].detach().clone().requires_grad_(),logits[0:1].detach().clone().requires_grad_()]
        y,x=torch.meshgrid(torch.arange(96,device='cuda'),torch.arange(128,device='cuda'),indexing='ij')
        weight=(.6+.2*torch.sin(x*.03)+.2*torch.cos(y*.04))[None]*torch.tensor([.8,.5,.3],device='cuda')[:,None,None]
        def loss(vals):return (native(module,cam,*vals,sigma[0:1],colors[0:1])*weight).sum()
        value=loss(args);grads=torch.autograd.grad(value,args);checks=[]
        for j,k in [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,2),(3,0)]:
            eps=2e-4 if j==0 else 1e-3
            plus=[z.detach().clone() for z in args];minus=[z.detach().clone() for z in args]
            plus[j][0,k]+=eps;minus[j][0,k]-=eps
            fd=float((loss(plus)-loss(minus))/(2*eps));ad=float(grads[j][0,k])
            rel=abs(fd-ad)/max(abs(fd),abs(ad),.02)
            checks.append(dict(parameter=['means','log_scales','quaternion','opacity_logit'][j],component=k,autograd=ad,finite_difference=fd,relative_error=rel))
        row['gradient_checks']=checks
        row['passed']=row['forward_mae']<2e-5 and row['forward_max']<.003 and max(c['relative_error'] for c in checks)<.06
        results.append(row)
    out=dict(status='passed' if all(x['passed'] for x in results) else 'failed',checks=results,
        camera='nonidentity pose; rectangular image and unequal fx/fy',reference='gsplat antialiased eps2d0.1',
        interpretation='forward image agreement and finite differences validate 3D smoothing plus view-dependent opacity gradients; training acceptance remains separate')
    (OUT/'validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
    return 0 if out['status']=='passed' else 1

if __name__=='__main__':raise SystemExit(main())
