"""Observed image correspondences constrain a shared static Gaussian surface.

Pixels are zero-based centers. Legacy cameras store a corner-coordinate
principal point and need +0.5; the opt-in consistent camera already stores
integer pixel centers and needs no second correction. The rendered
plane normal and plane distance provide the differentiable surface position.
No learned camera adjustment or per-frame deformation is introduced.
"""
import torch
import torch.nn.functional as F
import numpy as np


def surface_world(pkg,cam,pixel):
    size=pixel.new_tensor([cam.image_width-1,cam.image_height-1])
    grid=(pixel/size*2-1)[None,:,None]
    def sample(x):return F.grid_sample(x[None],grid,align_corners=True)[0,:,:,0].T
    normal=sample(pkg['rendered_normal']);distance=sample(pkg['rendered_distance'])[:,0]
    alpha=sample(pkg['rendered_alpha'])[:,0]
    offset=0. if getattr(cam,'shuolee_pixel_center_consistent',False) else .5
    rays=torch.cat([(pixel+offset-pixel.new_tensor([cam.Cx,cam.Cy]))/pixel.new_tensor([cam.Fx,cam.Fy]),torch.ones_like(pixel[:,:1])],1)
    denominator=-(normal*rays).sum(-1)
    safe=torch.where(denominator.abs()>1e-5,denominator,torch.ones_like(denominator))
    depth=distance/safe
    camera_point=rays*depth[:,None]
    world=(camera_point-camera_point.new_tensor(cam.T))@camera_point.new_tensor(cam.R.T)
    valid=(alpha.detach()>.8)&(denominator.detach()>1e-5)&(depth.detach()>.1)&torch.isfinite(world.detach()).all(-1)
    return world,valid,depth,alpha


def projection_loss(pkg,cam,source_pixel,target_pixel,target_w2c,target_K,weights=None):
    world,valid,depth,alpha=surface_world(pkg,cam,source_pixel)
    cp=torch.einsum('nij,nj->ni',target_w2c[:,:3,:3],world)+target_w2c[:,:3,3]
    h=torch.einsum('nij,nj->ni',target_K,cp)
    uv=h[:,:2]/h[:,2:3].clamp_min(1e-5)-.5
    valid=valid&(cp.detach()[:,2]>.1)&torch.isfinite(uv.detach()).all(-1)
    residual=(uv-target_pixel).norm(dim=-1)
    # Pseudo-Huber in training pixels, delta2. No residual-dependent admission:
    # a wrong model must not reject the observations needed to correct itself.
    penalty=(torch.sqrt(1+(residual/2).square())-1)*2
    if weights is None:weights=torch.ones_like(residual)
    loss=(penalty[valid]*weights[valid]).sum()/weights[valid].sum().clamp_min(1e-12) if valid.any() else pkg['render'].sum()*0
    return loss,dict(valid=valid,projection=uv,residual=residual,depth=depth,alpha=alpha)


class ObservedPairs:
    """Deterministic image-only correspondences; never consumes training RNG."""
    def __init__(self,path,cams,rows):
        a=np.load(path);self.cams=cams;self.groups=[];self.arrays=[]
        for i,cam in enumerate(cams):
            ids=np.flatnonzero(a['source']==i);target=a['target'][ids]
            assert len(ids)>0,(i,'No observed pairs')
            sc=np.array([cam.image_width,cam.image_height])/rows[i]['wh']
            target_sc=np.array([[cams[j].image_width,cams[j].image_height] for j in target])/np.array([rows[j]['wh'] for j in target])
            E=np.array([rows[j]['w2c'] for j in target]);K=np.array([np.diag([*scale,1])@np.array(rows[j]['K']) for j,scale in zip(target,target_sc)])
            values=[(a['source_pixel'][ids]+.5)*sc-.5,(a['target_pixel'][ids]+.5)*target_sc-.5,E,K]
            self.arrays.append([torch.as_tensor(v,dtype=torch.float32,device='cuda') for v in values])
            self.groups.append([torch.as_tensor(np.flatnonzero(a['head'][ids]==head),device='cuda') for head in [True,False]])

    def loss(self,pkg,source,visit):
        selected=[]
        for group,cap in zip(self.groups[source],[32,96]):
            n=len(group)
            if n:selected.append(group if n<=cap else group[(torch.arange(cap,device='cuda')+visit*cap)%n])
        ids=torch.cat(selected);values=[a[ids] for a in self.arrays[source]]
        return projection_loss(pkg,self.cams[source],*values)
