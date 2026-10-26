"""Opt-in shared-person geometry loss using the actual 3DGS RGB weights.

The visibility assignment is detached ONLY for this auxiliary geometry term.
Original RGB/mask derivatives remain untouched. Every bilinear corner has its
own calibrated ray and per-contributor plane intersection; mean layers cannot
cancel. These local planes approximate 3DGS surfaces, not a volumetric GT.
"""
from types import SimpleNamespace
import numpy as np
import torch
from gaussian_renderer import render as original_render

def initialize_surface(model, path, points_sha):
    a=np.load(path);assert str(a['points_sha256'])==points_sha
    assert len(a['normal'])==len(model._xyz) and a['confidence'].all()
    with torch.no_grad():
        model._rotation.copy_(torch.as_tensor(a['quaternion'],device='cuda'))
        model._scaling[:,2].add_(np.log(.25))
    n,axis=model.get_smallest_axis(True)
    expected=torch.as_tensor(a['normal'],device='cuda')
    assert bool((axis==2).all()) and float((n-expected).abs().max())<2e-6
    return dict(points=len(n),normal_axis=2,ratio=.25,normal_max_abs=float((n-expected).abs().max()))

def native_weights(model,cam,pipe,uv):
    """One RGB forward and four color adjoints, up to three queried features."""
    assert 1<=len(uv)<=3 and not pipe.compute_cov3D_python
    proxy=SimpleNamespace(get_xyz=model.get_xyz.detach(),get_opacity=model.get_opacity.detach(),
        get_scaling=model.get_scaling.detach(),get_rotation=model.get_rotation.detach(),
        active_sh_degree=model.active_sh_degree,use_app=False)
    color=torch.ones_like(proxy.get_xyz,requires_grad=True)
    p=original_render(cam,proxy,pipe,torch.zeros(3,device='cuda'),override_color=color,return_plane=False,return_depth_normal=False)
    base=torch.floor(uv).long();fraction=uv-base
    channel=torch.arange(len(uv),device=uv.device);corners=[];parity=[]
    for k,(dx,dy) in enumerate([(0,0),(1,0),(0,1),(1,1)]):
        px=base+base.new_tensor([dx,dy])
        assert bool((px>=0).all()) and bool((px[:,0]<cam.image_width).all()) and bool((px[:,1]<cam.image_height).all())
        value=p['render'][channel,px[:,1],px[:,0]]
        w=torch.autograd.grad(value.sum(),color,retain_graph=k<3)[0][:,:len(uv)].detach()
        coeff=(fraction[:,0] if dx else 1-fraction[:,0])*(fraction[:,1] if dy else 1-fraction[:,1])
        parity.append((w.sum(0)-value.detach()).abs())
        corners.append((px,w,coeff.detach()))
    err=float(torch.stack(parity).max())
    assert err<2e-4,('native contribution parity',err)
    return corners,err

def pixel_error(means,normals,weights,cam,pixel,target,sigma):
    w=weights.detach().double();ids=torch.nonzero(w>0).flatten()
    if not len(ids):return None,dict(reason='empty',alpha=0.,retained_mass=0.)
    n=normals[ids].double();m=means[ids].double();w=w[ids]
    # Camera is already corrected to integer sample centers, unlike source K.
    u,v=pixel
    ray=n.new_tensor([(float(u)-cam.Cx)/cam.Fx,(float(v)-cam.Cy)/cam.Fy,1.])
    rotation=cam.world_view_transform[:3,:3].detach().double()
    world_ray=rotation@ray;origin=cam.camera_center.detach().double()
    denominator=(n*world_ray).sum(-1);cosine=denominator/world_ray.norm()
    valid=cosine.detach().abs()>=.05
    depth=((m-origin)*n).sum(-1)/torch.where(valid,denominator,torch.ones_like(denominator))
    valid=valid&torch.isfinite(depth.detach())&(depth.detach()>.2)
    mass=(w*valid).sum()/w.sum()
    if float(mass)<.95:return None,dict(reason='grazing_or_invalid_depth',alpha=float(w.sum()),retained_mass=float(mass))
    hit=origin+depth[:,None]*world_ray
    normalized=w*valid/(w*valid).sum()
    delta=torch.where(valid[:,None],hit-target.detach().double(),torch.zeros_like(hit))
    expected=(normalized*delta.square().sum(-1)).sum()
    average=(normalized[:,None]*hit).sum(0)
    variance=(normalized*(hit-average).square().sum(-1)).sum()
    return expected/(sigma.detach().double()**2),dict(reason='valid',alpha=float(w.sum()),retained_mass=float(mass),
        expected_squared_error=float(expected.detach()),variance=float(variance.detach()),mean_squared_error=float((average-target).square().sum().detach()),contributors=len(ids))

def volume_pixel_error(means,quaternions,scales,weights,cam,pixel,target,sigma):
    """Ray-restricted 3D Gaussian mode, with its finite positive covariance.

    z* = d' Q (mu-o) / d' Q d, Q = R diag(1/s^2) R'. Unlike a
    tangent plane, this expression is well-defined at grazing incidence. The
    present scale eigenvalues and native RGB weights are detached for this
    auxiliary; center and rotation remain differentiable. Original photometry
    continues to optimize every parameter. No projection or sorting is changed.
    """
    w=weights.detach().double();ids=torch.nonzero(w>0).flatten()
    if not len(ids):return None,dict(reason='empty',alpha=0.,retained_mass=0.)
    w=w[ids];m=means[ids].double();q=torch.nn.functional.normalize(quaternions[ids].double(),dim=-1)
    qw,qx,qy,qz=q.unbind(-1)
    rot=torch.stack([1-2*(qy*qy+qz*qz),2*(qx*qy-qw*qz),2*(qx*qz+qw*qy),
        2*(qx*qy+qw*qz),1-2*(qx*qx+qz*qz),2*(qy*qz-qw*qx),
        2*(qx*qz-qw*qy),2*(qy*qz+qw*qx),1-2*(qx*qx+qy*qy)],-1).reshape(-1,3,3)
    u,v=pixel;ray=m.new_tensor([(float(u)-cam.Cx)/cam.Fx,(float(v)-cam.Cy)/cam.Fy,1.])
    d=cam.world_view_transform[:3,:3].detach().double()@ray;o=cam.camera_center.detach().double()
    inv_scale=scales[ids].detach().double().reciprocal()
    assert bool(torch.isfinite(inv_scale).all())
    inv_scale=inv_scale/inv_scale.max(-1,keepdim=True).values
    a=torch.einsum('nij,j->ni',rot.transpose(-1,-2),d)*inv_scale
    b=torch.einsum('nij,nj->ni',rot.transpose(-1,-2),m-o)*inv_scale
    depth=(a*b).sum(-1)/a.square().sum(-1)
    valid=torch.isfinite(depth.detach())&(depth.detach()>.2)
    mass=(w*valid).sum()/w.sum()
    if float(mass)<.95:return None,dict(reason='invalid_conditional_depth',alpha=float(w.sum()),retained_mass=float(mass))
    hit=o+depth[:,None]*d;normalized=w*valid/(w*valid).sum()
    delta=torch.where(valid[:,None],hit-target.detach().double(),torch.zeros_like(hit))
    expected=(normalized*delta.square().sum(-1)).sum();average=(normalized[:,None]*hit).sum(0)
    variance=(normalized*(hit-average).square().sum(-1)).sum()
    return expected/(sigma.detach().double()**2),dict(reason='valid',alpha=float(w.sum()),retained_mass=float(mass),expected_squared_error=float(expected.detach()),
        variance=float(variance.detach()),mean_squared_error=float((average-target).square().sum().detach()),contributors=len(ids))

class NativePersonAlignment:
    def __init__(self,path,cams,rows,geometry='plane'):
        assert geometry in ['plane','conditional_volume'];self.geometry=geometry
        a=np.load(path);assert np.array_equal(a['w2c'],np.asarray([r['w2c'] for r in rows]))
        assert all(np.array_equal(a['K'],np.asarray(r['K'])) for r in rows)
        assert np.array_equal(a['wh'],np.asarray([r['wh'] for r in rows]))
        self.cams=cams;self.xyz=torch.as_tensor(a['xyz'],device='cuda',dtype=torch.float32)
        self.sigma=torch.as_tensor(a['sigma_world'],device='cuda',dtype=torch.float32);self.point_ids=a['point_ids']
        ob=a['observations'];self.observations=[];self.serial=0
        for i,cam in enumerate(cams):
            use=(ob[:,0]==i)&(ob[:,4]==0)&(ob[:,5]==0);q=ob[use].copy()
            uv=(q[:,2:4]+.5)*np.array([cam.image_width,cam.image_height])/a['wh'][i]-.5
            self.observations.append(dict(ids=torch.as_tensor(q[:,1],device='cuda',dtype=torch.long),uv=torch.as_tensor(uv,device='cuda',dtype=torch.float32),regions=q[:,6].astype(int)))
    def select(self,i,visit,count=3):
        # Own RNG does not perturb the existing camera/NCC sampling schedule.
        n=len(self.observations[i]['ids']);return np.random.default_rng(6060+i*100003+visit).choice(n,min(n,count),replace=False)
    def loss(self,model,cam_index,pipe,visit=0,selection=None):
        obs=self.observations[cam_index];which=self.select(cam_index,visit) if selection is None else np.asarray(selection)
        ids=obs['ids'][which];uv=obs['uv'][which];corners,parity=native_weights(model,self.cams[cam_index],pipe,uv)
        normals=model.get_smallest_axis();losses=[];records=[]
        for j,k in enumerate(which):
            values=[];coefficients=[];details=[]
            for pixel,w,coeff in corners:
                if float(coeff[j])==0:continue
                if self.geometry=='conditional_volume':
                    value,info=volume_pixel_error(model.get_xyz,model._rotation,model.get_scaling,w[:,j],self.cams[cam_index],pixel[j],self.xyz[ids[j]],self.sigma[ids[j]])
                else:
                    value,info=pixel_error(model.get_xyz,normals,w[:,j],self.cams[cam_index],pixel[j],self.xyz[ids[j]],self.sigma[ids[j]])
                details.append(info)
                if value is not None:values.append(value);coefficients.append(coeff[j])
            valid=len(values)==len(details)
            if valid:
                scaled=sum(a*b for a,b in zip(values,coefficients));losses.append(2*(torch.sqrt(1+scaled)-1))
            records.append(dict(point_id=int(self.point_ids[int(ids[j])]),region=int(obs['regions'][k]),valid=valid,corners=details))
        # A missing query is reported, never called a successful zero residual.
        value=torch.stack(losses).sum()/len(which) if losses else model._xyz.sum()*0
        return value,dict(observation_count=len(which),available_count=len(losses),native_alpha_parity=parity,queries=records)
