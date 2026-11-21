"""Canonical Gaussians carried by exact affine covariance and body motion."""
from pathlib import Path
import torch,numpy as np
from gsplat.rendering import rasterization
from gsplat.cuda._wrapper import quat_scale_to_covar_preci,spherical_harmonics

class ArticulatedRenderer:
 def __init__(self,motion_file,device='cuda'):
  z=np.load(motion_file);self.lookup={(str(c),int(t)):i for i,(c,t) in enumerate(zip(z['camera'],z['frame']))};self.A=torch.tensor(z['linear'],device=device);self.b=torch.tensor(z['translation'],device=device);self.inverse=torch.linalg.inv(self.A)
  assert torch.isfinite(self.inverse).all()
 def render(self,p,camera,frame,W,K,width,height,degree):
  i=self.lookup[camera,frame];ids=p['anchor'];A=self.A[i,ids];b=self.b[i,ids];means=(A@p['means'][...,None])[...,0]+b
  cov,_=quat_scale_to_covar_preci(p['quats'],p['scales'].exp(),compute_covar=True,compute_preci=False,triu=False);cov=A@cov@A.transpose(-1,-2)
  center=-W[:3,:3].T@W[:3,3];dirs=(self.inverse[i,ids]@(means-center)[...,None])[...,0];dirs=torch.nn.functional.normalize(dirs,dim=-1)
  colors=(spherical_harmonics(degree,dirs,torch.cat([p['sh0'],p['shN']],1))+.5).clamp_min(0)
  return rasterization(means,None,None,p['opacities'].sigmoid(),colors,W[None],K[None],width,height,covars=cov,packed=False,near_plane=.1,far_plane=20,eps2d=.3)

def initial_parameters(path,device='cuda'):
 z=np.load(path);N=len(z['points']);t=lambda x:torch.tensor(x,dtype=torch.float32,device=device);param=lambda x:torch.nn.Parameter(t(x));C0=.28209479177387814
 return torch.nn.ParameterDict(dict(means=param(z['points']),scales=param(np.full((N,3),np.log(.009))),quats=param(np.tile([1,0,0,0],(N,1))),opacities=param(np.full(N,-2.1972246)),sh0=param((z['colors']-.5)/C0)[:,None,:],shN=param(np.zeros((N,8,3))),anchor=torch.nn.Parameter(torch.tensor(z['anchor'],device=device,dtype=torch.long),requires_grad=False)))
