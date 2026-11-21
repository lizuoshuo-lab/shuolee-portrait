"""Articulated Gaussian renderer with trainable bounded residual graph motion."""
from pathlib import Path
import math,numpy as np,torch
from gsplat.rendering import rasterization
from gsplat.cuda._wrapper import quat_scale_to_covar_preci,spherical_harmonics
from enerf_articulated_model import ArticulatedRenderer
from enerf_residual_graph import transform_gaussians,polynomial_transform,graph_rigidity

class GraphMotion(torch.nn.Module):
 def __init__(self,graph_path,scene_scale,device='cuda'):
  super().__init__();g=dict(np.load(graph_path));self.scene_scale=float(scene_scale)
  for key in ['node_points','node_anchor','edges','edge_weights']:
   dtype=torch.long if key in ['node_anchor','edges'] else torch.float32;self.register_buffer(key,torch.tensor(g[key],dtype=dtype,device=device))
  n=len(g['node_points']);self.rotation_raw=torch.nn.Parameter(torch.zeros(n,3,3,device=device));self.translation_raw=torch.nn.Parameter(torch.zeros(n,3,3,device=device))
 def forward(self,tau):return polynomial_transform(.1*self.rotation_raw.tanh(),(.03*self.scene_scale)*self.translation_raw.tanh(),tau)

def bind_graph(parameters,graph_path):
 g=dict(np.load(graph_path));device=parameters['means'].device
 assert len(parameters['means'])==len(g['point_node_indices'])
 parameters['node_indices']=torch.nn.Parameter(torch.tensor(g['point_node_indices'],device=device,dtype=torch.long),requires_grad=False)
 parameters['node_weights']=torch.nn.Parameter(torch.tensor(g['point_node_weights'],device=device,dtype=torch.float32),requires_grad=False)
 return parameters

class ResidualGraphRenderer:
 def __init__(self,motion_file,graph_motion):
  self.base=ArticulatedRenderer(motion_file);self.graph=graph_motion
  z=np.load(motion_file);self.times=torch.tensor(z['capture_time'],device=self.base.A.device,dtype=torch.float32);tau=(self.times-750)/6
  self.max_rotation_bound=float((tau.abs()+tau.square()+tau.abs().pow(3)).max()*.1*math.sqrt(3));assert self.max_rotation_bound<1.4
 def render(self,p,camera,frame,W,K,width,height,degree):
  i=self.base.lookup[camera,frame];anchors=p['anchor'];bodyA=self.base.A[i,anchors];bodyb=self.base.b[i,anchors];bodymeans=(bodyA@p['means'][...,None])[...,0]+bodyb
  cov,_=quat_scale_to_covar_preci(p['quats'],p['scales'].exp(),compute_covar=True,compute_preci=False,triu=False);bodycov=bodyA@cov@bodyA.transpose(-1,-2)
  nodeA=self.base.A[i,self.graph.node_anchor];nodeb=self.base.b[i,self.graph.node_anchor];nodes=(nodeA@self.graph.node_points[...,None])[...,0]+nodeb
  R,d=self.graph((self.times[i]-750)/6);weights=p['node_weights']/p['node_weights'].sum(-1,keepdim=True)
  means,cov,linear=transform_gaussians(bodymeans,bodycov,nodes,R,d,p['node_indices'],weights)
  center=-W[:3,:3].T@W[:3,3];directions=(self.base.inverse[i,anchors]@torch.linalg.solve(linear,(means-center)[...,None]))[...,0];directions=torch.nn.functional.normalize(directions,dim=-1)
  colors=(spherical_harmonics(degree,directions,torch.cat([p['sh0'],p['shN']],1))+.5).clamp_min(0)
  rgb,alpha,info=rasterization(means,None,None,p['opacities'].sigmoid(),colors,W[None],K[None],width,height,covars=cov,packed=False,near_plane=.1,far_plane=20,eps2d=.3)
  info['graph_rigidity']=graph_rigidity(nodes,R,d,self.graph.edges,self.graph.edge_weights)/(self.graph.scene_scale**2)
  info['graph_rotation_size']=R.sub(torch.eye(3,device=R.device)).square().mean()
  info['graph_translation_size']=(d/self.graph.scene_scale).square().mean()
  return rgb,alpha,info
