"""Jointly depth-sorted per-actor contributions for weak instance supervision."""
import torch
import enerf_graph_renderer as base_module
from enerf_graph_renderer import ResidualGraphRenderer

class ActorSupervisedRenderer(ResidualGraphRenderer):
 def render(self,p,camera,frame,W,K,width,height,degree):
  raster=base_module.rasterization
  def with_actor_features(*args,**kwargs):
   membership=torch.nn.functional.one_hot(p['anchor']//6890,2).to(args[4])
   result,alpha,info=raster(*args[:4],torch.cat([args[4],membership],-1),*args[5:],**kwargs)
   info['actor_contributions']=result[...,3:5]
   return result[...,:3],alpha,info
  # Single-threaded render call; restoration also occurs if rasterization fails.
  base_module.rasterization=with_actor_features
  try:return super().render(p,camera,frame,W,K,width,height,degree)
  finally:base_module.rasterization=raster

def ownership_loss(contributions,alpha,labels):
 # Normalize joint contributions by accumulated alpha; this is not a new matte loss.
 probability=(contributions[0]+1e-6)/(alpha[0]+2e-6)
 values=[]
 for person in [0,1]:
  known=labels==person
  if known.any():values.append(-torch.log(probability[...,person][known].clamp_min(1e-6)).mean())
 return torch.stack(values).mean() if values else contributions.sum()*0
