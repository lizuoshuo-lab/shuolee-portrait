"""Soft point-to-tangent constraint in the middle of bare forearms only."""
import numpy as np,torch
from scipy.spatial import cKDTree

class ForearmPrior:
 def __init__(self,path):
  z=np.load(path);self.vertices=z['vertices'];self.joints=z['joints'];f=z['faces'];self.arms=[]
  for p in range(2):
   v=self.vertices[p];tri=v[f];fn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n=np.zeros_like(v)
   for k in range(3):np.add.at(n,f[:,k],fn)
   n/=np.maximum(np.linalg.norm(n,axis=-1,keepdims=True),1e-8)
   for elbow,wrist in [(18,20),(19,21)]:
    a=self.joints[p,elbow];b=self.joints[p,wrist];axis=b-a;L=np.linalg.norm(axis);t=(v-a)@axis/L**2;rad=np.linalg.norm(v-a-t[:,None]*axis,axis=-1)
    mask=(t>.1)&(t<.9)&(rad<.4*L);assert mask.sum()>20
    self.arms.append(dict(a=a,b=b,L=L,v=v[mask],n=n[mask],tree=cKDTree(v[mask])))
  self.last_step=-1000;self.last_count=-1;self.cache=None
 def loss(self,means,opacities,step):
  if len(means)!=self.last_count or step-self.last_step>=100:
   x=means.detach().cpu().numpy();rs=[];ts=[]
   for a in self.arms:
    axis=a['b']-a['a'];t=(x-a['a'])@axis/a['L']**2;r=np.linalg.norm(x-a['a']-t[:,None]*axis,axis=-1)/a['L'];rs.append(r);ts.append(t)
   rs=np.stack(rs,1);ts=np.stack(ts,1);which=rs.argmin(1);r=rs[np.arange(len(x)),which];t=ts[np.arange(len(x)),which]
   # Middle forearm only; never supervise the elbow, wrist, hand or held object.
   selected=np.where((r<.4)&(t>.15)&(t<.8))[0];targets=[];normals=[];lengths=[];allidx=[]
   for ai,a in enumerate(self.arms):
    idx=selected[which[selected]==ai]
    if not len(idx):continue
    _,j=a['tree'].query(x[idx],workers=4);allidx.append(idx);targets.append(a['v'][j]);normals.append(a['n'][j]);lengths.append(np.full(len(idx),a['L']))
   assert allidx,'No Gaussian support around forearms'
   ten=lambda x:torch.as_tensor(np.concatenate(x),dtype=means.dtype,device=means.device)
   self.cache=(torch.as_tensor(np.concatenate(allidx),dtype=torch.long,device=means.device),ten(targets),ten(normals),ten(lengths))
   self.last_count=len(means);self.last_step=step
  ids,v,n,L=self.cache
  d=((means[ids]-v)*n).sum(-1).abs()/L
  penalty=(d-.04).clamp_min(0).square();w=opacities[ids].sigmoid().detach()
  return (penalty*w).sum()/w.sum().clamp_min(1),len(ids)
