"""Diagnostic only: selected material uses camera04/frame732 world snapshot in every view."""
import torch
from gsplat.rendering import rasterization
from gsplat.cuda._wrapper import spherical_harmonics
from enerf_hand_joint_model import HandJointRenderer

class HandSnapshotRenderer(HandJointRenderer):
    def geometry(self,p,c,t):
        means,cov,linear,meta=super().geometry(p,c,t)
        if c=='04' and t==732:return means,cov,linear,meta
        mr,cr,lr,_=super().geometry(p,'04',732)
        ids=self.ids
        return means.index_copy(0,ids,mr[ids]),cov.index_copy(0,ids,cr[ids]),linear.index_copy(0,ids,lr[ids]),meta

    def render(self,p,camera,frame,W,K,width,height,degree):
        means,cov,linear,(i,nodes,R,d)=self.geometry(p,camera,frame)
        ir=self.base.lookup['04',732];inverse=self.base.inverse[i,p['anchor']].clone();inverse[self.ids]=self.base.inverse[ir,p['anchor'][self.ids]]
        center=-W[:3,:3].T@W[:3,3];directions=(inverse@torch.linalg.solve(linear,(means-center)[...,None]))[...,0];directions=torch.nn.functional.normalize(directions,dim=-1)
        colors=(spherical_harmonics(degree,directions,torch.cat([p['sh0'],p['shN']],1))+.5).clamp_min(0)
        return rasterization(means,None,None,p['opacities'].sigmoid(),colors,W[None],K[None],width,height,covars=cov,packed=False,near_plane=.1,far_plane=20,eps2d=.3)
