"""Component deformation graph, exact blended covariance, canonical SH color.

Local research baseline, not an official implementation of 4K4D or OmniRe.
"""
import torch
from gsplat.rendering import rasterization
from gsplat.cuda._wrapper import quat_scale_to_covar_preci,spherical_harmonics
from enerf_temporal_graph import GraphMotion as BaseGraph
from enerf_residual_graph import rotation_matrix,transform_gaussians,graph_rigidity

class ActorGraph(BaseGraph):
    def forward(self,tau):
        poly=torch.stack([tau,tau*tau,tau*tau*tau]);local=self.basis(tau)
        rv=torch.einsum('k,nkd->nd',poly,.15*self.rotation_raw.tanh())+torch.einsum('k,nkd->nd',local,.05*self.local_rotation_raw.tanh())
        tr=torch.einsum('k,nkd->nd',poly,.08*self.scene_scale*self.translation_raw.tanh())+torch.einsum('k,nkd->nd',local,.02*self.scene_scale*self.local_translation_raw.tanh())
        return rotation_matrix(rv),tr

def geometry(p,g,time_index):
    tau=torch.as_tensor((float(time_index)-612)/12,device=p['means'].device,dtype=p['means'].dtype)
    R,d=g(tau);cov,_=quat_scale_to_covar_preci(p['quats'],p['scales'].exp(),compute_covar=True,compute_preci=False,triu=False)
    weights=p['node_weights']/p['node_weights'].sum(-1,keepdim=True)
    means,cov,linear=transform_gaussians(p['means'],cov,g.node_points,R,d,p['node_indices'],weights)
    return means,cov,linear,R,d

def render(p,g,time_index,W,K,width=1920,height=1080,degree=2):
    means,cov,linear,R,d=geometry(p,g,time_index)
    center=-W[:3,:3].T@W[:3,3];directions=torch.linalg.solve(linear,(means-center)[...,None])[...,0];directions=torch.nn.functional.normalize(directions,dim=-1)
    colors=(spherical_harmonics(degree,directions,torch.cat([p['sh0'],p['shN']],1))+.5).clamp_min(0)
    rgb,alpha,info=rasterization(means,None,None,p['opacities'].sigmoid(),colors,W[None],K[None],width,height,covars=cov,packed=False,near_plane=.1,far_plane=20,eps2d=.3)
    info['rigidity']=graph_rigidity(g.node_points,R,d,g.edges,g.edge_weights)/(g.scene_scale**2)
    info['rotation_size']=(R-torch.eye(3,device=R.device)).square().mean()
    info['translation_size']=(d/g.scene_scale).square().mean()
    return rgb,alpha,info
