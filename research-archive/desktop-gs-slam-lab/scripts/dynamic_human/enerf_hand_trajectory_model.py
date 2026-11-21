"""Shared-time residual translation: node interpolation or per-Gaussian coefficients.

No camera ID enters the residual field. These are local research controls,
not an implementation of an author method or a physical hand model.
"""
import numpy as np
import torch
from enerf_second_temporal_graph import cubic_bspline
from enerf_hand_joint_model import HandJointRenderer


class HandTrajectory(torch.nn.Module):
    def __init__(self,mode,support,scene_scale,device='cuda'):
        super().__init__();assert mode in ['nodes','points'];self.mode=mode;self.scene_scale=float(scene_scale)
        z=np.load(support);ids=torch.as_tensor(z['active_Gaussian_ids'],device=device,dtype=torch.long);self.register_buffer('ids',ids)
        idx=torch.as_tensor(z['point_node_indices'],device=device,dtype=torch.long);w=torch.as_tensor(z['point_node_weights'],device=device,dtype=torch.float32);w=w/w.sum(-1,keepdim=True)
        self.register_buffer('indices',idx);self.register_buffer('weights',w);self.register_buffer('centers',torch.linspace(-1.5,1.5,7,device=device))
        count=int(idx.max())+1 if mode=='nodes' else len(ids);assert count==32 if mode=='nodes' else count==6330
        self.coefficients=torch.nn.Parameter(torch.zeros(count,7,3,device=device))

    def basis(self,tau):
        return cubic_bspline((tau[...,None]-self.centers)/.5)-cubic_bspline(-self.centers/.5)

    def forward(self,tau):
        q=torch.einsum('...k,nkd->...nd',self.basis(tau),.03*self.scene_scale*self.coefficients.tanh())
        if self.mode=='nodes':q=(q[...,self.indices,:]*self.weights[...,None]).sum(-2)
        return q

    def regularization(self):
        # Physical time coordinate is nominal frame index; consecutive samples
        # are one frame apart. No conversion to seconds without capture FPS.
        tau=torch.arange(-6,7,device=self.coefficients.device,dtype=self.coefficients.dtype)/6
        delta=self(tau);acc=delta[2:]-2*delta[1:-1]+delta[:-2]
        return .1*(delta/self.scene_scale).square().mean()+10*(acc/self.scene_scale).square().mean()


class HandTrajectoryRenderer(HandJointRenderer):
    def __init__(self,motion_file,graph,active,trajectory):
        super().__init__(motion_file,graph,graph,active);self.trajectory=trajectory
        assert torch.equal(self.ids,trajectory.ids)

    def residual(self,c,t):
        i=self.base.lookup[c,t];return self.trajectory((self.times[i]-726)/6)

    def positions(self,p,c,t):
        means=super().positions(p,c,t);return means.index_add(0,self.ids,self.residual(c,t))

    def geometry(self,p,c,t):
        means,cov,linear,meta=super().geometry(p,c,t)
        return means.index_add(0,self.ids,self.residual(c,t)),cov,linear,meta
