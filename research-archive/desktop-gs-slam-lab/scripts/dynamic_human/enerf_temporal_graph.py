"""Local research control: cubic motion plus compact cubic B-spline residuals.

The reference time stays exactly identity. Inputs, spatial graph and body
motion are unchanged; this is not an implementation of an author method.
"""
import math
import torch
from enerf_graph_renderer import GraphMotion as CubicGraphMotion
from enerf_graph_renderer import ResidualGraphRenderer as CubicRenderer, bind_graph
from enerf_residual_graph import rotation_matrix


def cubic_bspline(x):
    a=x.abs()
    return torch.where(a<1,2/3-x.square()+0.5*a.pow(3),
                       torch.where(a<2,(2-a).pow(3)/6,torch.zeros_like(a)))


class GraphMotion(CubicGraphMotion):
    local_rotation_scale=0.03
    local_translation_scale=0.01
    local_width=0.5

    def __init__(self,graph_path,scene_scale,device='cuda'):
        super().__init__(graph_path,scene_scale,device)
        self.register_buffer('local_centers',torch.linspace(-1.5,1.5,7,device=device))
        n=len(self.node_points)
        self.local_rotation_raw=torch.nn.Parameter(torch.zeros(n,7,3,device=device))
        self.local_translation_raw=torch.nn.Parameter(torch.zeros(n,7,3,device=device))

    def basis(self,tau):
        return (cubic_bspline((tau-self.local_centers)/self.local_width)
                -cubic_bspline(-self.local_centers/self.local_width))

    def forward(self,tau):
        polynomial=torch.stack([tau,tau*tau,tau*tau*tau])
        rv=torch.einsum('k,nkd->nd',polynomial,0.1*self.rotation_raw.tanh())
        tr=torch.einsum('k,nkd->nd',polynomial,0.03*self.scene_scale*self.translation_raw.tanh())
        local=self.basis(tau)
        rv=rv+torch.einsum('k,nkd->nd',local,self.local_rotation_scale*self.local_rotation_raw.tanh())
        tr=tr+torch.einsum('k,nkd->nd',local,self.local_translation_scale*self.scene_scale*self.local_translation_raw.tanh())
        return rotation_matrix(rv),tr


class ResidualGraphRenderer(CubicRenderer):
    def __init__(self,motion_file,graph_motion):
        super().__init__(motion_file,graph_motion)
        # Each nonnegative cardinal B-spline family sums to at most1;
        # subtracting its reference value gives an L1 norm at most2.
        self.max_rotation_bound+=2*graph_motion.local_rotation_scale*math.sqrt(3)
        assert self.max_rotation_bound<math.pi/2
