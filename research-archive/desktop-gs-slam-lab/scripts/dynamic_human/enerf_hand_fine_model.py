"""Local research residual graph:32 advected hand controls after the original motion."""
import numpy as np,torch
from enerf_second_temporal_graph import GraphMotion
from enerf_hand_joint_model import HandJointRenderer
from enerf_residual_graph import graph_rigidity

class HandFineGraph(GraphMotion):
    def __init__(self,path,scene_scale,device='cuda'):
        super().__init__(path,scene_scale,device)
        z=np.load(path)
        for key in ['point_node_indices','point_node_weights','node_coarse_indices','node_coarse_weights','active_Gaussian_ids','parent_Gaussian_ids']:
            dtype=torch.long if ('indices' in key or 'ids' in key) else torch.float32
            self.register_buffer(key,torch.as_tensor(z[key],dtype=dtype,device=device))
        self.point_node_weights=self.point_node_weights/self.point_node_weights.sum(-1,keepdim=True)
        self.node_coarse_weights=self.node_coarse_weights/self.node_coarse_weights.sum(-1,keepdim=True)

class HandFineRenderer(HandJointRenderer):
    def __init__(self,motion_file,graph,reference_graph,active,fine):
        super().__init__(motion_file,graph,reference_graph,active);self.fine=fine
        assert torch.equal(self.ids,fine.active_Gaussian_ids)
    def fine_state(self,c,t):
        i=self.base.lookup[c,t];f=self.fine
        body=(self.base.A[i,f.node_anchor]@f.node_points[...,None])[...,0]+self.base.b[i,f.node_anchor]
        coarse=(self.base.A[i,self.graph.node_anchor]@self.graph.node_points[...,None])[...,0]+self.base.b[i,self.graph.node_anchor]
        tau=(self.times[i]-726)/6;R0,d0=self.reference_graph(tau)
        centers=self.blend(body,coarse,R0,d0,f.node_coarse_indices,f.node_coarse_weights)
        R,d=f(tau);return centers,R,d
    def residual(self,x,centers,R,d):
        f=self.fine;idx=f.point_node_indices;w=f.point_node_weights;eye=torch.eye(3,device=x.device,dtype=x.dtype)
        delta=R[idx]-eye
        displacement=(delta@(x[:,None]-centers[idx])[...,None])[...,0]+d[idx]
        new=x+(w[...,None]*displacement).sum(1)
        linear=eye+(w[...,None,None]*delta).sum(1)
        return new,linear
    def positions(self,p,c,t):
        means=super().positions(p,c,t);centers,R,d=self.fine_state(c,t)
        x,_=self.residual(means[self.ids],centers,R,d)
        return means.index_copy(0,self.ids,x)
    def geometry(self,p,c,t):
        means,cov,linear,meta=super().geometry(p,c,t);centers,R,d=self.fine_state(c,t)
        x,B=self.residual(means[self.ids],centers,R,d)
        C=B@cov[self.ids]@B.transpose(-1,-2);total=B@linear[self.ids]
        return means.index_copy(0,self.ids,x),cov.index_copy(0,self.ids,C),linear.index_copy(0,self.ids,total),meta
    def render(self,p,camera,frame,W,K,width,height,degree):
        rgb,alpha,info=super().render(p,camera,frame,W,K,width,height,degree)
        centers,R,d=self.fine_state(camera,frame);f=self.fine
        info['fine_graph_rigidity']=graph_rigidity(centers,R,d,f.edges,f.edge_weights)/(f.scene_scale**2)
        info['fine_graph_rotation_size']=(R-torch.eye(3,device=R.device)).square().mean()
        info['fine_graph_translation_size']=(d/f.scene_scale).square().mean()
        return rgb,alpha,info

