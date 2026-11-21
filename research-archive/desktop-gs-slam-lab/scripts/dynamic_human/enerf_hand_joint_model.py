"""Local joint control: existing motion graph updates affect only selected hand Gaussians."""
import torch
from gsplat.rendering import rasterization
from gsplat.cuda._wrapper import quat_scale_to_covar_preci,spherical_harmonics
from enerf_second_temporal_graph import ResidualGraphRenderer
from enerf_residual_graph import transform_gaussians,graph_rigidity
from enerf_hand_flow_model import project,sample

class HandJointRenderer(ResidualGraphRenderer):
    def __init__(self,motion_file,graph,reference_graph,active):
        super().__init__(motion_file,graph);self.reference_graph=reference_graph;self.active=active;self.ids=torch.where(active)[0]
        assert all(not x.requires_grad for x in reference_graph.parameters())

    def common(self,p,c,t):
        i=self.base.lookup[c,t];A=self.base.A[i,p['anchor']];b=self.base.b[i,p['anchor']];body=(A@p['means'][...,None])[...,0]+b
        nodes=(self.base.A[i,self.graph.node_anchor]@self.graph.node_points[...,None])[...,0]+self.base.b[i,self.graph.node_anchor]
        tau=(self.times[i]-726)/6;R,d=self.graph(tau);R0,d0=self.reference_graph(tau);weights=p['node_weights']/p['node_weights'].sum(-1,keepdim=True)
        return i,A,body,nodes,R,d,R0,d0,weights

    @staticmethod
    def blend(body,nodes,R,d,idx,weights):
        g=nodes[idx];moved=(R[idx]@(body[:,None]-g)[...,None])[...,0]+g+d[idx]
        return (moved*weights[...,None]).sum(1)

    def positions(self,p,c,t):
        i,A,body,nodes,R,d,R0,d0,w=self.common(p,c,t);means=self.blend(body,nodes,R0,d0,p['node_indices'],w);ids=self.ids
        changed=self.blend(body[ids],nodes,R,d,p['node_indices'][ids],w[ids])
        return means.index_copy(0,ids,changed)

    def geometry(self,p,c,t):
        i,A,body,nodes,R,d,R0,d0,w=self.common(p,c,t)
        C,_=quat_scale_to_covar_preci(p['quats'],p['scales'].exp(),compute_covar=True,compute_preci=False,triu=False);C=A@C@A.transpose(-1,-2)
        means,cov,linear=transform_gaussians(body,C,nodes,R0,d0,p['node_indices'],w);ids=self.ids
        m1,c1,l1=transform_gaussians(body[ids],C[ids],nodes,R,d,p['node_indices'][ids],w[ids])
        return means.index_copy(0,ids,m1),cov.index_copy(0,ids,c1),linear.index_copy(0,ids,l1),(i,nodes,R,d)

    def render(self,p,camera,frame,W,K,width,height,degree):
        means,cov,linear,(i,nodes,R,d)=self.geometry(p,camera,frame)
        center=-W[:3,:3].T@W[:3,3];directions=(self.base.inverse[i,p['anchor']]@torch.linalg.solve(linear,(means-center)[...,None]))[...,0];directions=torch.nn.functional.normalize(directions,dim=-1)
        colors=(spherical_harmonics(degree,directions,torch.cat([p['sh0'],p['shN']],1))+.5).clamp_min(0)
        rgb,alpha,info=rasterization(means,None,None,p['opacities'].sigmoid(),colors,W[None],K[None],width,height,covars=cov,packed=False,near_plane=.1,far_plane=20,eps2d=.3)
        info['graph_rigidity']=graph_rigidity(nodes,R,d,self.graph.edges,self.graph.edge_weights)/(self.graph.scene_scale**2);info['graph_rotation_size']=R.sub(torch.eye(3,device=R.device)).square().mean();info['graph_translation_size']=(d/self.graph.scene_scale).square().mean()
        return rgb,alpha,info

    def current_normal_flow(self,p,pair,source):
        c,t,t1=pair['camera'],pair['source_frame'],pair['target_frame'];W,K=source['W'],source['K']
        m0=self.positions(p,c,t);m1=self.positions(p,c,t1);uv0,_=project(m0,W,K);uv1,_=project(m1,W,K)
        with torch.no_grad():means,cov,_,_=self.geometry(p,c,t)
        values,alpha,_=rasterization(means,None,None,p['opacities'].sigmoid().detach(),uv1-uv0,W[None],K[None],160,160,covars=cov,packed=False,near_plane=.1,far_plane=20,eps2d=.3)
        a=sample(alpha[0],pair['xy']);flow=sample(values[0],pair['xy'])/a.clamp_min(1e-6)
        return (flow*pair['normal']).sum(-1),flow,a
