"""Free reference-space Gaussians with a body-derived deformation field.

Local adapter inspired by canonical-point/LBS avatar representations. This is
not an official GauHuman/GART reproduction. No point is constrained to a naked
mesh triangle; learned XYZ is in reference-world units. Covariance uses the
same affine deformation as positions. Anchors are motion support, not targets.
"""
import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree
from .gaussian_model import GaussianModel
from utils.general_utils import build_rotation,strip_symmetric


class ClothingGaussianModel(GaussianModel):
    def __init__(self,sh_degree=3):
        super().__init__(sh_degree)
        self.anchor_ids=None;self.anchor_weights=None;self.reference_frame=0;self._cached_deformation=None
        self.clothing_control=None;self.control_ids=None;self.control_weights=None

    def load_motion(self,path):
        z=np.load(path)
        self.vertices_sequence=torch.as_tensor(z['vertices_world'],dtype=torch.float32,device='cuda')
        self.affine_sequence=torch.as_tensor(z['vertex_affine'],dtype=torch.float32,device='cuda')
        self.reference_frame=int(z['reference_frame']);self.reference_vertices=self.vertices_sequence[self.reference_frame]
        self.world_scale=float(z['world_scale']);self.tree=cKDTree(self.reference_vertices.cpu().numpy())
        self.num_timesteps=len(self.vertices_sequence);self.timestep=self.reference_frame
        eye=torch.eye(3,device='cuda')
        assert (self.affine_sequence[self.reference_frame]-eye).abs().max()<2e-5
        from pathlib import Path
        controls=Path(path).with_name('controls.npz')
        if controls.exists():
            q=np.load(controls)
            assert int(q['reference_frame'])==self.reference_frame
            assert len(q['delta_world'])==self.num_timesteps
            assert np.abs(q['delta_world'][self.reference_frame]).max()<1e-6
            self.control_nodes=q['nodes_world'];self.control_support=q['support_world']
            self.clothing_control=torch.as_tensor(q['delta_world'],dtype=torch.float32,device='cuda')
            self.control_ids=None;self.control_weights=None

    @torch.no_grad()
    def refresh_anchors(self):
        dist,ids=self.tree.query(self._xyz.detach().cpu().numpy(),k=4)
        # Fixed weights until the next explicit refresh give an affine field
        # with the exact covariance Jacobian used below.
        w=1/(dist**2+(.015*self.world_scale)**2);w/=w.sum(1,keepdims=True)
        self.anchor_ids=torch.as_tensor(ids,dtype=torch.long,device='cuda')
        self.anchor_weights=torch.as_tensor(w,dtype=torch.float32,device='cuda')
        if self.clothing_control is not None:
            from .clothing_motion_support import support_weights
            index,weight=support_weights(self._xyz.detach().cpu().numpy(),self.control_nodes,self.control_support,self.world_scale)
            self.control_ids=torch.as_tensor(index,dtype=torch.long,device='cuda')
            self.control_weights=torch.as_tensor(weight,dtype=torch.float32,device='cuda')
        self._cached_deformation=None

    def select_mesh_by_timestep(self,timestep,original=False):
        self.timestep=int(timestep)
        self._cached_deformation=None

    def deformation(self):
        if self.anchor_ids is None or len(self.anchor_ids)!=len(self._xyz):self.refresh_anchors()
        if self._cached_deformation is not None:return self._cached_deformation
        ids,w=self.anchor_ids,self.anchor_weights
        A=self.affine_sequence[self.timestep][ids]
        v=self.vertices_sequence[self.timestep][ids];ref=self.reference_vertices[ids]
        linear=(A*w[:,:,None,None]).sum(1)
        translation=((v-torch.einsum('nkij,nkj->nki',A,ref))*w[:,:,None]).sum(1)
        if self.clothing_control is not None:
            translation=translation+(self.clothing_control[self.timestep][self.control_ids]*self.control_weights[:,:,None]).sum(1)
        self._cached_deformation=(linear,translation)
        return self._cached_deformation

    @property
    def get_xyz(self):
        A,t=self.deformation();return torch.einsum('nij,nj->ni',A,self._xyz)+t

    def get_covariance(self,scaling_modifier=1):
        A,_=self.deformation();rot=build_rotation(self._rotation)
        L=(A@rot)*(self.get_scaling*scaling_modifier)[:,None,:]
        return strip_symmetric(L@L.transpose(1,2))

    @torch.no_grad()
    def export_world_ply(self,path,timestep=0):
        self.select_mesh_by_timestep(timestep);A,_=self.deformation()
        L=(A@build_rotation(self._rotation))*self.get_scaling[:,None,:];cov=L@L.transpose(1,2)
        values,rotation=torch.linalg.eigh(cov)
        rotation[:,:,2]*=torch.linalg.det(rotation)[:,None]
        from pytorch3d.transforms import matrix_to_quaternion
        output=GaussianModel(self.max_sh_degree)
        output._xyz=self.get_xyz.detach();output._features_dc=self._features_dc.detach();output._features_rest=self._features_rest.detach()
        output._opacity=self._opacity.detach();output._scaling=values.clamp_min(1e-14).sqrt().log();output._rotation=matrix_to_quaternion(rotation)
        output.save_ply(str(path))

    def densify_and_prune(self,*args,**kwargs):
        # Parent is in unbound mode: no hidden minimum-one-point-per-face rule.
        assert self.binding is None
        super().densify_and_prune(*args,**kwargs)
        self.refresh_anchors()
