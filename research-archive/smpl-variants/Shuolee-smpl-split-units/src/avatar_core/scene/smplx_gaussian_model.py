"""Local SMPL-X adapter for the incomplete public GaussianAvatars_SMPLX repo.

Uses the published mesh-bound GaussianModel. The author did not publish this
module; it is not represented as the missing author's implementation. Meshes
come exclusively from this experiment's fresh neutral-template fitting.
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz
from .gaussian_model import GaussianModel
from utils.graphics_utils import compute_face_orientation
from utils.general_utils import build_rotation


class SMPLXGaussianModel(GaussianModel):
    def __init__(self, sh_degree=3):
        super().__init__(sh_degree)
        self.vertices_sequence = None
        self.faces = None

    def load_fresh_meshes(self, path):
        data = np.load(Path(path), allow_pickle=False)
        self.vertices_sequence = torch.as_tensor(data['vertices_world'], dtype=torch.float32, device='cuda')
        self.faces = torch.as_tensor(data['faces'], dtype=torch.long, device='cuda')
        self.num_timesteps = len(self.vertices_sequence)
        assert self.vertices_sequence.shape[1:] == (10475, 3)
        assert self.faces.shape == (20908, 3)
        assert torch.isfinite(self.vertices_sequence).all()
        self.select_mesh_by_timestep(0)

    def select_mesh_by_timestep(self, timestep, original=False):
        self.timestep = int(timestep)
        verts = self.vertices_sequence[self.timestep]
        self.face_center = verts[self.faces].mean(1)
        self.face_orien_mat, scale = compute_face_orientation(verts, self.faces, return_scale=True)
        self.face_scaling = scale.clamp_min(1.e-8)
        self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))

    @torch.no_grad()
    def initialize_surface(self, count=200000, seed=6022):
        """Cover every triangle, then allocate the remaining points by area."""
        assert count >= len(self.faces)
        gen = torch.Generator(device='cuda').manual_seed(seed)
        triangle = self.vertices_sequence[0][self.faces]
        area = .5 * torch.cross(triangle[:,1]-triangle[:,0], triangle[:,2]-triangle[:,0], dim=-1).norm(dim=-1)
        base = torch.arange(len(self.faces), device='cuda')
        extra = torch.multinomial(area, count-len(base), replacement=True, generator=gen)
        self.binding = torch.cat((base, extra))
        self.binding_counter = torch.bincount(self.binding, minlength=len(self.faces)).to(torch.int32)
        self.create_from_pcd(None, spatial_lr_scale=1.0)
        uv = torch.rand((count, 2), device='cuda', generator=gen)
        su = uv[:,0].sqrt()
        bary = torch.stack((1-su, su*(1-uv[:,1]), su*uv[:,1]), dim=-1)
        pos = (triangle[self.binding]*bary[:,:,None]).sum(1)
        delta = (pos-self.face_center[self.binding])/self.face_scaling[self.binding]
        local = torch.bmm(self.face_orien_mat[self.binding].transpose(1,2), delta[...,None]).squeeze(-1)
        self._xyz = nn.Parameter(local)
        # Tangential width follows per-face sampling density, normal width is
        # smaller. Local axis 1 is the normal in public compute_face_orientation.
        tangent = .75*torch.sqrt(area[self.binding]/self.binding_counter[self.binding])
        relative = tangent/self.face_scaling[self.binding,0]
        scale = torch.stack((relative, relative*.35, relative), dim=-1).clamp_min(1.e-5)
        self._scaling = nn.Parameter(scale.log())
        self._features_dc = nn.Parameter(torch.zeros_like(self._features_dc))
        self._features_rest = nn.Parameter(torch.zeros_like(self._features_rest))
        assert int(self.binding_counter.min()) >= 1
        assert torch.allclose(self.get_xyz, pos, atol=2.e-5, rtol=1.e-5)
        return {'initial_points': count, 'covered_triangles': int((self.binding_counter>0).sum()),
                'triangles': len(self.faces), 'color_initialization': 'neutral gray; SH coefficients zero',
                'position_initialization': 'fresh area-weighted sampling with at least one point per triangle',
                'spatial_lr_scale': 1.0, 'max_reconstruction_error': float((self.get_xyz-pos).abs().max())}

    def get_covariance(self, scaling_modifier=1):
        # Use the same world rotation as the normal CUDA-rendering branch.
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """Public split procedure with offsets consistently in face-local units."""
        n = len(self._xyz)
        padded = torch.zeros(n, device='cuda'); padded[:len(grads)] = grads.reshape(-1)
        chosen = (padded >= grad_threshold) & (self.get_scaling.max(1).values > self.percent_dense*scene_extent)
        local_std = self._scaling[chosen].exp().repeat(N,1)
        sample = torch.normal(torch.zeros_like(local_std), local_std)
        rotation = build_rotation(self._rotation[chosen]).repeat(N,1,1)
        xyz = torch.bmm(rotation, sample[...,None]).squeeze(-1)+self._xyz[chosen].repeat(N,1)
        scaling = (local_std/(.8*N)).log()
        binding = self.binding[chosen].repeat(N)
        self.binding = torch.cat((self.binding, binding))
        self.binding_counter.scatter_add_(0,binding,torch.ones_like(binding,dtype=torch.int32))
        self.densification_postfix(xyz,self._features_dc[chosen].repeat(N,1,1),
            self._features_rest[chosen].repeat(N,1,1),self._opacity[chosen].repeat(N,1),
            scaling,self._rotation[chosen].repeat(N,1))
        prune = torch.cat((chosen,torch.zeros(N*int(chosen.sum()),device='cuda',dtype=torch.bool)))
        self.prune_points(prune)

    @torch.no_grad()
    def export_world_ply(self, path, timestep=0):
        """Standard 3DGS snapshot; bound local coordinates are not a world PLY."""
        self.select_mesh_by_timestep(timestep)
        world = GaussianModel(self.max_sh_degree)
        world._xyz = self.get_xyz.detach()
        world._features_dc = self._features_dc.detach()
        world._features_rest = self._features_rest.detach()
        world._scaling = self.get_scaling.detach().log()
        world._rotation = self.get_rotation.detach()
        world._opacity = self._opacity.detach()
        world.save_ply(str(path))
