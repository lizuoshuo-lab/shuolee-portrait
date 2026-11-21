"""Free reference-space Gaussians with a body-derived deformation field.

The explicit triangle mode inherits motion support across densification.
Learned centres and standard deviations are in reference-world units, and the
posed covariance follows the same similarity as the centre. Legacy 4-NN
anchors continue to provide affine motion for existing clothing bundles.
"""
import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree
from .gaussian_model import GaussianModel
from utils.general_utils import build_rotation,strip_symmetric
from utils.graphics_utils import compute_face_orientation


class ClothingGaussianModel(GaussianModel):
    def __init__(self,sh_degree=3):
        super().__init__(sh_degree)
        self.anchor_ids=None;self.anchor_weights=None;self.reference_frame=0;self._cached_deformation=None
        self.clothing_control=None;self.control_ids=None;self.control_weights=None
        self.support_face_ids = None
        self._cached_triangle_pose = None

    @torch.no_grad()
    def configure_triangle_support(self, vertices_sequence, faces, reference_frame, support_face_ids,
                                   *, stable_ids=None, next_stable_id=None):
        """Freeze one inherited motion-support face per free reference-world row."""
        from roma import quat_xyzw_to_wxyz, rotmat_to_unitquat

        if self.binding is not None or self._xyz.ndim != 2 or self._xyz.shape[1] != 3 or not len(self._xyz):
            raise ValueError('Triangle motion requires initialized, unbound reference-world Gaussian rows.')
        device, dtype = self._xyz.device, self._xyz.dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError('Triangle motion requires float32 or float64 world geometry.')

        def index_tensor(value):
            result = torch.as_tensor(value, device=device)
            if result.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                raise ValueError('Motion support and stable identities require integer indices.')
            return result.to(dtype=torch.long, copy=True)

        vertices = torch.as_tensor(vertices_sequence, device=device, dtype=dtype).detach().clone()
        topology, owners = index_tensor(faces), index_tensor(support_face_ids)
        if (vertices.ndim != 3 or vertices.shape[2] != 3 or not torch.isfinite(vertices).all()
                or topology.ndim != 2 or topology.shape[1] != 3 or not len(topology)
                or topology.min() < 0 or topology.max() >= vertices.shape[1]
                or not 0 <= int(reference_frame) < len(vertices) or int(reference_frame) != reference_frame
                or owners.shape != (len(self._xyz),) or owners.min() < 0 or owners.max() >= len(topology)):
            raise ValueError('Invalid posed body triangles, reference frame, or explicit row support.')
        reference = int(reference_frame)
        if stable_ids is None:
            if hasattr(self, 'densification_ids'):
                stable_ids = self.densification_ids
                if next_stable_id is None:
                    next_stable_id = self.next_densification_id
            else:
                stable_ids = torch.arange(len(owners), device=device)
        ids = index_tensor(stable_ids)
        if ids.shape != owners.shape or ids.min() < 0 or not (ids[1:] > ids[:-1]).all():
            raise ValueError('Stable Gaussian IDs must be nonnegative, aligned and strictly increasing.')
        next_id = int(ids[-1])+1 if next_stable_id is None else int(next_stable_id)
        if next_id <= int(ids[-1]) or (next_stable_id is not None and next_id != next_stable_id):
            raise ValueError('The next stable Gaussian ID must exceed every current ID.')

        # Use the bound renderer's exact frame and length rule. This preserves
        # its initial posed centres and covariance even when the body triangles
        # change shape, before the free garment optimizer makes any updates.
        supported = owners.unique(sorted=True)
        supported_faces = topology[supported]
        triangles = vertices[:, supported_faces]
        edge, other = triangles[:, :, 1]-triangles[:, :, 0], triangles[:, :, 2]-triangles[:, :, 0]
        cross = torch.linalg.cross(edge, other, dim=-1)
        edge_length, twice_area = edge.norm(dim=-1), cross.norm(dim=-1)
        if not ((edge_length > 0) & (twice_area > 0) & torch.isfinite(twice_area)).all():
            raise ValueError('A triangle used for motion support is degenerate in a frozen pose.')
        basis, lengths = compute_face_orientation(vertices, supported_faces, return_scale=True)
        basis = basis.double()
        rotations = basis @ basis[reference].transpose(-1, -2)
        rotations[reference] = torch.eye(3, device=device, dtype=rotations.dtype)
        lengths = lengths.squeeze(-1).clamp_min(1e-8)
        scales = lengths/lengths[reference]
        if not (torch.isfinite(scales) & (scales > 0)).all():
            raise ValueError('The triangle motion scale is invalid in the Gaussian dtype.')
        lookup = torch.full((len(topology),), -1, device=device, dtype=torch.long)
        lookup[supported] = torch.arange(len(supported), device=device)

        self.vertices_sequence, self.faces = vertices, topology
        self.reference_frame, self.num_timesteps = reference, len(vertices)
        self.reference_vertices = vertices[reference]
        self.support_face_ids, self.densification_ids, self.next_densification_id = owners, ids, next_id
        self._triangle_face_lookup = lookup
        self._triangle_centroids = triangles.mean(dim=2)
        self._triangle_rotations = rotations.to(dtype=dtype)
        self._triangle_quaternions = quat_xyzw_to_wxyz(rotmat_to_unitquat(rotations)).to(dtype=dtype)
        self._triangle_scales = scales
        self.select_mesh_by_timestep(reference)

    def _triangle_pose(self):
        if self.support_face_ids.shape != (len(self._xyz),):
            raise RuntimeError('Free-clothing rows and inherited motion support are misaligned.')
        if self._cached_triangle_pose is None:
            indices = self._triangle_face_lookup[self.support_face_ids]
            frame = self.timestep
            self._cached_triangle_pose = (
                self._triangle_centroids[self.reference_frame, indices],
                self._triangle_centroids[frame, indices],
                self._triangle_rotations[frame, indices],
                self._triangle_quaternions[frame, indices],
                self._triangle_scales[frame, indices],
            )
        return self._cached_triangle_pose

    def load_motion(self,path):
        self.support_face_ids = None
        self._cached_triangle_pose = None
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
        if self.support_face_ids is not None:
            raise RuntimeError('Explicit triangle motion support is inherited, so nearest rebind is unavailable.')
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
        if self.support_face_ids is not None and (int(timestep) != timestep or not 0 <= int(timestep) < self.num_timesteps):
            raise ValueError('The requested posed triangle frame is outside the frozen sequence.')
        self.timestep=int(timestep)
        self._cached_deformation=None
        self._cached_triangle_pose = None

    def deformation(self):
        if self.support_face_ids is not None:
            reference, posed, rotation, _, scale = self._triangle_pose()
            linear = rotation*scale[:, None, None]
            translation = posed-torch.einsum('nij,nj->ni', linear, reference)
            return linear, translation
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
        if self.support_face_ids is not None:
            if self.timestep == self.reference_frame:
                return self._xyz
            reference, posed, rotation, _, scale = self._triangle_pose()
            return scale[:, None]*torch.einsum('nij,nj->ni', rotation, self._xyz-reference)+posed
        A,t=self.deformation();return torch.einsum('nij,nj->ni',A,self._xyz)+t

    @property
    def get_scaling(self):
        scaling = super().get_scaling
        if self.support_face_ids is not None and self.timestep != self.reference_frame:
            scaling = scaling*self._triangle_pose()[-1][:, None]
        return scaling

    @property
    def get_rotation(self):
        rotation = super().get_rotation
        if self.support_face_ids is not None and self.timestep != self.reference_frame:
            from shuolee_portrait.free_clothing_torch import quaternion_product

            rotation = self.rotation_activation(quaternion_product(self._triangle_pose()[-2], rotation))
        return rotation

    def get_covariance(self,scaling_modifier=1):
        if self.support_face_ids is not None:
            from shuolee_portrait.free_clothing_torch import normalized_quaternion_matrix

            rotation = normalized_quaternion_matrix(self.get_rotation)
            axes = rotation*(self.get_scaling*scaling_modifier)[:, None, :]
            covariance = axes @ axes.transpose(-1, -2)
            return covariance[:, (0, 0, 0, 1, 1, 2), (0, 1, 2, 1, 2, 2)]
        A,_=self.deformation();rot=build_rotation(self._rotation)
        L=(A@rot)*(self.get_scaling*scaling_modifier)[:,None,:]
        return strip_symmetric(L@L.transpose(1,2))

    @torch.no_grad()
    def export_world_ply(self,path,timestep=0):
        if self.support_face_ids is not None:
            previous = self.timestep
            self.select_mesh_by_timestep(timestep)
            try:
                output = GaussianModel(self.max_sh_degree)
                output._xyz, output._rotation = self.get_xyz, self.get_rotation
                output._scaling = self.get_scaling.log()
                output._features_dc, output._features_rest = self._features_dc, self._features_rest
                output._opacity = self._opacity
                output.save_ply(str(path))
            finally:
                self.select_mesh_by_timestep(previous)
            return
        self.select_mesh_by_timestep(timestep);A,_=self.deformation()
        L=(A@build_rotation(self._rotation))*self.get_scaling[:,None,:];cov=L@L.transpose(1,2)
        values,rotation=torch.linalg.eigh(cov)
        rotation[:,:,2]*=torch.linalg.det(rotation)[:,None]
        from pytorch3d.transforms import matrix_to_quaternion
        output=GaussianModel(self.max_sh_degree)
        output._xyz=self.get_xyz.detach();output._features_dc=self._features_dc.detach();output._features_rest=self._features_rest.detach()
        output._opacity=self._opacity.detach();output._scaling=values.clamp_min(1e-14).sqrt().log();output._rotation=matrix_to_quaternion(rotation)
        output.save_ply(str(path))

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities,
                              new_scaling, new_rotation, *, constraint_parents=None):
        previous = self.support_face_ids
        if previous is not None and len(new_xyz):
            if (not isinstance(constraint_parents, torch.Tensor) or constraint_parents.dtype != torch.long
                    or constraint_parents.device != previous.device or constraint_parents.shape != (len(new_xyz),)
                    or constraint_parents.min() < 0 or constraint_parents.max() >= len(previous)):
                raise ValueError('Every new clothing row requires its exact existing parent index.')
            # The base method reads get_xyz after growing the optimizer rows.
            self.support_face_ids = torch.cat((previous, previous[constraint_parents]))
            self._cached_triangle_pose = None
        try:
            return super().densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities,
                                                new_scaling, new_rotation, constraint_parents=constraint_parents)
        except Exception:
            if previous is not None and len(self._xyz) == len(previous):
                self.support_face_ids = previous
                self._cached_triangle_pose = None
            raise

    def prune_points(self, mask):
        if self.support_face_ids is not None:
            if mask.dtype != torch.bool or mask.shape != self.support_face_ids.shape:
                raise ValueError('The clothing prune mask must align with stable support rows.')
            support = self.support_face_ids[~mask]
            super().prune_points(mask)
            self.support_face_ids = support
            self._cached_triangle_pose = None
            return
        return super().prune_points(mask)

    def densify_and_prune(self,*args,**kwargs):
        # Parent is in unbound mode: no hidden minimum-one-point-per-face rule.
        assert self.binding is None
        if self.support_face_ids is not None:
            previous = self.timestep
            self.select_mesh_by_timestep(self.reference_frame)
            try:
                return super().densify_and_prune(*args,**kwargs)
            finally:
                self.select_mesh_by_timestep(previous)
        super().densify_and_prune(*args,**kwargs)
        self.refresh_anchors()
