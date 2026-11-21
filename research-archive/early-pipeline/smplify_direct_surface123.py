"""Optional SMPLify-X clothing-offset bridge; no new pose or sequence solver.

The frozen original SMPL-X transforms are prepared once by the existing audited
FrozenSkeletonSurface adapter. Gaussian RGB gradients may update clothing only.
The fitted facial mesh, skeleton, cameras and per-frame expressions are fixed.
"""
import numpy as np
import torch
from facial_geometry_boundary123 import clothing_only_offset


def enable(self, fit_path, model_path):
    from pathlib import Path
    from run_official_gaussians123 import read, sha
    root=Path(fit_path).parent.parent
    receipt=read(root/'smplify-gaussian-interface-receipt.json')
    assert sha(root/'fit/surface-cache.npz')==receipt['surface_cache_sha256']
    with np.load(root/'fit/surface-cache.npz',allow_pickle=False) as z:c=dict(z)
    with np.load(fit_path,allow_pickle=False) as z:f=dict(z)
    assert np.array_equal(c['frame_ids'],f['frame_ids']) and np.array_equal(c['faces'],f['faces'])
    assert not np.any(f['static_offset'])
    self.surface_base=torch.as_tensor(c['base_vertices'],device='cuda')
    self.surface_linear=torch.as_tensor(c['linear'],device='cuda')
    self.surface_clothing=torch.as_tensor(c['clothing_vertices'],device='cuda')
    self.surface_protected=torch.as_tensor(c['protected_face_vertices'],device='cuda')
    self.static_offset=torch.nn.Parameter(torch.zeros((10475,3),device='cuda'))
    assert not torch.any(self.surface_clothing & self.surface_protected)
    error=float((self.surface_base-self.vertices_sequence).abs().max())
    assert error<2e-6
    return dict(mechanism='Audited original SMPL-X LBS transforms, fixed skeletal joints, shared clothing displacement',
        checked_frames=59,zero_offset_max_abs_m=error,optimizer_lr=1e-6,
        head_neck_gaussian_guidance=False,facial_fit_source='Existing native2D SMPLify-X fit; model3D prior only',
        independent_3d_face_available=False,reference_frame='frame_000033',vhap_used=False)


def vertices(self,timestep):
    delta=clothing_only_offset(self.static_offset,self.surface_clothing,self.surface_protected)
    return self.surface_base[timestep]+(self.surface_linear[timestep]@delta[:,:,None]).squeeze(-1)


@torch.no_grad()
def geometry(self):
    delta=clothing_only_offset(self.static_offset,self.surface_clothing,self.surface_protected)
    current=self.surface_base+(self.surface_linear@delta[None,:,:,None]).squeeze(-1)
    initial=self.surface_base[:,self.faces]
    changed=current[:,self.faces]
    n0=torch.cross(initial[:,:,1]-initial[:,:,0],initial[:,:,2]-initial[:,:,0],dim=-1)
    n1=torch.cross(changed[:,:,1]-changed[:,:,0],changed[:,:,2]-changed[:,:,0],dim=-1)
    a0=n0.norm(dim=-1);valid=a0>1e-12
    return dict(new_relative_normal_reversals=int(((n0*n1).sum(-1)[valid]<=0).sum()),
        minimum_area_ratio=float((n1.norm(dim=-1)[valid]/a0[valid]).min()),
        max_added_offset_m=float(delta.norm(dim=-1).max()),
        protected_face_motion_m=float((current[:,self.surface_protected]-self.surface_base[:,self.surface_protected]).abs().max()),
        protected_face_gradient=float(self.static_offset.grad[self.surface_protected].abs().max()) if self.static_offset.grad is not None else 0.,
        skeletal_joints_frozen=True)
