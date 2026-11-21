"""Exercise actual R23 016 frozen poses and R24 shared graph on CPU."""

from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

sys.path[:0] = ['src', 'src/avatar_core']
from shuolee_portrait.garment_surface import GarmentSurfaceModel, RAW_PARAMETERS
from shuolee_portrait.free_clothing_torch import normalized_quaternion_matrix
from audit_r23_numerical import ply_parameters
from scene import gaussian_model


torch.set_num_threads(4)
seed = Path('/opt/gs-slam-lab/shuolee-smpl-delivery-round23-016/run/gaussian-run/raw-step-018000/drive')
data = dict(np.load(seed/'motion.npz', allow_pickle=False))
arrays = ply_parameters(seed/'clothing-canonical.ply')
ref = data['reference_index'].item()
face = data['faces']
asset = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz')
with np.load(asset, allow_pickle=False) as z:
    dominant = z['weights'][face].mean(1).argmax(1)
groups = np.zeros(len(face), dtype=np.int64)
for index, joints in enumerate(((13,16,18,20,22),(14,17,19,21,23),(1,4,7,10),(2,5,8,11)), 1):
    groups[np.isin(dominant, joints)] = index


def model():
    m = GarmentSurfaceModel(0)
    for key in RAW_PARAMETERS:
        setattr(m,key,torch.nn.Parameter(torch.from_numpy(arrays[key].copy())))
    m.configure_triangle_support(data['vertices_world'],face,ref,data['clothing_support_face_ids'],
        stable_ids=data['clothing_stable_ids'],next_stable_id=data['next_clothing_stable_id'].item())
    return m


start = time.monotonic()
m = model()
baseline = m.get_xyz.detach().clone()
m.configure_garment_surface(face_groups=groups)
assert torch.equal(m.get_xyz, baseline)
assert not m._xyz.requires_grad
print('construction_seconds', time.monotonic()-start, 'rows',len(m._xyz),'nodes',len(m.node_centers),
      'edges',len(m.graph_edges),'node_normals',int(m._node_has_normal.sum()),
      'multi_node_rows',int((m.row_node_weights.count_nonzero(dim=1)>1).sum()), flush=True)
with torch.no_grad():
    m.node_centers.add_(torch.randn(m.node_centers.shape,generator=torch.Generator().manual_seed(17))*.002)
    m.apply_graph_limits(.004)
    assert float((m.node_centers-m.node_rest_xyz).norm(dim=1).max()) < .004001
m.select_mesh_by_timestep(4)
regularization = m.garment_regularization(torch.arange(len(m._xyz)) % 3 == 0)
loss = m.get_xyz.square().mean() + sum(regularization.values())
grads = torch.autograd.grad(loss, (m.node_centers,m._scaling,m._rotation))
for field, gradient in zip(('node','logscale','quaternion'),grads,strict=True):
    assert torch.isfinite(gradient).all() and float(gradient.abs().max()) > 0
    print('gradient_max',field,float(gradient.abs().max()),flush=True)
expected = normalized_quaternion_matrix(m.get_rotation)*m.get_scaling[:,None,:]
cov = expected@expected.transpose(-1,-2)
assert torch.equal(m.get_covariance(),cov[:,(0,0,0,1,1,2),(0,1,2,1,2,2)])
reload = model()
reload.configure_garment_surface(graph=m.graph_arrays())
reload.select_mesh_by_timestep(4)
for attribute in ('get_xyz','get_scaling','get_rotation','get_features','get_opacity'):
    assert torch.equal(getattr(m,attribute),getattr(reload,attribute))
assert torch.equal(m.get_covariance(),reload.get_covariance())

rows = torch.tensor([0,2,2,7])
preview_xyz = torch.nn.Parameter(m._xyz[rows].detach().clone())
preview = m.preview_rows(rows, {'_xyz':preview_xyz})
assert torch.equal(preview.get_xyz,m.get_xyz[rows])
for _ in range(2):
    torch.autograd.grad(preview.get_xyz.square().mean(), preview_xyz)
assert not preview.node_centers.requires_grad and m.node_centers.requires_grad
assert torch.equal(preview.row_node_ids,m.row_node_ids[rows])


class CpuZeros:
    def __getattr__(self,name):
        return getattr(torch,name)

    def zeros(self,*shape,device='cpu',**kwargs):
        return torch.zeros(*shape,device='cpu',**kwargs)


with torch.no_grad(), patch.object(gaussian_model,'torch',CpuZeros()):
    m.optimizer = torch.optim.Adam([{'name':name,'params':[getattr(m,field)]} for name,field in
       zip(('xyz','f_dc','f_rest','scaling','rotation','opacity'),RAW_PARAMETERS,strict=True)],lr=.001)
    m.max_radii2D = torch.zeros(len(m._xyz))
    parents = torch.tensor([2,7])
    identity = id(m.node_centers)
    old_ids = m.row_node_ids[parents].clone()
    old_weights = m.row_node_weights[parents].clone()
    m.densification_postfix(m._xyz[parents]+.001,m._features_dc[parents],m._features_rest[parents],
        m._opacity[parents],m._scaling[parents],m._rotation[parents],constraint_parents=parents)
    assert torch.equal(m.row_node_ids[-2:],old_ids) and torch.equal(m.row_node_weights[-2:],old_weights)
    mask = torch.zeros(len(m._xyz),dtype=torch.bool)
    mask[parents] = True
    m.prune_points(mask)
    assert torch.equal(m.row_node_ids[-2:],old_ids) and torch.equal(m.row_node_weights[-2:],old_weights)
    assert not m._xyz.requires_grad and id(m.node_centers) == identity
    assert m.timestep == 4 and torch.isfinite(m.get_xyz).all()
print('CPU actual geometry, gradients, covariance, exact graph reload, independent preview, append/prune passed',flush=True)
