"""R25 exact initialization, actual garment gradients and split-Adam parity on CPU."""

from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

sys.path[:0] = ['src','src/avatar_core']
from scene import gaussian_model
from shuolee_portrait.anchored_garment_surface import SurfaceAnchoredGarmentModel, NORMAL_GROUP
from shuolee_portrait.garment_surface import RAW_PARAMETERS
from audit_r23_numerical import ply_parameters


torch.set_num_threads(4)
subject = sys.argv[1] if len(sys.argv)>1 else '016'
drive = Path(f'/opt/gs-slam-lab/shuolee-smpl-delivery-round24-{subject}/run/gaussian-run/raw-step-018000/drive')
motion = dict(np.load(drive/'motion.npz',allow_pickle=False))
graph = dict(np.load(drive/'garment-graph.npz',allow_pickle=False))
params = ply_parameters(drive/'clothing-canonical.ply')
reference = motion['reference_index'].item()


def model():
    cloth = SurfaceAnchoredGarmentModel(0)
    for key in RAW_PARAMETERS:
        setattr(cloth,key,torch.nn.Parameter(torch.from_numpy(params[key].copy())))
    cloth.configure_triangle_support(motion['vertices_world'],motion['faces'],reference,motion['clothing_support_face_ids'],
        stable_ids=motion['clothing_stable_ids'],next_stable_id=motion['next_clothing_stable_id'].item())
    cloth.configure_garment_surface(graph=graph)
    return cloth


cloth = model()
baseline = []
with torch.no_grad():
    for index in range(cloth.num_timesteps):
        cloth.select_mesh_by_timestep(index)
        baseline.append(cloth.get_xyz.clone())
    cloth.configure_surface_anchors()
    for index, before in enumerate(baseline):
        cloth.select_mesh_by_timestep(index)
        assert torch.equal(cloth.get_xyz,before), float((cloth.get_xyz-before).abs().max())
print('exact zero-initialization in all poses',cloth.num_timesteps,'metrics',cloth.anchor_metrics(),flush=True)

cloth.optimizer = torch.optim.Adam([dict(name=name,params=[getattr(cloth,key)]) for name,key in
    zip(('xyz','f_dc','f_rest','scaling','rotation','opacity'),RAW_PARAMETERS,strict=True)],lr=.001)
cloth.configure_anchor_optimizer(.01)
cloth._opacity.requires_grad_(False)
surface = torch.optim.Adam([cloth.node_centers,cloth.anchor_tangent_raw],lr=.001)
rows = len(cloth._xyz)
with torch.enable_grad():
    loss = cloth.get_xyz.square().mean()+sum(cloth.garment_regularization().values())
    loss.backward()
    for key in ('node_centers','anchor_normal_raw','anchor_tangent_raw','_scaling','_rotation'):
        gradient = getattr(cloth,key).grad
        assert torch.isfinite(gradient).all() and gradient.abs().max()>0
    cloth.optimizer.step()
    surface.step()
    cloth.optimizer.zero_grad(set_to_none=True)
    surface.zero_grad(set_to_none=True)

reload = model()
for key in RAW_PARAMETERS:
    setattr(reload,key,torch.nn.Parameter(getattr(cloth,key).detach().clone()))
reload.configure_garment_surface(graph=cloth.graph_arrays())
reload.configure_surface_anchors(anchors=cloth.anchor_arrays())
reload.select_mesh_by_timestep(cloth.timestep)
assert torch.equal(cloth.get_xyz,reload.get_xyz)
assert torch.equal(cloth.get_covariance(),reload.get_covariance())


class CpuZeros:
    def __getattr__(self,name):
        return getattr(torch,name)

    def zeros(self,*shape,device='cpu',**kwargs):
        return torch.zeros(*shape,device='cpu',**kwargs)


with torch.no_grad(), patch.object(gaussian_model,'torch',CpuZeros()):
    cloth.max_radii2D = torch.zeros(rows)
    parents = torch.where(cloth.anchor_trusted)[0][:3]
    inherited = parents.repeat(2)
    kept_mask = torch.ones(rows,dtype=torch.bool)
    kept_mask[parents] = False
    kept = torch.where(kept_mask)[0]
    next_id = cloth.next_densification_id
    child = {key:getattr(cloth,key)[inherited].clone() for key in RAW_PARAMETERS}
    child['_xyz'][:3,0] -= .001
    child['_xyz'][3:,0] += .001
    attributes = {key:torch.cat((getattr(cloth,key)[kept],child[key])) for key in RAW_PARAMETERS}
    preview = cloth.preview_rows(torch.cat((kept,inherited)),attributes)
    old_normal = cloth.anchor_normal_raw[parents].clone()
    old_moment = cloth.optimizer.state[cloth.anchor_normal_raw]['exp_avg'][kept].clone()
    cloth.densification_postfix(child['_xyz'],child['_features_dc'],child['_features_rest'],child['_opacity'],
         child['_scaling'],child['_rotation'],constraint_parents=inherited)
    cloth.prune_points(torch.cat((~kept_mask,torch.zeros(len(inherited),dtype=torch.bool))))
    assert torch.equal(cloth.get_xyz,preview.get_xyz) and torch.equal(cloth.get_covariance(),preview.get_covariance())
    assert torch.equal(cloth.anchor_normal_raw[-6:],old_normal.repeat(2,1))
    assert torch.equal(cloth.densification_ids[-6:],torch.arange(next_id,next_id+6))
    state = cloth.optimizer.state[cloth.anchor_normal_raw]
    assert torch.equal(state['exp_avg'][:-6],old_moment) and not state['exp_avg'][-6:].count_nonzero()
    [group] = [group for group in cloth.optimizer.param_groups if group['name']==NORMAL_GROUP]
    assert group['params'][0] is cloth.anchor_normal_raw
    assert not cloth._xyz.requires_grad and surface.param_groups[0]['params'][1] is cloth.anchor_tangent_raw
print('actual native pose, gradients, anchor reload, preview commit, normal Adam inheritance passed',flush=True)
