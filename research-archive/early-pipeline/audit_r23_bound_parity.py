"""CPU comparison of R23 initialization against exact frozen R22 geometry."""

import json
from pathlib import Path
import sys

import numpy as np
from plyfile import PlyData
import torch

sys.path[:0] = ['src', 'src/avatar_core']
from scene.clothing_gaussian_model import ClothingGaussianModel
from scene.smplx_gaussian_model import SMPLXGaussianModel
from shuolee_portrait import runtime as rt
from shuolee_portrait.bound_replay import LOCAL_PLY, MANIFEST
from shuolee_portrait.free_clothing_torch import normalized_quaternion_matrix
from shuolee_portrait.training import _frozen_smpl_surface


def matrix_covariance(model):
    axes = normalized_quaternion_matrix(model.get_rotation)*model.get_scaling[:, None, :]
    return axes @ axes.transpose(-1, -2)


@torch.no_grad()
def main(subject):
    torch.set_num_threads(4)
    root = Path(f'/opt/gs-slam-lab/shuolee-smpl-delivery-round22-{subject}/run')
    parent = root/'gaussian-run'
    snapshot = parent/'step-018000'
    manifest = rt.read(snapshot/MANIFEST)
    path = snapshot/LOCAL_PLY
    assert rt.sha(path) == manifest['bound_ply_sha256']
    fit = dict(np.load(root/'fit/fitted.npz', allow_pickle=False))
    mesh = dict(np.load(snapshot/'reference-mesh.npz', allow_pickle=False))
    assets = Path(rt.read(root/'run.json')['assets'])
    base, linear, _, _, fit_error = _frozen_smpl_surface(fit, assets/'smplx/SMPLX_NEUTRAL.npz')
    offset = torch.from_numpy(mesh['static_offset'])[..., None]
    vertices = torch.stack([v+(A@offset).squeeze(-1) for v, A in zip(base, linear, strict=True)])
    reference = int(mesh['reference_pose_index'])
    mesh_error = float((vertices[reference]-torch.from_numpy(mesh['vertices_world'])).abs().max())
    assert mesh_error < 2e-6
    held = snapshot/'heldout-replay'
    held_receipt = rt.read(held/'result.json')
    held_mesh = held/held_receipt['mesh_file']
    assert rt.sha(held_mesh) == held_receipt['mesh_sha256']
    assert held_receipt['bound_ply_sha256'] == manifest['bound_ply_sha256']
    holdout = dict(np.load(held_mesh, allow_pickle=False))
    frames = fit['frame_ids'].tolist()+holdout['frame_ids'].tolist()
    vertices = torch.cat((vertices, torch.from_numpy(holdout['vertices_world'])))
    faces = torch.from_numpy(fit['faces']).long()
    guide = rt.read(parent/'clothing-surface.json')
    support = parent/guide['support_file']
    assert rt.sha(support) == guide['support_file_sha256']
    with np.load(support, allow_pickle=False) as archive:
        mask = torch.from_numpy(archive['eligible_vertices'])
    rows = PlyData.read(path)['vertex']
    model = SMPLXGaussianModel(0)
    model.vertices_sequence, model.faces = vertices, faces
    model.binding = torch.from_numpy(rows['binding_0'].astype('int64'))
    garment = mask[faces].all(dim=1)[model.binding]
    model.binding = model.binding[garment]
    model._xyz = torch.from_numpy(np.stack([rows[axis] for axis in ('x', 'y', 'z')], axis=1))[garment]
    model._scaling = torch.from_numpy(np.stack([rows[f'scale_{axis}'] for axis in range(3)], axis=1))[garment]
    model._rotation = torch.from_numpy(np.stack([rows[f'rot_{axis}'] for axis in range(4)], axis=1))[garment]
    model.select_mesh_by_timestep(reference)
    cloth = ClothingGaussianModel(0)
    cloth._xyz, cloth._scaling, cloth._rotation = model.get_xyz, model.get_scaling.log(), model.get_rotation
    cloth.configure_triangle_support(vertices, faces, reference, model.binding)
    evidence = []
    for slot, frame in enumerate(frames):
        cloth.select_mesh_by_timestep(slot)
        model.select_mesh_by_timestep(slot)
        centers = float((cloth.get_xyz-model.get_xyz).abs().max())
        scales = float(((cloth.get_scaling-model.get_scaling)/model.get_scaling).abs().max())
        covariance = float((matrix_covariance(cloth)-matrix_covariance(model)).abs().max())
        assert centers < 3e-6 and covariance < 1e-7
        evidence.append(dict(frame=frame, center_max_m=centers, covariance_max_m2=covariance, relative_scale_max=scales))
    print(json.dumps(dict(subject=subject, point_count=len(rows), free_garment_points=int(garment.sum()),
        protected_points=int((~garment).sum()), garment_support_faces=len(model.binding.unique()),
        reference=frames[reference], train_views=len(fit['frame_ids']), holdouts=len(holdout['frame_ids']),
        frozen_fit_error_m=fit_error, frozen_reference_mesh_error_m=mesh_error,
        max_center_m=max(v['center_max_m'] for v in evidence),
        max_covariance_m2=max(v['covariance_max_m2'] for v in evidence),
        max_relative_scale=max(v['relative_scale_max'] for v in evidence), records=evidence), indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
