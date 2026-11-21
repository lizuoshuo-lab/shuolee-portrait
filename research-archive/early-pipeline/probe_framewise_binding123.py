"""Bounded CPU control of the existing repository's mesh-relative Gaussians.

The saved mesh is test geometry only. No old fit or camera becomes a new
initializer. Exact source methods are selected by AST to avoid GPU constructors.
"""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import os
import time
import traceback
import numpy as np
import torch
from roma import (rotmat_to_unitquat, quat_xyzw_to_wxyz, quat_wxyz_to_xyzw,
                  quat_product, unitquat_to_rotmat)
from scipy.spatial.transform import Rotation

H = Path(__file__).parent
T = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REF = T / 'shuolee/current/local-clothed-surface-123/smplifyx-reference-control'
OUT = REF / 'framewise-canonical-route'
REPO = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/GaussianAvatars_SMPLX-shuolee')
RULE = Path('/mnt/d/00/记录/规则.md')
THOUGHT = Path('/mnt/d/00/思路.md')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def save(p, value): p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def selected_functions(path, names, cls=None):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    body = tree.body if cls is None else next(n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    nodes = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    for node in nodes: node.decorator_list = []
    return compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec')

def main():
    assert not OUT.exists(), 'Inspect the existing control instead of rerunning.'
    OUT.mkdir()
    mesh = REF / 'conditioned-reference/slot000-fit.npz'
    files = [THOUGHT, RULE, mesh, REPO/'scene/gaussian_model.py',
             REPO/'scene/smplx_gaussian_model.py', REPO/'utils/graphics_utils.py', Path(__file__)]
    hashes = {str(p): sha(p) for p in files}
    save(OUT/'plan.json', dict(registered=now(), authorization=str(THOUGHT), input_hashes=hashes,
        user_route='SMPLify-X per-frame articulated pose and expression with one shared identity; GaussianAvatars_SMPLX shared mesh-relative Gaussian field; export a selected static pose.',
        updates_previous_static_only_intermediate=True, final_deliverable_remains_static=True,
        face_priority=True, unreliable_hand_fit_optional=True,
        hypothesis='The existing local SMPLX adapter can transport one shared Gaussian field between mesh frames and produce a single world-space static snapshot without pooling frame clouds.',
        test='Exact existing source selector and xyz/rotation/scale/SH getters on a known rigid+scale transformed mesh and a small synthetic nonrigid mesh. Independent rigid-motion oracle and deliberate covariance error control.',
        budget=dict(cpu_seconds=120, gaussian_samples=128, mesh_frames=3, GPU_jobs=0, person_fits=0, camera_fits=0, training_steps=0),
        limitations=['Not a person alignment or physical camera test.', 'Independent old per-image fits are not a valid shared-identity sequence.',
                     'Local adapter is not a complete official SMPLX implementation.',
                     'No adaptive parameter scan or old checkpoint initialization.'],
        stops='Any source drift or numerical failure stops this control; preserve failures.'))
    save(OUT/'progress.json', dict(state='running', pid=os.getpid(), started=now()))
    began = time.monotonic()
    torch.set_num_threads(4)
    ns = dict(torch=torch, rotmat_to_unitquat=rotmat_to_unitquat,
              quat_xyzw_to_wxyz=quat_xyzw_to_wxyz, quat_wxyz_to_xyzw=quat_wxyz_to_xyzw,
              quat_product=quat_product)
    exec(selected_functions(REPO/'utils/graphics_utils.py', ['dot', 'length', 'safe_normalize', 'compute_face_orientation']), ns)
    exec(selected_functions(REPO/'scene/smplx_gaussian_model.py', ['select_mesh_by_timestep'], 'SMPLXGaussianModel'), ns)
    exec(selected_functions(REPO/'scene/gaussian_model.py', ['get_xyz', 'get_rotation', 'get_scaling', 'get_features'], 'GaussianModel'), ns)
    z = np.load(mesh, allow_pickle=False)
    v = torch.tensor(z['vertices_camera'][0], dtype=torch.float64)
    faces = torch.tensor(z['faces'], dtype=torch.long)
    Q = torch.tensor(Rotation.from_euler('zyx', [55, -13, 7], degrees=True).as_matrix(), dtype=v.dtype)
    shift = torch.tensor([.2, -.3, .4], dtype=v.dtype)
    scale = 1.25
    v1 = scale*(v@Q.T)+shift
    centred = v-v.mean(0)
    v2 = v.clone()
    v2[:, 0] += .08*centred[:, 1]**2
    class Model: pass
    m = Model()
    m.vertices_sequence = torch.stack([v, v1, v2])
    m.faces = faces
    m.binding = torch.linspace(0, len(faces)-1, 128).long()
    gen = torch.Generator().manual_seed(118)
    m._xyz = torch.randn((128, 3), generator=gen, dtype=v.dtype)*.4
    m._rotation = torch.nn.functional.normalize(torch.randn((128, 4), generator=gen, dtype=v.dtype), dim=-1)
    m._scaling = torch.log(torch.tensor([.2, .07, .03], dtype=v.dtype).repeat(128, 1))
    m._features_dc = torch.randn((128, 1, 3), generator=gen, dtype=v.dtype)
    m._features_rest = torch.randn((128, 15, 3), generator=gen, dtype=v.dtype)
    m.scaling_activation = torch.exp
    m.rotation_activation = lambda x: torch.nn.functional.normalize(x, dim=-1)
    snapshots = []
    for frame in range(3):
        ns['select_mesh_by_timestep'](m, frame)
        xyz, scaling, rotation = (ns[f'get_{key}'](m).clone() for key in ['xyz', 'scaling', 'rotation'])
        r = unitquat_to_rotmat(quat_wxyz_to_xyzw(rotation))
        cov = r@torch.diag_embed(scaling**2)@r.transpose(-1, -2)
        local = torch.einsum('nij,nj->ni', m.face_orien_mat[m.binding].transpose(-1, -2),
                            (xyz-m.face_center[m.binding])/m.face_scaling[m.binding])
        assert torch.max(torch.abs(local-m._xyz)) < 1e-10
        snapshots.append(dict(xyz=xyz, scaling=scaling, rotation=rotation, covariance=cov,
                              features=ns['get_features'](m).clone(), local_recovery_error=float((local-m._xyz).abs().max())))
    expected_xyz = scale*(snapshots[0]['xyz']@Q.T)+shift
    expected_cov = scale**2*(Q@snapshots[0]['covariance']@Q.T)
    position_error = float((snapshots[1]['xyz']-expected_xyz).abs().max())
    covariance_error = float((snapshots[1]['covariance']-expected_cov).abs().max())
    wrong_cov_error = float((scale**2*snapshots[0]['covariance']-expected_cov).abs().max())
    assert position_error < 1e-10 and covariance_error < 1e-12 and wrong_cov_error > 1e-8
    assert all(torch.equal(snapshots[0]['features'], s['features']) for s in snapshots)
    ns['select_mesh_by_timestep'](m, 0)
    assert torch.equal(ns['get_xyz'](m), snapshots[0]['xyz'])
    assert torch.equal(ns['get_rotation'](m), snapshots[0]['rotation'])
    assert torch.equal(ns['get_scaling'](m), snapshots[0]['scaling'])
    assert all(sha(Path(p)) == h for p, h in hashes.items())
    assert time.monotonic()-began < 120
    summary = dict(completed=now(), state='passed', source_methods_executed=True, GPU_used=False,
        gaussian_samples=128, mesh_frames=3, mesh_vertices=len(v), triangles=len(faces),
        known_rigid_scale_position_max_error=position_error, known_rigid_scale_covariance_max_error=covariance_error,
        deliberate_unrotated_covariance_max_error=wrong_cov_error,
        per_frame_local_recovery_max_error=[s['local_recovery_error'] for s in snapshots],
        all_SH3_coefficients_preserved=True, reference_snapshot_position_rotation_scale_exact=True,
        source_hashes_unchanged=True, person_alignment_passed=False, world_PLY_export_tested=False,
        native_image_equivalence_tested=False, shared_identity_fitting_implemented=False,
        new_camera_candidates=0, gaussian_training_steps=0, physical_quality_passed=False,
        seconds=time.monotonic()-began,
        next='Implement a generic shared-identity, per-frame pose/expression observation consumer. First validate known synthetic motion and independent facial coverage; do not average independent per-frame body shapes or clouds.')
    save(OUT/'result.json', summary)
    save(OUT/'progress.json', dict(state='complete', pid=None, finished=now()))
    print(json.dumps(summary))

if __name__ == '__main__':
    try: main()
    except Exception as exc:
        if OUT.exists():
            save(OUT/'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(), auto_retry=False))
            save(OUT/'progress.json', dict(state='failed', pid=None, finished=now()))
        raise
