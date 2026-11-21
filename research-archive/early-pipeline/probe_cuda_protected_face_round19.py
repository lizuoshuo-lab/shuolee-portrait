"""Compare the three guarded face paths on an immutable real R17 endpoint."""

import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import launch_unified_cold_016_006 as cold

PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-protected-face-20261111')
ARCHIVE_SOURCE = Path('/mnt/d/Projects/Shuolee-smpl-multiview-quality-20261111')
R17 = cold.LAB/'shuolee-smpl-delivery-round17-016/run'
R18 = cold.LAB/'shuolee-smpl-delivery-round18-016/run'
PROOF = Path(__file__).parents[1]/'outputs/smpl-delivery-round19/input-audit/cuda-protected-face-proof.json'
sys.path.insert(0, str(PROJECT/'src'))

from shuolee_portrait import runtime as rt
from shuolee_portrait import protected_face
from shuolee_portrait.bound_replay import SOURCE_FILES, _inputs, _load_model
from shuolee_portrait.face_supplement import prepare_face_supplements
from shuolee_portrait.face_supplement_training import prepare_targets
from shuolee_portrait.sh0_main import Constraints
from shuolee_portrait.training import _frozen_smpl_surface, _native_foreground_target, camera, smpl_sh0_effective_config
from diff_gaussian_rasterization import _C
from gaussian_renderer import render


def load_fixture():
    snapshot = R17/'gaussian-run/step-010000'
    manifest = rt.read(snapshot/'bound-replay.json')
    assert manifest['source_files_sha256'] == {name: rt.sha(ARCHIVE_SOURCE/name) for name in SOURCE_FILES}
    assert rt.sha(snapshot/'bound-local.ply') == manifest['bound_ply_sha256']
    for path in ('fit/fitted.npz', 'data/training-inputs.json'):
        assert rt.sha(R17/path) == rt.sha(R18/path), path
    asset = cold.ASSETS/'smplx/SMPLX_NEUTRAL.npz'
    fit, mesh, facts = _inputs(R17, snapshot, asset)
    assert manifest['asset_sha256'] == facts['asset_sha256']
    assert manifest['inputs_sha256'] == facts['inputs_sha256']
    assert manifest['frames'] == facts['frames']
    model, geometry = _load_model(snapshot/'bound-local.ply', snapshot, fit, mesh, facts, asset)
    assert len(model.binding) == manifest['points'] == 92739
    return model, fit, facts, manifest, geometry


def set_optimizer(model, effective):
    # R17 PLY never serialized Adam. Both branches start with the same new Adam.
    model.spatial_lr_scale = 1.
    model.training_setup(SimpleNamespace(**effective['optimization']))
    for group in model.optimizer.param_groups:
        for param in group['params']:
            model.optimizer.state[param] = dict(step=torch.tensor(0.),
                exp_avg=torch.zeros_like(param), exp_avg_sq=torch.zeros_like(param))


def main():
    if PROOF.exists():
        raise FileExistsError(PROOF)
    torch.manual_seed(6022)
    torch.cuda.manual_seed_all(6022)
    rt.configure(cold.ASSETS)
    model, fit, facts, manifest, geometry = load_fixture()
    rows = rt.read(R18/'data/training-inputs.json')['records']
    supplements, admission = prepare_face_supplements(R18, rows)
    assert admission == rt.read(R18/'gaussian-run/face-supplement/admission.json')
    assert len(rows) == 24 and len(supplements) == 4
    slots = [entry['pose_index'] for entry in facts['frames']]
    _, _, _, protected, _ = _frozen_smpl_surface(fit, rt.ASSETS/'smplx/SMPLX_NEUTRAL.npz')
    head_faces = torch.as_tensor(protected, device='cuda')[model.faces].all(dim=1)
    first_aux = len(model.vertices_sequence)
    model.vertices_sequence = torch.cat((model.vertices_sequence, torch.as_tensor(
        np.stack([item.pose['vertices_world'] for item in supplements]), dtype=torch.float32, device='cuda')))
    effective = smpl_sh0_effective_config()
    set_optimizer(model, effective)
    cameras = [camera(row, i, pose_index=slot)[0] for i, (row, slot) in enumerate(zip(rows, slots, strict=True))]
    native_targets = []
    for row in rows:
        rgb, alpha, box = _native_foreground_target(row)
        native_targets.append((rgb.cuda(), alpha.cuda(), box))
    bins = rt.read(R17/'gaussian-run/config.json')['training_source_bins']
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.zeros(3, device='cuda')
    extent = float(np.linalg.norm(np.ptp(fit['vertices_world'][facts['reference_pose_index']], axis=0)))
    with torch.no_grad():
        constraints = Constraints(model, rows, slots, cameras, fit, pipe, background, native_targets,
                                  bins, extent, effective['unified_main'])
    proof_dir = PROOF.parent/'real-016'
    proof_dir.mkdir(parents=True, exist_ok=False)
    targets = prepare_targets(supplements, first_aux, camera, proof_dir)
    print('R17 real model and R18 frozen cameras validated; comparing 24 original + 4 supplemental frames.', flush=True)
    before_file = rt.sha(R17/'gaussian-run/step-010000/bound-local.ply')
    config = dict(protected_face.CONFIG, max_cycles=2, min_cycles=2)
    with patch.dict(protected_face.CONFIG, config):
        result = protected_face.optimize_protected_face(model, constraints, targets, head_faces, render, pipe,
                    background, observed=fit['observed_native_keypoints'], directory=proof_dir)
    rt.save(proof_dir/'comparison.json', result)
    assert before_file == rt.sha(R17/'gaussian-run/step-010000/bound-local.ply')
    assert result['baseline']['all_guards_pass'] and result['final']['all_guards_pass']
    assert result['rng_unchanged'] and result['unprotected']['same_endpoint']
    assert len(result['final']['frames']) == 28 and result['attempted_updates'] >= 28
    assert result['state'] in ('rolled_back', 'committed')
    if result['state'] == 'rolled_back':
        assert result['live_final'] == result['endpoint'] and result['committed_updates'] == 0
    else:
        assert result['final_gains']['supplemental'] > 0 and result['committed_updates'] > 0
    # Force an infeasible acceptance threshold in a separate bounded control.
    # The branch executes real CUDA updates, then has to leave every live tensor and moment unchanged.
    rollback_dir = proof_dir/'forced-rollback'
    rollback_dir.mkdir()
    with patch.dict(protected_face.CONFIG, dict(max_cycles=1, min_cycles=1, minimum_gain=1e6)):
        rollback = protected_face.optimize_protected_face(model, constraints, targets, head_faces, render, pipe,
                    background, observed=fit['observed_native_keypoints'], directory=rollback_dir)
    rt.save(rollback_dir/'comparison.json', rollback)
    assert rollback['state'] == 'rolled_back' and rollback['attempted_updates'] >= 28
    assert rollback['live_final'] == rollback['endpoint'] and rollback['final']['all_guards_pass']
    assert rollback['optimizer_state'] is None and rollback['committed_updates'] == 0
    files = ['training.py', 'protected_face.py', 'face_supplement.py', 'face_supplement_training.py',
             'sh0_main.py', 'render_support.py', 'bound_replay.py']
    helper = PROJECT/'src/shuolee_portrait/protected_face_views.py'
    if helper.exists():
        files.append(helper.name)
    proof = dict(state='passed', gpu=torch.cuda.get_device_name(),
        runtime=dict(torch=torch.__version__, cuda=torch.version.cuda, raster_extension_sha256=rt.sha(Path(_C.__file__))),
        source_sha256={f'src/shuolee_portrait/{name}':rt.sha(PROJECT/'src/shuolee_portrait'/name) for name in files},
        fixture_sha256=rt.sha(Path(__file__)),
        archive=dict(subject='016', points=manifest['points'], bound_ply_sha256=before_file,
                     fitted_sha256=facts['inputs_sha256']['fitted'], optimizer='fresh; R17 archive is PLY',
                     fit_replay_max_abs_m=geometry['fit_replay_max_abs_m']),
        real_same_endpoint=dict(state=result['state'], original_views=len(rows), auxiliary_views=len(targets),
            baseline_verified=result['baseline']['all_guards_pass'], final_verified=result['final']['all_guards_pass'],
            same_endpoint=result['unprotected']['same_endpoint'], rng_unchanged=result['rng_unchanged'],
            unprotected_failed_frames=result['unprotected']['measurement']['failed_frames'],
            final_gains=result['final_gains'], attempted_updates=result['attempted_updates'],
            committed_updates=result['committed_updates'], comparison_sha256=rt.sha(proof_dir/'comparison.json')),
        forced_rollback=dict(verified=True, attempted_updates=rollback['attempted_updates'],
                             comparison_sha256=rt.sha(rollback_dir/'comparison.json')))
    rt.save(PROOF, proof)
    print(f'{PROOF}: {rt.sha(PROOF)}', flush=True)


if __name__ == '__main__':
    main()
