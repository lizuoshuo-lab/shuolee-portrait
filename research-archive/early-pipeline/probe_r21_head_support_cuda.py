"""Check the selective split against actual R20 colors, cameras and fitted geometry."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

import launch_unified_cold_016_006 as cold

PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-head-support-20261111')
R20_SOURCE = Path('/mnt/d/Projects/Shuolee-smpl-head-align-20261111')
OUT = Path(__file__).resolve().parents[1]/'outputs/smpl-delivery-round21/input-audit'
sys.path.insert(0, str(PROJECT/'src'))

from shuolee_portrait import runtime as rt
from shuolee_portrait.bound_replay import SOURCE_FILES, _inputs, _load_model
from shuolee_portrait.head_support import correct_head_support
from shuolee_portrait.head_support_pitch import HeadSupportPitchGuard
from shuolee_portrait.sh0_main import Constraints
from shuolee_portrait.training import _frozen_smpl_surface, _native_foreground_target, camera, smpl_sh0_effective_config
from shuolee_portrait.world_scale import WorldScale
from diff_gaussian_rasterization import _C


def main():
    output = OUT/'cuda-head-support-proof.json'
    if output.exists():
        raise FileExistsError(output)
    subject = '016'
    archive = cold.LAB/f'shuolee-smpl-delivery-round20-{subject}/run'
    snapshot = archive/'gaussian-run/step-010000'
    manifest = rt.read(snapshot/'bound-replay.json')
    assert manifest['source_files_sha256'] == {name:rt.sha(R20_SOURCE/name) for name in SOURCE_FILES}
    assert manifest['bound_ply_sha256'] == rt.sha(snapshot/'bound-local.ply')
    rt.configure(cold.ASSETS)
    asset = cold.ASSETS/'smplx/SMPLX_NEUTRAL.npz'
    fit, mesh, facts = _inputs(archive, snapshot, asset)
    assert manifest['inputs_sha256'] == facts['inputs_sha256'] and manifest['asset_sha256'] == facts['asset_sha256']
    model, geometry = _load_model(snapshot/'bound-local.ply', snapshot, fit, mesh, facts, asset)
    model.posed_vertices = lambda slot: model.vertices_sequence[slot]
    model.spatial_lr_scale = 1.
    effective = smpl_sh0_effective_config()
    model.training_setup(SimpleNamespace(**effective['optimization']))
    # The original R20 PLY carries no optimizer moments. Initialize only the
    # testing clock and verify untouched rows; formal training keeps live Adam.
    for group in model.optimizer.param_groups:
        for parameter in group['params']:
            model.optimizer.state[parameter] = dict(step=torch.tensor(1.), exp_avg=torch.zeros_like(parameter),
                                                   exp_avg_sq=torch.zeros_like(parameter))
    rows = rt.read(archive/'data/training-inputs.json')['records']
    slots = [row['pose_index'] for row in facts['frames']]
    reference = facts['reference_pose_index']
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.zeros(3, device='cuda')
    _, _, _, protected, _ = _frozen_smpl_surface(fit, asset)
    head_faces = torch.as_tensor(protected, device='cuda')[model.faces].all(dim=1)
    cameras = [camera(row,i,pose_index=slot)[0] for i,(row,slot) in enumerate(zip(rows,slots,strict=True))]
    targets = []
    for row in rows:
        rgb, alpha, box = _native_foreground_target(row)
        targets.append((rgb.cuda(), alpha.cuda(), box))
    cfg = rt.read(archive/'gaussian-run/config.json')
    bins = cfg['training_source_bins']
    extent = float(np.linalg.norm(np.ptp(fit['vertices_world'][reference], axis=0)))
    schedule = effective['unified_main']
    path = OUT/f'head-support-views-{subject}.json'
    stage = OUT/'actual-016-archive-trial-joint-pitch'
    stage.mkdir(exist_ok=False)
    with torch.no_grad():
        scale = WorldScale(model, extent, reference, schedule['world_scale'])
        constraints = Constraints(model, rows, slots, cameras, fit, pipe, background, targets,
                                  bins, extent, schedule)
        pitch = HeadSupportPitchGuard(model, reference, head_faces, rt.read(path), pipe, background)
        before = model.densification_ids[~head_faces[model.binding]].clone()
        print('Testing geometry, 24 native RGB/alpha guards and 15 pitch cameras on the R20 endpoint.',flush=True)
        result = correct_head_support(model, constraints, scale, head_faces,
            observed=fit['observed_native_keypoints'], diagnostic_dir=stage, pitch_guard=pitch)
    rt.save(stage/'result.json', result)
    assert torch.equal(model.densification_ids[~head_faces[model.binding]], before)
    assert result['attempts'] and result['optimizer_updates'] > 0
    assert result['pitch_guard_required'] and len(pitch.views) == 15
    assert rt.sha(snapshot/'bound-local.ply') == manifest['bound_ply_sha256']
    for trial in result['attempts']:
        if trial['state'] == 'committed':
            assert trial['geometry']['accepted'] and trial['native_guard']['accepted'] and trial['pitch_guard']['accepted']
    files = ('head_support.py','head_support_pitch.py','head_support_views.py',
             'render_support.py','size_child_optimization.py','size_child_commit.py')
    proof = dict(state='passed', subject=subject, gpu=torch.cuda.get_device_name(),
        fixture_sha256=rt.sha(Path(__file__)), runtime=dict(torch=torch.__version__, cuda=torch.version.cuda,
                                                           raster_extension_sha256=rt.sha(Path(_C.__file__))),
        source_sha256={f'src/shuolee_portrait/{name}':rt.sha(PROJECT/'src/shuolee_portrait'/name) for name in files},
        archive=dict(bound_ply_sha256=manifest['bound_ply_sha256'], world_ply_sha256=facts['inputs_sha256']['world_ply'],
                     points=manifest['points'], fit_replay_max_abs_m=geometry['fit_replay_max_abs_m'],
                     optimizer='fresh initialized moments for archive check only'),
        camera_manifest_sha256=rt.sha(path), report_sha256=rt.sha(stage/'result.json'), native_views=24,
        initial_geometric_ranking_control_sha256=rt.sha(OUT/'cuda-head-support-proof-initial.json'),
        visible_rank_control_sha256=rt.sha(OUT/'cuda-head-support-proof-visible-support.json'), pitch_views=len(pitch.views), verified_unchanged_nonhead_ids=True, attempts=len(result['attempts']),
        optimizer_updates=result['optimizer_updates'], parents_committed=result['parents_committed'],
        children_committed=result['children_committed'], result=result['state'])
    rt.save(output, proof)
    print(f'{output}: {rt.sha(output)}',flush=True)


if __name__ == '__main__':
    main()
