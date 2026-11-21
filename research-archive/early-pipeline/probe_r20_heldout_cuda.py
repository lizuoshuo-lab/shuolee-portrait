"""Exercise the exact reserved-view CUDA replay on an immutable R19 model."""

from pathlib import Path
from shutil import copyfile
import sys
from tempfile import TemporaryDirectory

import torch


ROOT = Path('/opt/gs-slam-lab')
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-head-align-20261111')
ARCHIVE = Path('/mnt/d/Projects/Shuolee-smpl-protected-face-20261111')
OUT = Path(__file__).resolve().parents[1]/'outputs/smpl-delivery-round20/input-audit/heldout-replay-cuda.json'
sys.path.insert(0, str(PROJECT/'src'))

from shuolee_portrait import runtime as rt
from shuolee_portrait.bound_replay import SOURCE_FILES, _inputs, _load_model
from shuolee_portrait.heldout_pose import render_holdouts, verify_holdouts
from shuolee_portrait.training import _frozen_smpl_surface


def main():
    run = ROOT/'shuolee-smpl-delivery-round20-016/run'
    archive = ROOT/'shuolee-smpl-delivery-round19-016/run'
    snapshot = archive/'gaussian-run/step-010000'
    assets = Path(rt.read(run/'run.json')['assets'])
    rt.configure(assets, run)
    torch.set_num_threads(4)
    manifest = rt.read(snapshot/'bound-replay.json')
    model_path = assets/'smplx/SMPLX_NEUTRAL.npz'
    fit, mesh, facts = _inputs(archive, snapshot, model_path)
    assert manifest['source_files_sha256'] == {name: rt.sha(ARCHIVE/name) for name in SOURCE_FILES}
    assert manifest['asset_sha256'] == facts['asset_sha256']
    assert manifest['inputs_sha256'] == facts['inputs_sha256']
    assert manifest['frames'] == facts['frames']
    assert manifest['bound_ply_sha256'] == rt.sha(snapshot/'bound-local.ply')
    model, _ = _load_model(snapshot/'bound-local.ply', snapshot, fit, mesh, facts, model_path)
    plan, poses, _ = verify_holdouts(run)
    _, _, _, _, replay_error = _frozen_smpl_surface(poses, model_path)
    count_before = model.num_timesteps
    sequence_before = model.vertices_sequence
    with TemporaryDirectory(prefix='r20-holdout-cuda-', dir=ROOT) as directory:
        directory = Path(directory)
        copyfile(snapshot/'reference-mesh.npz', directory/'reference-mesh.npz')
        result = render_holdouts(run, directory, (model, manifest))
        assert [row['frame'] for row in result['records']] == plan['frames']
        assert all(set(row['files']) == {'black','white','alpha','mesh','feature'} for row in result['records'])
        assert model.num_timesteps == count_before and model.vertices_sequence is sequence_before
        assert result['optimizer_updates'] == 0
        proof = dict(state='passed', subject='016', scope='Frozen R19 Gaussian and offset replay with R20 heldout local poses',
            gpu=torch.cuda.get_device_name(), archive_bound_ply_sha256=manifest['bound_ply_sha256'],
            heldout_plan_sha256=result['holdout_plan_sha256'], native_frames=plan['frames'],
            replay_max_abs_m=replay_error, output_receipt_sha256=rt.sha(directory/'heldout-replay/result.json'),
            camera_projection_max_abs_px=max(row['camera']['projection_error_px'] for row in result['records']),
            source_sha256={name:rt.sha(PROJECT/'src/shuolee_portrait'/name) for name in
                           ('head_fit_stage.py','heldout_pose.py','training.py','sequence_pipeline.py')})
    if OUT.exists():
        raise FileExistsError(OUT)
    rt.save(OUT, proof)
    print(f'{OUT}: {rt.sha(OUT)}', flush=True)


if __name__ == '__main__':
    main()
