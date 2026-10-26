"""Frozen experiment 113: fresh fitting/initialization and static PGSR training.

This is a port of the recorded local experiment, not an upstream official method
or a quality-accepted new-video pipeline. Input cameras are estimated by COLMAP.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import types

PROJECT = Path(__file__).resolve().parents[1]
CORE = PROJECT / 'src/avatar_core'
ROUTE = PROJECT / 'src/route113'
STAGES = ['prepare', 'head', 'fit', 'clothing', 'hull', 'patch', 'normals',
          'targets', 'preflight', 'train', 'export']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_files(root, manifest):
    for row in manifest:
        path = root / row['path']
        if not path.is_file():
            raise RuntimeError(f'Missing input: {path}')
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise RuntimeError(f'Input differs from frozen 113: {path}')


def audit(dataset, assets):
    import numpy as np
    from PIL import Image
    spec = read(PROJECT / 'configs/route113-inputs.json')
    for source in read(PROJECT / 'provenance/route113-sources.json')['files']:
        assert sha(PROJECT / source['path']) == source['portable_sha256'], source['path']
    verify_files(dataset, spec['data_files'])
    verify_files(assets, spec['asset_files'])
    rows = read(dataset / 'training-inputs.json')['records']
    module_spec = importlib.util.spec_from_file_location('route113_camera_contract', ROUTE / 'pgsr/camera_contract.py')
    contract = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(contract)
    contract.require_registered_cameras(read(dataset / 'training-inputs.json'))
    assert len(rows) == 175
    assert sorted(r['source_frame_index'] for r in rows) == list(range(1, 176))
    for row in rows:
        assert row['camera_kind'] == 'fresh_full_scene_sfm'
        assert not row['camera_registration']['interpolation_used']
        E, K = np.asarray(row['w2c']), np.asarray(row['K'])
        assert E.shape == (4, 4) and K.shape == (3, 3)
        assert np.isfinite(E).all() and np.isfinite(K).all()
        assert np.max(abs(E[:3, :3] @ E[:3, :3].T - np.eye(3))) < 1e-5
        assert np.linalg.det(E[:3, :3]) > .999
        for key, digest in [('image', 'source_sha256'), ('mask', 'mask_sha256')]:
            assert sha(dataset / row[key]) == row[digest]
            with Image.open(dataset / row[key]) as image:
                assert image.size == tuple(row['wh'])
    with np.load(dataset / 'route113/shared-observations.npz') as z:
        assert set(z.files) == set(spec['shared_observation_keys'])
        assert np.array_equal(z['w2c'], np.asarray([r['w2c'] for r in rows]))
        assert all(np.array_equal(z['K'], np.asarray(r['K'])) for r in rows)
    return dict(route='113', frames=175, camera_and_pixel_identity_verified=True,
                assets_sha256_verified=True, old_person_geometry_loaded=False,
                physical_calibration_accuracy_accepted=False, quality_accepted=False)


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def prepare(dataset, assets, run):
    report = audit(dataset, assets)
    if run.exists():
        raise RuntimeError('Fresh run required: destination already exists. Use a new run name.')
    run.mkdir(parents=True)
    for name in ['data', 'inference', 'fit', 'head-fit', 'gaussian-run', 'source-final']:
        (run / name).mkdir()
    for name in ['inputs.json', 'training-inputs.json']:
        doc = read(dataset / name)
        for row in doc['records']:
            for field in ['image', 'mask']:
                if row[field] is not None:
                    row[field] = str((dataset / row[field]).resolve())
        save(run / 'data' / name, doc)
    copy(dataset / 'observations.npz', run / 'data/observations.npz')
    for name in ['prediction-postprocessed.npz', 'time-camera-inputs.npz', 'vitpose-keypoints.pt']:
        copy(dataset / 'inference' / name, run / 'inference' / name)
    mapping = {
        'hand-observations.npz': 'runtime/current/smplx-hand-depth-fresh-011/hand-observations.npz',
        'inspection/floor-reference.json': 'runtime/current/smplx-observation-support-014/floor-reference.json',
        'inspection/surface-support.npz': 'runtime/current/smplx-observation-support-014/surface-support.npz',
        'inspection/route-comparison.json': 'runtime/current/smplx-whole-body-prior-fresh-006/gaussianavatar-comparison/route-comparison.json',
        'route113/head-observations.npz': 'head-fit/observations.npz',
        'route113/head-input-manifest.json': 'head-fit/input-manifest.json',
        'route113/shared-observations.npz': 'runtime/current/shared-point-bundle-convergence-101/input.npz',
        'route113/observed-pairs.npz': 'runtime/observed-pairs.npz',
    }
    for source, dest in mapping.items():
        copy(dataset / source, run / dest)
    save(run / 'source-inputs.json', dict(route='113', old_subject_fit_loaded=False,
         old_gaussian_loaded=False, existing_detector_predictions_reused=True,
         fresh_body_fit=True, fresh_neutral_head_fit=True, fresh_optimizer=True))
    save(run / 'runtime/work-status.json', dict(local_training_running=False,
         latest_local_execution=dict(state='prepared', running=False), quality_accepted=False))
    save(run / 'input-audit.json', report)
    config = dict(route='113', dataset=str(dataset), assets=str(assets), frames=175,
                  points=200000, iterations=18000, height=1536, seed=6060, fit_steps=1200,
                  source_manifest_sha256=sha(PROJECT / 'provenance/route113-sources.json'),
                  input_manifest_sha256=sha(PROJECT / 'configs/route113-inputs.json'))
    save(run / 'config.json', config)


def shim(name, **values):
    module = types.ModuleType(name)
    module.__dict__.update(values)
    sys.modules[name] = module


def configure(dataset, assets, run, pgsr=False):
    sys.path[:0] = [str(ROUTE / ('pgsr' if pgsr else 'preparation')), str(CORE)]
    import shuolee_common as c
    c.LAB = PROJECT / 'third_party'
    c.BASE = run / 'runtime'
    c.ROOT = run
    c.PRIVATE = assets
    c.MODEL = assets / 'smplx/SMPLX_NEUTRAL.npz'

    def runtime():
        import torch
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError('Working NVIDIA CUDA GPU required.')
        return torch
    c.runtime = runtime
    import shuolee_vposer as vp
    vp.MODEL = assets / 'vposer/V02_05'
    vp.CODE = PROJECT / 'third_party/human_body_prior'
    shim('fresh_sequence_pipeline', c=c, REPO=assets / 'gvhmr', RUN=run)
    sfm = dataset / 'route113/sfm'
    shim('prepare_head_surface_082', O=run / 'head-fit', A=run, S=sfm)
    shim('prepare_human_surface_083', R=run, REF=run)

    def intrinsics(p, cv_centers=False):
        import numpy as np
        f, cx, cy, k = p
        if cv_centers:
            cx -= .5
            cy -= .5
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.]]), np.array([k, 0, 0, 0, 0.])
    shim('prepare_fullscene_person_075', R=run, S=sfm, intrinsics=intrinsics)
    if pgsr:
        import shuolee_static as s
        s.LAB = c.LAB
        s.B = c.BASE
        s.PRIVATE = assets
    return c


@contextmanager
def task_lock():
    # A kernel lock is released even after a crash; no stale PID deletion needed.
    import fcntl
    path = Path.home() / '.local/state/shuolee-portrait/gpu.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another shuolee task owns the GPU lock.')
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(dict(pid=os.getpid(), command=sys.argv)))
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def execute(stage, dataset, assets, run):
    if stage == 'prepare':
        prepare(dataset, assets, run)
        return
    cfg = read(run / 'config.json')
    assert cfg['dataset'] == str(dataset) and cfg['assets'] == str(assets)
    assert cfg['source_manifest_sha256'] == sha(PROJECT / 'provenance/route113-sources.json')
    assert cfg['input_manifest_sha256'] == sha(PROJECT / 'configs/route113-inputs.json')
    index = STAGES.index(stage)
    previous = read(run / 'stages' / (STAGES[index - 1] + '.json'))
    assert previous['state'] == 'completed'
    c = configure(dataset, assets, run, stage in ['preflight', 'train'])
    if stage == 'head':
        from initialize_ncc_pair_093 import head
        head(run)
    elif stage == 'fit':
        from fit_fresh_sequence import main
        main(run, 1200)
    elif stage == 'clothing':
        from prepare_fullscene_clothing_075 import main
        main()
    elif stage == 'hull':
        from initialize_static_clothing_cloud import main
        main(run)
    elif stage == 'patch':
        from initialize_ncc_pair_093 import patch
        patch(run)
    elif stage in ['normals', 'targets']:
        import prepare_native_geometry_113 as geometry
        geometry.R = run
        geometry.B = c.BASE
        getattr(geometry, stage)()
    elif stage == 'preflight':
        import check_native_person_113 as check
        check.R = run
        check.main(volume=True)
        result = read(run / 'native-volume-preflight/result.json')
        if not result['passed']:
            raise RuntimeError('113 numerical/support preflight failed. Do not change thresholds to proceed.')
    elif stage == 'train':
        import shuolee_static as s
        s.main(run, 18000, 1536, mask_foreground=True, pixel_centers_consistent=True,
               observed_path=c.BASE / 'observed-pairs.npz',
               native_person_path=run / 'shared-targets/native-targets.npz')
        result = read(run / 'gaussian-run/result.json')
        assert result['steps'] == 18000 and result['from_scratch']
    elif stage == 'export':
        import types
        from scene.gaussian_model import GaussianModel
        from compatible_gaussian_export import export_and_verify
        result = read(run / 'gaussian-run/result.json')
        assert result['steps'] == 18000 and result['from_scratch']
        assert sha(run / 'gaussian-run/point_cloud.ply') == result['source_ply_sha256']
        model = GaussianModel(3)
        model.load_ply(str(run / 'gaussian-run/point_cloud.ply'))
        model.representation = 'shared_static_world'
        model.select_mesh_by_timestep = types.MethodType(lambda self, t: setattr(self, 'timestep', int(t)), model)
        model.export_world_ply = types.MethodType(lambda self, path, t: self.save_ply(str(path)), model)
        export_and_verify(model, run, 18000, 99, existing_only=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['audit', 'doctor', 'all', *STAGES])
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    dataset, assets = args.dataset.expanduser().resolve(), args.assets.expanduser().resolve()
    if args.stage == 'audit':
        print(json.dumps(audit(dataset, assets), ensure_ascii=False))
        return
    if args.stage == 'doctor':
        c = configure(dataset, assets, Path('/tmp/shuolee-doctor-unused'), pgsr=True)
        torch = c.runtime()
        assert torch.__version__.split('+')[0] == '2.7.1' and torch.version.cuda == '12.8', 'Use the pinned server environment.'
        import diff_plane_rasterization, diff_gaussian_rasterization, simple_knn._C
        import nvdiffrast.torch, pytorch3d, smplx
        from check_native_person_113 import synthetic
        with task_lock():
            result = synthetic(volume=True)
        if not result['passed']:
            raise RuntimeError(json.dumps(result))
        print(json.dumps(dict(state='CUDA_controls_passed', torch=torch.__version__,
              gpu=torch.cuda.get_device_name(), controls=result,
              full_training_verified=False, quality_accepted=False)))
        return
    if not args.run:
        parser.error('--run is required')
    run = args.run.expanduser().resolve()
    if args.stage == 'all':
        if run.exists():
            raise RuntimeError('Output already exists. No automatic resume; choose a fresh run name.')
        for stage in STAGES:
            print(f'113 stage: {stage}', flush=True)
            command = [sys.executable, '-s', str(Path(__file__).resolve()), stage,
                       '--dataset', str(dataset), '--assets', str(assets), '--run', str(run)]
            if stage == 'prepare':
                subprocess.run(command, check=True)
            else:
                with (run / 'logs' / (stage + '.log')).open('w', encoding='utf-8') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f'Finished: {run}. Export and reconstruction quality are separate; review the viewer PLY.')
        return
    marker = run / 'stages' / (args.stage + '.json')
    if args.stage == 'prepare' and run.exists():
        raise RuntimeError('Fresh run required: destination already exists.')
    if marker.exists():
        raise RuntimeError(f'Stage already attempted: {marker}. Inspect evidence; no silent overwrite or retry.')
    t0 = time.time()
    with task_lock():
        try:
            execute(args.stage, dataset, assets, run)
        except Exception as exc:
            if run.is_dir():
                save(marker, dict(state='failed', stage=args.stage, error=str(exc), seconds=time.time()-t0))
            raise
        (run / 'logs').mkdir(exist_ok=True)
        save(marker, dict(state='completed', stage=args.stage, seconds=time.time()-t0,
                         quality_accepted=False))
    print(f'{args.stage}: completed')


if __name__ == '__main__':
    main()
