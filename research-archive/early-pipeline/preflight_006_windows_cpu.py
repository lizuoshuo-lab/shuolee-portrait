"""One native Windows CPU fit parity check for the frozen, admitted 006 frame 021."""

import argparse
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

sys.dont_write_bytecode = True
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')

import numpy as np


FRAME = 'frame_000021'
WSL = Path('//wsl.localhost/Ubuntu-20.04')
ARCHIVE = WSL / 'opt/gs-slam-lab'
REFERENCE = ARCHIVE / 'shuolee-smpl-fit-stages006-methodhook'
RUN = ARCHIVE / 'shuolee-smpl-placement-006/run'
PROJECT = Path('D:/Projects/Shuolee-smpl-body-placement')
SMPLX = ARCHIVE / 'shuolee-portrait-clean-001/env/lib/python3.10/site-packages/smplx'
OUTPUT = Path(__file__).resolve().with_name('native-windows-fit-006-frame021')
REFERENCE_HASHES = {
    REFERENCE / 'preregistration.json': 'c0353892df1539c9e3bca670148a1e87d70630463d3ba6a532a2d0e276e92823',
    REFERENCE / 'result.json': 'bd5e91d743e94c61c9f96770ce86774ce40d461e6cd608ab8d32ae98e4bd0a16',
}
SMPLX_HASHES = {
    '__init__.py': 'c45c111b7736fe9f00b6b6ec7d1ced44ccd0ad2838cc57012d80d8b7080a9786',
    'body_models.py': '3de717b66a31368120570dedd45a42effc1ce7b93d780bd71ce6879c10eeff75',
    'joint_names.py': 'c62c19a75b730ca427563be659ecac5c1002b355b1f81ea388bc54dc0ef9dc43',
    'lbs.py': '13e2bfc41be990763487b1c6ed4d01b24fae421d320da1592185bb7b8c725575',
    'utils.py': '310cb0da8a5e886af1d8f69b163fe2440134acb318ab287a4f13c96c06c96b39',
    'vertex_ids.py': '1fa22bcbdb89eec7640224b70140a1e9404baf078fa8667f9e0118026b4276be',
    'vertex_joint_selector.py': 'eb542d80acc31d50a258a7582203fe7bfdfeb95c082aaa155965f9a4cfdbf007',
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def windows_path(path):
    if path.startswith('/opt/'):
        return WSL / path.lstrip('/')
    if path.startswith(('/mnt/c/', '/mnt/d/')):
        return Path(path[5] + ':/' + path[7:])
    candidate = Path(path)
    need(candidate.is_absolute(), f'Unbound Windows input path {path}')
    return candidate


def sha(path):
    digest = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, obj):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write('\n')


def exact(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()


def prepare(output):
    import cv2
    import omegaconf
    import torch
    import tqdm
    import yaml

    need(os.name == 'nt' and torch.__version__ == '2.7.1+cu128'
         and torch.version.git_version == 'e2d141dbde55c2a4370fac5165b0561b6af4798b',
         'Expected the existing native Windows Torch environment.')
    pins = dict(REFERENCE_HASHES)
    for path, expected in pins.items():
        need(sha(path) == expected, f'Archive changed: {path}')
    reference = read(REFERENCE / 'preregistration.json')
    prior = read(REFERENCE / 'result.json')
    need(prior['exact_stock_parity'] and prior['records'][FRAME]['exact_archive_parity']
         and not prior['records'][FRAME]['screen_flags']
         and prior['preregistration_sha256'] == REFERENCE_HASHES[REFERENCE / 'preregistration.json'],
         'Expected the original admitted and exact WSL observer.')
    pins.update({windows_path(path): digest for path, digest in reference['source_pins'].items()})
    pins.update({SMPLX / name: value for name, value in SMPLX_HASHES.items()})
    pins[Path(__file__).resolve()] = sha(Path(__file__).resolve())
    for path, expected in pins.items():
        need(sha(path) == expected, f'Frozen path differs: {path}')
    head = subprocess.check_output([r'C:\Program Files\Git\cmd\git.exe', '-C', str(PROJECT),
                                    'rev-parse', 'HEAD'], text=True).strip()
    need(head == reference['source_commit'] == '28ea12f5a89511770213689a52ebf0ecb714d1d0',
         'Original placement worktree changed.')
    need('smplx' not in sys.modules, 'Unexpected SMPL-X module already loaded.')
    spec = spec_from_file_location('smplx', SMPLX / '__init__.py', submodule_search_locations=[str(SMPLX)])
    smplx = module_from_spec(spec)
    sys.modules['smplx'] = smplx
    spec.loader.exec_module(smplx)
    sys.path.insert(0, str(PROJECT / 'src'))
    from shuolee_portrait import official_sequence as seq, stable_selection

    manifest = read(RUN / 'data/inputs.json')
    cfg = read(RUN / 'config.json')
    info = read(RUN / 'run.json')
    fit = read(RUN / 'fit/official-fit-result.json')
    common = read(RUN / 'fit/shared-body.json')
    selection = read(RUN / 'fit/selection.json')
    scope = stable_selection.selection_scope(dict(rows=manifest['records'], source_manifest=manifest),
                                             arc=cfg['capture_layout']['arc'], capture_layout=cfg['capture_layout'])
    need(fit['selection'] == selection and FRAME in scope['fit_frame_ids']
         and FRAME not in scope['retained_holdouts'] and not selection['reasons'][FRAME]['screen_flags'],
         'Frozen frame or holdout policy differs.')
    row = next(dict(r) for r in manifest['records'] if r['frame'] == FRAME)
    need(row['source_frame_index'] == 160 and row['wh'] == [2704, 1520],
         'Original native frame identity differs.')
    for key in ('image', 'mask'):
        row[key] = str(windows_path(row[key]))
        need(sha(Path(row[key])) == row[{'image': 'source_sha256', 'mask': 'mask_sha256'}[key]],
             f'Windows source boundary changed {key}.')
    with np.load(RUN / 'data/observations.npz', allow_pickle=False) as obs:
        names = obs['frame_ids'].astype(str).tolist()
        slots = [names.index(frame) for frame in scope['fit_frame_ids']]
        targets, confidence = seq.official.adapt_rtmw_observations(
            *(obs[key][slots] for key in ('keypoints', 'scores', 'boxes', 'flip_disagreement')))
    need(confidence == fit['source']['confidence'], 'Native observation normalization drifted before fitting.')
    target = targets[scope['fit_frame_ids'].index(FRAME):scope['fit_frame_ids'].index(FRAME)+1]
    x, y = target[..., 0], target[..., 1]
    outside = (x < -.5) | (x >= row['wh'][0]-.5) | (y < -.5) | (y >= row['wh'][1]-.5)
    target[..., 2][outside] = 0
    K = np.asarray(row['K_integer_centers'], dtype=np.float64)
    beta = np.asarray(common['betas'], dtype=np.float32).reshape(1, 10)
    latent = np.asarray(common['pose_latent'], dtype=np.float32).reshape(1, 32)
    official = windows_path(cfg['official_smplifyx']['source_root'])
    assets = windows_path(info['assets'])
    receipt = read(RUN / 'fit/frames' / FRAME / 'result.json')
    with np.load(RUN / 'fit/frames' / FRAME / 'winner-camera-space.npz', allow_pickle=False) as winner:
        old = dict(winner)
    need(exact(K, old['K_integer_centers']) and exact(beta, old['betas']) and receipt['orientation_count'] == 1,
         'Original camera, shape or orientation differs.')
    expected_counts = [p['monitor_closure_calls'] for p in prior['records'][FRAME]['phases']]
    plan = dict(state='preregistered', frame=FRAME, source_frame_index=160, source_pins={str(k): v for k, v in pins.items()},
        source_commit=head, expected_stage_closure_counts=expected_counts,
        shared_target_sha256=reference['shared_body_target_sha256'], workers=1, threads=1, max_fit_seconds=75,
        official_source_changed=False, gpu_calls=0, gaussian_updates=0, cold_to_export_measured=False,
        environment=dict(python=sys.version, torch=torch.__version__, torch_source_commit=torch.version.git_version,
                         cpu_capability=torch.backends.cpu.get_cpu_capability(), opencv=cv2.__version__,
                         numpy=np.__version__, omegaconf=omegaconf.__version__, yaml=yaml.__version__,
                         tqdm=tqdm.__version__, torch_build=torch.__config__.show()))
    save(output / 'preregistration.json', plan)
    return seq, stable_selection, plan, row, target, K, beta, latent, official, assets, receipt, old, selection


def run(output, progress):
    seq, screening, plan, row, target, K, beta, latent, official, assets, receipt, old, selection = prepare(output)
    seq.rt.configure(assets)
    seq._init_cpu_fit(official, assets)
    _, _, vposer, _ = seq._CPU_FIT
    pose = seq.official.decode_shared_body_pose(vposer, latent, device='cpu')
    actual = sha256(pose.tobytes()).hexdigest()
    progress['shared_target_sha256'] = actual
    progress['reference_shared_target_sha256'] = plan['shared_target_sha256']
    progress['shared_target_exact'] = actual == plan['shared_target_sha256']
    if not progress['shared_target_exact']:
        return dict(state='rejected_before_fit_shared_target_numeric_drift', **progress)
    loader = seq.official.load_official
    counts = []
    started = perf_counter()

    def observed_load(path):
        utils, prior, fitting, optimizer = loader(path)
        monitor = fitting.FittingMonitor
        create = monitor.create_fitting_closure

        def observed_create(self, *args, **kwargs):
            closure = create(self, *args, **kwargs)
            slot = len(counts)
            need(slot < len(plan['expected_stage_closure_counts']), 'Unexpected orientation or extra stage.')
            counts.append(0)

            def observed_closure(*call_args, **call_kwargs):
                counts[slot] += 1
                need(perf_counter() - started < plan['max_fit_seconds']
                     and counts[slot] <= plan['expected_stage_closure_counts'][slot],
                     'Fit exceeded the fixed time or the archived stock phase count.')
                return closure(*call_args, **call_kwargs)
            return observed_closure

        monitor.create_fitting_closure = observed_create
        return utils, prior, fitting, optimizer

    progress['fit_calls'] = 1
    with patch.object(seq.official, 'load_official', new=observed_load):
        arrays, info = seq._fit_cpu_job((row, output / FRAME, target, K, beta, True, pose))
    progress['fit_seconds'] = perf_counter() - started
    progress['stage_closure_counts'] = counts
    report = seq.rt.read(info['path'])
    seq.check_result(report, row)
    scores, flags, _, _ = screening._candidate(row, arrays, target[0], K, beta, None, report, True)
    archive_scores = selection['reasons'][FRAME]['scores']
    different_arrays = sorted(key for key in arrays.keys() & old.keys() if not exact(arrays[key], old[key]))
    different_report = sorted(key for key in report.keys() | receipt.keys() if report.get(key) != receipt.get(key))
    same = (counts == plan['expected_stage_closure_counts'] and arrays.keys() == old.keys()
            and not different_arrays and not different_report and flags == [] and scores == archive_scores)
    for path, expected in plan['source_pins'].items():
        need(sha(Path(path)) == expected, f'Frozen source changed: {path}')
    return dict(state='windows_cpu_exact_stock_parity' if same else 'windows_cpu_numeric_drift',
                **progress, exact_archive_parity=same, differing_arrays=different_arrays,
                differing_report_fields=different_report, screen_flags=flags, score_exact=scores == archive_scores,
                source_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    need(output.parent == OUTPUT.parent and not output.exists(), 'Use a new isolated external evidence directory.')
    output.mkdir()
    progress = dict(frame=FRAME, fit_calls=0, gpu_calls=0, gaussian_updates=0)
    try:
        result = run(output, progress)
    except ModuleNotFoundError as exc:
        result = dict(state='dependency_unavailable', error=repr(exc), **progress)
    except BaseException as exc:
        save(output / 'failure.json', dict(state='preflight_stopped', error=repr(exc), **progress))
        raise
    save(output / 'result.json', result)
    print(json.dumps(result), flush=True)
    if result['state'] != 'windows_cpu_exact_stock_parity':
        sys.exit(2)


if __name__ == '__main__':
    main()
