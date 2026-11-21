"""Measure original 006 shared-body fitting with fresh 4- and 6-worker WSL pools."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
from hashlib import sha256
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys
from time import perf_counter


LAB = Path('/opt/gs-slam-lab')
RUN = LAB/'shuolee-smpl-placement-006/run'
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-body-placement')
LOCK = Path('/mnt/d/Projects/Shuolee-smpl-view-coverage/.gpu-task.lock')
COMMIT = '28ea12f5a89511770213689a52ebf0ecb714d1d0'
PINNED = {
    RUN.parent/'preregistration.json': 'fd4a793f1ff1a58bc20d4117e007d25ebea24f55e12ecdde86f1d8bc24e96710',
    RUN/'config.json': 'c7484fffaaa74680e0fa5034e34cf93f9d84fa67d06dbfa8908d21256654a6f8',
    RUN/'fit/official-fit-result.json': '5b8e6d6c471b0f5886ba74201209acc07c84a3aa2e3299a38d79863a6ca9a65b',
}
THREAD_ENV = dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
os.environ.update(THREAD_ENV)
sys.path.insert(0, str(PROJECT/'src'))

import numpy as np
from shuolee_portrait import official_sequence as seq, stable_selection


def need(condition, message):
    if not condition:
        raise ValueError(message)


def exact(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def prepare():
    rt = seq.rt
    seq.check_bound(PINNED)
    launch, cfg, fit = (rt.read(path) for path in PINNED)
    run = rt.read(RUN/'run.json')
    need(rt.PROJECT == PROJECT and launch['candidate_commit'] == COMMIT
         and rt.read(RUN/'status.json')['state'] == 'completed' and fit['fitted_candidate_frames'] == 52,
         'The archived 006 original fitting run is not bound to this benchmark.')
    source_pins = {Path(name): digest for name, digest in fit['source']['input_files'].items()}
    source_pins.update({PROJECT/name: digest for name, digest in launch['code_sha256'].items()})
    selection, shared = (RUN/'fit'/name for name in ('selection.json', 'shared-body.json'))
    need(rt.read(selection) == fit['selection'] and rt.sha(shared) == fit['shared_body_sha256'],
         'Archived selection or common body receipt changed.')
    common = rt.read(shared)
    beta = np.asarray(common['betas'], np.float32).reshape(1, 10)
    latent = np.asarray(common['pose_latent'], np.float32).reshape(1, 32)
    official = Path(cfg['official_smplifyx']['source_root'])
    assets = Path(run['assets'])
    asset_rows = rt.read(PROJECT/'configs/server-assets.json')['files']
    for row in asset_rows:
        if row['path'].startswith(('smplx/', 'vposer/')):
            source_pins[assets/row['path']] = row['sha256']
    source_pins.update({official/name: digest for name, digest in seq.official.PINNED.items()})
    seq.check_bound(source_pins)
    inputs = seq.load_inputs(RUN/'data/inputs.json', RUN/'data/observations.npz',
        camera_length_contract=cfg['camera_length_contract'], reference_frame=fit['selection']['reference_frame'],
        capture_layout=cfg['capture_layout'])
    scope = inputs['scope']
    ids = scope['fit_frame_ids']
    holds = scope['retained_holdouts']
    need(len(ids) == 52 and holds == launch['expected_holdouts'] and not set(holds) & set(ids)
         and inputs['confidence'] == fit['source']['confidence'], 'Frozen 006 candidate cohort or q90 changed.')
    cohort_indices = [(2*i+1)*len(ids)//24 for i in range(12)]
    rows = [inputs['rows'][inputs['ids'].index(ids[i])] for i in cohort_indices]
    need([row['source_frame_index'] for row in rows] == sorted(row['source_frame_index'] for row in rows),
         'Expected source-ordered candidate samples.')
    archived = {}
    targets = set()
    assembled_path = RUN/'fit/fitted.npz'
    assembled = load_arrays(assembled_path)
    selected_ids = assembled['frame_ids'].tolist()
    pins = PINNED | source_pins | {selection: rt.sha(selection), shared: fit['shared_body_sha256'],
                                   assembled_path: fit['fit_sha256'],
                                   Path(__file__).resolve(): rt.sha(__file__)}
    for row in rows:
        frame = row['frame']
        slot = inputs['ids'].index(frame)
        target = inputs['targets'][slot:slot+1]
        folder = RUN/'fit/frames'/frame
        array_file, result_file = folder/'winner-camera-space.npz', folder/'result.json'
        arrays, report = load_arrays(array_file), rt.read(result_file)
        seq.check_result(report, row)
        need(exact(arrays['betas'], beta) and np.array_equal(arrays['K_integer_centers'], inputs['K'][slot])
             and report['freeze_betas'] is True and report['body_placement_parameters'] == 3,
             f'Archived shared-body replay contract differs: {frame}')
        if frame in selected_ids:
            selected_slot = selected_ids.index(frame)
            need(exact(assembled['observed_native_keypoints'][selected_slot:selected_slot+1], target),
                 f'Archived assembled detector observations differ: {frame}')
        prior = fit['selection']['reasons'][frame]
        scores, flags, _, _ = stable_selection._candidate(row, arrays, target[0], inputs['K'][slot],
                                                            beta, None, report, True)
        need(scores == prior['scores'] and flags == prior['screen_flags'], f'Archived pose screen differs: {frame}')
        archived[frame] = (arrays, report, flags)
        targets.add(report['shared_body_target_sha256'])
        pins.update({path: rt.sha(path) for path in (array_file, result_file)})
        pins[Path(row['image'])] = row['source_sha256']
        pins[Path(row['mask'])] = row['mask_sha256']
    need(len(targets) == 1, 'Archived candidate frames do not have one shared body target.')
    plan = dict(scope='CPU shared-body fit throughput only. Each arm uses a fresh spawn pool; model initialization and teardown are timed.',
                source_commit=COMMIT, source=str(RUN), protected_holdouts=holds, frame_ids=[row['frame'] for row in rows],
                source_frame_indices=[row['source_frame_index'] for row in rows], candidate_order_indices=cohort_indices,
                archived_orientation_counts={frame: record[1]['orientation_count'] for frame, record in archived.items()},
                worker_counts=[4, 6], torch_threads_per_worker=1, environment=THREAD_ENV, start_method='spawn',
                shared_body_target_sha256=targets.pop(), cold_to_export_measured=False,
                source_pins={str(path): digest for path, digest in pins.items()})
    seq.check_bound(pins)
    return plan, inputs, rows, beta, latent, official, assets, archived


def guard_machine():
    need('microsoft' in Path('/proc/sys/kernel/osrelease').read_text().lower(), 'This timing benchmark is WSL-only.')
    available = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    need(available >= 8 * 1024**2 and len(os.sched_getaffinity(0)) >= 6,
         'At least 8 GiB available WSL memory and six vCPUs are required.')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            args = (proc/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        need(not (b'train' in args and any(arg.endswith(b'/run_smplx.py') for arg in args)),
             'A reconstruction command is still live.')


def measure(prepared, output):
    import torch

    plan, inputs, rows, beta, latent, official, assets, archived = prepared
    seq.rt.configure(assets)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _, vposer, _ = seq.official.create_models(official_root=official, assets=assets, device='cpu', static_topology=False)
    pose = seq.official.decode_shared_body_pose(vposer, latent, device='cpu')
    need(sha256(pose.tobytes()).hexdigest() == plan['shared_body_target_sha256'], 'Shared target replay differs.')
    del vposer
    output.mkdir(parents=True, exist_ok=False)
    plan['runtime'] = dict(python=sys.version, numpy=np.__version__, torch=torch.__version__,
                           wsl_vcpus=len(os.sched_getaffinity(0)), environment_prefix=sys.prefix)
    seq.rt.save(output/'preregistration.json', plan)
    times, arms, slot_by_id = {}, {}, {frame: i for i, frame in enumerate(inputs['ids'])}
    for workers in plan['worker_counts']:
        arm = output/f'workers-{workers}'
        arm.mkdir()
        jobs = [(row, arm/row['frame'], inputs['targets'][slot_by_id[row['frame']]:slot_by_id[row['frame']]+1],
                 inputs['K'][slot_by_id[row['frame']]], beta, True, pose) for row in rows]
        start = perf_counter()
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'),
                                 initializer=seq._init_cpu_fit, initargs=(official, assets)) as pool:
            completed = list(pool.map(seq._fit_cpu_job, jobs))
        times[workers] = perf_counter()-start
        records = {}
        for row, (arrays, receipt) in zip(rows, completed, strict=True):
            frame = row['frame']
            need(frame == receipt['frame_id'] and seq.rt.sha(receipt['path']) == receipt['sha256'],
                 f'Worker receipt identity differs: {frame}')
            report = seq.rt.read(receipt['path'])
            seq.check_result(report, row)
            slot = slot_by_id[frame]
            _, flags, _, _ = stable_selection._candidate(row, arrays, inputs['targets'][slot], inputs['K'][slot],
                                                           beta, None, report, True)
            records[frame] = (arrays, report, flags)
        arms[workers] = records
        print(json.dumps(dict(workers=workers, fits=len(completed), seconds=times[workers])), flush=True)
    seq.check_bound(plan['source_pins'])
    comparisons = {}
    for frame in plan['frame_ids']:
        comparisons[frame] = {}
        for label, left, right in (('four_vs_six', arms[4][frame], arms[6][frame]),
                                   ('four_vs_archive', arms[4][frame], archived[frame]),
                                   ('six_vs_archive', arms[6][frame], archived[frame])):
            arrays, reports = left[0].keys() | right[0].keys(), left[1].keys() | right[1].keys()
            delta_arrays = sorted(key for key in arrays if key not in left[0] or key not in right[0]
                                  or not exact(left[0][key], right[0][key]))
            delta_reports = sorted(key for key in reports if key not in left[1] or key not in right[1]
                                   or left[1][key] != right[1][key])
            comparisons[frame][label] = dict(array_fields=len(arrays), differing_arrays=delta_arrays,
                differing_report_fields=delta_reports, pose_flags_equal=left[2] == right[2],
                exact=not delta_arrays and not delta_reports and left[2] == right[2])
    parity = all(record['exact'] for row in comparisons.values() for record in row.values())
    seq.rt.save(output/'result.json', dict(state='complete' if parity else 'parity_failed',
        preregistration_sha256=seq.rt.sha(output/'preregistration.json'), comparisons=comparisons,
        seconds_four=times[4], seconds_six=times[6], six_worker_speedup=times[4]/times[6],
        startup_included=True, first_arm_cache_order_not_counterbalanced=True, exact_parity=parity,
        cold_to_export_measured=False, gaussian_training_started=False, quality_accepted=False))
    need(parity, 'Exact worker/archive parity failed. See the benchmark result.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'benchmark'))
    parser.add_argument('--output', type=Path, default=LAB/'shuolee-smpl-fit-workers006')
    args = parser.parse_args()
    output = args.output.resolve()
    need(not output.exists() and not output.is_relative_to(RUN.parent) and not output.is_relative_to(PROJECT),
         'Use a new external output directory.')
    prepared = prepare()
    if args.mode == 'check':
        plan = prepared[0]
        print(json.dumps(dict(state='frozen_inputs_verified', fits_started=0, gpu_calls=0,
                              frame_ids=plan['frame_ids'], source_frame_indices=plan['source_frame_indices'],
                              orientation_counts=list(plan['archived_orientation_counts'].values()),
                              protected_holdouts=plan['protected_holdouts'], pins=len(plan['source_pins']))))
        return
    with LOCK.open('rb') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard_machine()
        measure(prepared, output)


if __name__ == '__main__':
    main()
