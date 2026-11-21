"""Bounded CPU test of the 008 shared-body rotation weight with frozen inputs."""

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

os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
import numpy as np
from scipy.spatial.transform import Rotation

import probe_008_hidden_feet_fit as feet


LAB = Path('/opt/gs-slam-lab')
RUN = LAB / 'shuolee-smpl-visibility-008/run'
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-source-visibility')
OUTPUT = LAB / 'shuolee-smpl-visibility-008-shared-body-weight'
UPPER = LAB / 'shuolee-smpl-visibility-008-upperbody'
LOCK = Path('/tmp/shuolee-native122-gpu.lock')
PINNED = {
    RUN.parent / 'preregistration.json': '06b7e045856c042efea1d235e34efc1d1e81b71381c7e59545ffba40955a5e1c',
    RUN.parent / 'registration-correction.json': 'd3f8cb1f8bee2c4c910eddc23fe530a285b52cffe2b718881b575557973bf6e7',
    RUN / 'result.json': '4f2605fca0b011798308fa414de604c3c3dc00dfcb5d4779d31d5338dc89aa84',
    RUN / 'config.json': '6ae1e685be00858eb0083b4dbff005a7b7a62ed3ca29ee2fa13011b706d397b8',
    RUN / 'fit/official-fit-result.json': 'ead9039816741cb6c780ecc888fc139c6f826c7580a851344e12bfca45925ef0',
    RUN / 'fit/selection.json': '44e85e52a01642cda7c294f9afbb8ef4520f3e31b53cf218fadca0dcf5c0950d',
    RUN / 'fit/shared-body.json': '72672b67af8b3fcd6865df2b5343cdd900b6ddc228bc157c8522cd17abbeffc9',
    UPPER / 'shoulder-span-readonly.json': 'e3f794f28f35f296cdb05db094224044838504e0c2dd2e8a68572eb9c6094480',
}
PROTECTED = ['frame_000015', 'frame_000043', 'frame_000071', 'frame_000098']
EXPECTED = ('frame_000011', 'frame_000106', 'frame_000004', 'frame_000052')
TARGET_SHA = '2da253e55a6fded5601e722098e85bae5f2f1d753941f160c629d28e184b992d'
SHOULDERS = np.array([2, 5])
sys.path.insert(0, str(PROJECT / 'src'))
from shuolee_portrait import official_sequence as seq, smplifyx_runtime as runtime, stable_selection as screen


def need(condition, message):
    if not condition:
        raise ValueError(message)


def exact(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def choose(selection, audited, layout):
    reasons = selection['reasons']

    def middle(items):
        ordered = sorted(items, key=lambda frame: reasons[frame]['source_index'])
        need(len(ordered) > 0, 'A preregistered cohort stratum is empty.')
        return ordered[(len(ordered) - 1) // 2]

    def shoulder_middle(bounds):
        start, stop = bounds
        return middle([row['frame'] for row in audited['rows']
                       if start <= reasons[row['frame']]['source_index'] <= stop])

    return dict(
        shoulder_early=shoulder_middle(layout['front_source_ranges'][0]),
        shoulder_late=shoulder_middle(layout['front_source_ranges'][1]),
        front_admitted=middle([f for f in selection['selected_frames'] if reasons[f]['side'] == 'front']),
        rear_admitted=middle([f for f in selection['selected_frames'] if reasons[f]['side'] == 'rear']),
    )


def prepare():
    seq.check_bound(PINNED)
    cfg = seq.rt.read(RUN / 'config.json')
    fit = seq.rt.read(RUN / 'fit/official-fit-result.json')
    selection = seq.rt.read(RUN / 'fit/selection.json')
    shared = seq.rt.read(RUN / 'fit/shared-body.json')
    terminal = seq.rt.read(RUN / 'result.json')
    need(seq.rt.PROJECT == PROJECT and fit['selection'] == selection and fit['shared_identity']
         and fit['new_fitting_started'] and fit['fitted_candidate_frames'] == 60
         and fit['shared_body_sha256'] == PINNED[RUN / 'fit/shared-body.json']
         and terminal['state'] == 'completed' and terminal['steps'] == 10_000 and terminal['points'] == 58_253,
         'Expected the frozen 008 visibility delivery.')
    beta = np.asarray(shared['betas'], dtype=np.float32).reshape(1, 10)
    latent = np.asarray(shared['pose_latent'], dtype=np.float32).reshape(1, 32)
    official = Path(cfg['official_smplifyx']['source_root'])
    assets = Path(seq.rt.read(RUN / 'run.json')['assets'])
    manifest = seq.rt.read(PROJECT / 'configs/server-assets.json')
    launch = seq.rt.read(RUN.parent / 'preregistration.json')
    pins = PINNED | {Path(p): digest for p, digest in fit['source']['input_files'].items()}
    pins.update({PROJECT / p: digest for p, digest in launch['code_sha256'].items()})
    pins.update({official / p: digest for p, digest in seq.official.PINNED.items()})
    pins.update({assets / row['path']: row['sha256'] for row in manifest['files']
                 if row['path'].startswith(('smplx/', 'vposer/'))})
    pins[RUN / 'fit/fitted.npz'] = fit['fit_sha256']
    inputs = seq.load_inputs(RUN / 'data/inputs.json', RUN / 'data/observations.npz',
                             camera_length_contract=cfg['camera_length_contract'],
                             reference_frame=selection['reference_frame'], capture_layout=cfg['capture_layout'])
    scope = inputs['scope']
    holds = scope['retained_holdouts']
    need(len(inputs['ids']) == 111 and len(scope['fit_frame_ids']) == 60
         and holds == selection['retained_holdouts'] == launch['expected_holdouts'] == PROTECTED
         and inputs['confidence'] == fit['source']['confidence']
         and all(not inputs['targets'][inputs['ids'].index(f)].any() for f in holds),
         'The frozen confidence recipe, fit roster or protected frames differ.')
    audit = seq.rt.read(UPPER / 'shoulder-span-readonly.json')
    need(audit['selection_sha256'] == PINNED[RUN / 'fit/selection.json']
         and audit['detector_npz_sha256'] == pins[RUN / 'data/observations.npz']
         and len(audit['rows']) == 18, 'Shoulder-source audit does not bind the same observations and selection.')
    by_frame = {row['frame']: row for row in audit['rows']}
    cohort = choose(selection, audit, cfg['capture_layout'])
    need(tuple(cohort.values()) == EXPECTED and not set(cohort.values()) & set(holds)
         and set(cohort.values()) <= set(scope['fit_frame_ids']), 'The four fixed training-side cases changed.')
    samples, rows = {}, []
    for role, frame in cohort.items():
        i = inputs['ids'].index(frame)
        row, target, K = inputs['rows'][i], inputs['targets'][i:i + 1], inputs['K'][i]
        winner_dir = RUN / 'fit/frames' / frame
        p, report_path = winner_dir / 'winner-camera-space.npz', winner_dir / 'result.json'
        old, report = feet.archive(p), seq.rt.read(report_path)
        seq.check_result(report, row)
        scores, flags, _, _ = screen._candidate(row, old, target[0], K, beta, None, report, True)
        reason = selection['reasons'][frame]
        need(scores == reason['scores'] and flags == reason['screen_flags'] and exact(old['betas'], beta)
             and report['shared_body_weight'] == 30. and report['shared_body_target_sha256'] == TARGET_SHA
             and report['shared_root_weight'] == 0. and report['body_placement_parameters'] == 3,
             f'Archived pose or shared target differs on {frame}.')
        if role.startswith('shoulder'):
            height = row['initial_box_xyxy'][3] - row['initial_box_xyxy'][1]
            worst_body = feet.worst(old['joints_projected_native'][0], target[0], screen.BODY, height)
            need(reason['status'] == 'pose_screen_rejection' and worst_body['worst_slot'] in SHOULDERS
                 and worst_body['worst_slot'] == by_frame[frame]['worst']
                 and seq.rt.sha(p) == by_frame[frame]['fit_winner_sha256']
                 and row['source_sha256'] == by_frame[frame]['sha256'],
                 f'Shoulder case changed on {frame}.')
        else:
            need(not flags and frame in selection['selected_frames'], f'Admitted control changed on {frame}.')
        pins.update({p: seq.rt.sha(p), report_path: seq.rt.sha(report_path),
                     Path(row['image']): row['source_sha256'], Path(row['mask']): row['mask_sha256']})
        rows.append((role, row, target, K, old, report, reason))
        samples[frame] = dict(role=role, source_index=row['source_frame_index'], source_bin=reason['source_bin'],
                              source_image_sha256=row['source_sha256'], mask_sha256=row['mask_sha256'],
                              target_sha256=sha256(target.tobytes()).hexdigest(), K_integer_centers=K.tolist())
    for source in (Path(__file__), Path(feet.__file__)):
        pins[source.resolve()] = seq.rt.sha(source)
    seq.check_bound(pins)
    import torch

    seq.rt.configure(assets)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _, vposer, _ = seq.official.create_models(
        official_root=official, assets=assets, device='cpu', static_topology=False)
    pose = seq.official.decode_shared_body_pose(vposer, latent, device='cpu')
    need(sha256(pose.tobytes()).hexdigest() == TARGET_SHA, 'Frozen target CPU decode differs from archived 008.')
    plan = dict(scope='Diagnostic only. Shared-body rotation penalty weight 30 to 0.',
                cohort_rule='Chronological lower median of shoulder-leading rejects in each audited front arc, '
                            'and lower median of selected frames per front/rear side.',
                cases=samples, frozen_body_target_sha256=TARGET_SHA, frozen_betas=beta.tolist(),
                frozen_q90=inputs['confidence']['per_joint_q90'], unchanged_ankle_foot_visibility_recipe=True,
                original_screen_unchanged=True, original_five_stage_optimizer=True,
                protected_frames=holds, source_pins={str(p): digest for p, digest in pins.items()},
                weights=[30., 0.], workers=4, threads_per_worker=1, no_model_source_edits=True)
    return plan, rows, beta, pose, official, assets


def weighted_fit(item):
    weight, job = item
    previous = runtime.SHARED_BODY_WEIGHT
    need(previous == 30. and weight in (30., 0.), 'Unexpected process-local weight.')
    try:
        runtime.SHARED_BODY_WEIGHT = weight
        return seq._fit_cpu_job(job)
    finally:
        runtime.SHARED_BODY_WEIGHT = previous


def run_arm(folder, rows, beta, pose, official, assets, weight):
    folder.mkdir()
    jobs = [(weight, (row, folder / row['frame'], target, K, beta, True, pose))
            for _, row, target, K, _, _, _ in rows]
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context('spawn'),
                             initializer=seq._init_cpu_fit, initargs=(official, assets)) as pool:
        outputs = list(pool.map(weighted_fit, jobs))
    records = []
    for (row_arrays, ref), (_, row, _, K, old, old_report, _) in zip(outputs, rows, strict=True):
        receipt = Path(ref['path'])
        report = seq.rt.read(receipt)
        need(seq.rt.sha(receipt) == ref['sha256'] and receipt.parent.parent == folder
             and ref['frame_id'] == row['frame'], 'Fit output identity differs.')
        if weight == 0.:
            # Check the common receipt contract, then verify the diagnostic weight separately.
            seq.check_result(dict(report, shared_body_weight=30.), row)
            need(report['last_evaluated_body_loss'] == 0., 'Nonzero regularizer at zero weight.')
        else:
            seq.check_result(report, row)
        need(report['shared_body_weight'] == weight and report['shared_body_pose_soft']
             and report['runtime_optimization'] is None
             and all(report[k] == old_report[k] for k in (
                'shared_body_target_sha256', 'shared_root_weight', 'body_placement_parameters',
                'body_placement_initial_m', 'orientation_count', 'initialization', 'camera_initialization',
                'source_hashes', 'official_commit'))
             and all(exact(row_arrays[k], old[k]) for k in ('betas', 'K_compute_float32', 'faces'))
             and np.array_equal(row_arrays['K_integer_centers'], K),
             'Frozen solver, starting placement, orientation branches, shape or native camera differs.')
        if weight == 30.:
            need(report == old_report and row_arrays.keys() == old.keys()
                 and all(exact(a, old[key]) for key, a in row_arrays.items()),
                 f'Stock CPU control failed byte-exact numerical and receipt parity on {row["frame"]}.')
        records.append((row_arrays, report, ref))
    return records


def compare(rows, baseline, treatment, beta):
    results = []
    for (role, row, target, K, old, _, reason), stock, test in zip(rows, baseline, treatment, strict=True):
        height = row['initial_box_xyxy'][3] - row['initial_box_xyxy'][1]
        fields = {}
        for label, (arrays, report, ref) in (('weight_30', stock), ('weight_0', test)):
            scores, flags, _, _ = screen._candidate(
                row, arrays, target[0], K, beta, (old['vertices_camera'].shape, old['faces']), report, True)
            if label == 'weight_30':
                need(scores == reason['scores'] and flags == reason['screen_flags'], 'Original screen parity failed.')
            proj = arrays['joints_projected_native'][0]
            fields[label] = dict(
                scores=scores, flags=flags, body=feet.worst(proj, target[0], screen.BODY, height),
                shoulders=feet.worst(proj, target[0], SHOULDERS, height),
                body_without_shoulders=feet.worst(proj, target[0], np.setdiff1d(screen.BODY, SHOULDERS), height),
                face=feet.worst(proj, target[0], screen.FACE, height),
                feet=feet.worst(proj, target[0], screen.FEET, height),
                min_depth_m=float(arrays['vertices_camera'][..., 2].min()),
                placement_initial_m=report['body_placement_initial_m'],
                placement_final_m=report['body_placement_final_m'], receipt=ref)
        a, b = stock[0], test[0]
        relative = Rotation.from_rotvec(b['global_orient'].reshape(3)) * Rotation.from_rotvec(a['global_orient'].reshape(3)).inv()
        results.append(dict(
            role=role, frame=row['frame'], source_index=row['source_frame_index'],
            source_sha256=row['source_sha256'], shoulder_span_x_px={
                key: float(array['joints_projected_native'][0, 5, 0] - array['joints_projected_native'][0, 2, 0])
                for key, array in (('weight_30', a), ('weight_0', b))},
            root_change_deg=float(np.degrees(relative.magnitude())),
            placement_change_m=float(np.linalg.norm(b['camera_translation'] - a['camera_translation'])),
            **fields))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'benchmark'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    need(output.parent == LAB and not output.exists(), 'Use a new isolated local diagnostic directory.')
    plan, rows, beta, pose, official, assets = prepare()
    if args.mode == 'check':
        print(json.dumps(dict(state='frozen_inputs_verified', frames=plan['cases'], gpu_calls=0,
                              fit_calls=0, protected_frames=PROTECTED, pins=len(plan['source_pins']))))
        return
    with LOCK.open('a+b') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        available = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
        need(available >= 8 * 1024**2 and len(os.sched_getaffinity(0)) >= 4,
             'At least 8 GiB available WSL memory and four vCPUs are required.')
        output.mkdir()
        seq.rt.save(output / 'preregistration.json', plan)
        started = perf_counter()
        try:
            a = run_arm(output / 'weight-30', rows, beta, pose, official, assets, 30.)
            print('weight-30: four original CPU outputs reproduced exactly', flush=True)
            b = run_arm(output / 'weight-0', rows, beta, pose, official, assets, 0.)
            records = compare(rows, a, b, beta)
            seq.check_bound(plan['source_pins'])
            seq.rt.save(output / 'result.json', dict(state='completed', elapsed_seconds=perf_counter()-started,
                preregistration_sha256=seq.rt.sha(output / 'preregistration.json'), stock_exact=True,
                records=records, protected_frames=PROTECTED, fit_calls=8, gpu_calls=0,
                gaussian_updates=0, model_changed=False, quality_accepted=False,
                scope='Four training-side frames only. Physical and held-out quality remain untested.'))
            print('weight-0: four CPU comparisons complete', flush=True)
        except BaseException as exc:
            seq.rt.save(output / 'failure.json', dict(state='failed', error=str(exc),
                                                     elapsed_seconds=perf_counter()-started))
            raise


if __name__ == '__main__':
    main()
