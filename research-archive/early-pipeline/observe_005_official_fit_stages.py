"""Observe the frozen 005 stock/seed A/B stages without changing either fit."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
from multiprocessing import get_context
import os
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import probe_005_nearest_admitted_latent as ab
import numpy as np


seq, screen, need = ab.seq, ab.screen, ab.need
OUTPUT = ab.LAB / 'shuolee-smpl-coverage-005-nearest-admitted-stages'
ARMS = ('stock-zero', 'nearest-admitted')
ARCHIVE = {
    'result.json': '5966404eee12b8e74fb9f860c079eee9dc03868e468c6c4bddbae07c38f8ea3f',
    'preregistration.json': '165f938e25de6f12b37a1048f369771a85191b5d891fab67dcf7dded229ab422',
}
SEEDED_WINNERS = {
    'frame_000013': '63891abe8945714fad565367e0490dad2428df06b4fa0ccffa321a4a60d0df0c',
    'frame_000022': 'ab008ecf841b51b4e21bd91bd5e9546b01a1a94a0a44bdafb5dd8e59490919dc',
    'frame_000020': 'f5848b0b1eedb4eff18c5f91d69b99644366d94018fd1d3495d7ae1bf684078e',
}
REGIONS = dict(body=screen.BODY, feet=screen.FEET, left_hand=np.arange(25, 46),
               right_hand=np.arange(46, 67), face=screen.FACE)


def prepare():
    pins = {ab.OUTPUT / path: digest for path, digest in ARCHIVE.items()}
    seq.check_bound(pins)
    result = seq.rt.read(ab.OUTPUT / 'result.json')
    need(result['state'] == 'paired_cpu_comparison_complete' and result['stock_exact']
         and result['preregistration_sha256'] == ARCHIVE['preregistration.json']
         and tuple((row['frame'], row['donor']) for row in result['records']) == ab.EXPECTED,
         'Expected the completed exact-parity 005 A/B and the original three targets.')
    for row in result['records']:
        frame = row['frame']
        for arm, key, winner in zip(ARMS, ('stock_receipt', 'seeded_receipt'),
                                   (ab.WINNERS, SEEDED_WINNERS), strict=True):
            folder = ab.OUTPUT / arm / frame
            receipt = row[key]
            need(Path(receipt['path']) == folder / 'result.json' and receipt['frame_id'] == frame,
                 f'Unexpected archived receipt: {arm}/{frame}')
            pins.update({folder / 'result.json': receipt['sha256'],
                         folder / 'winner-camera-space.npz': winner[frame]})
    seq.check_bound(pins)
    original, pairs, checked, beta, pose, official, assets = ab.prepare()
    need(original == seq.rt.read(ab.OUTPUT / 'preregistration.json'),
         'Frozen source plan differs from the executed A/B.')
    pins.update({Path(path): digest for path, digest in original['source_pins'].items()})
    pins[Path(__file__).resolve()] = seq.rt.sha(__file__)
    cfg = seq.official.load_official_config(official)
    schedules = {key: cfg[key] for key in (
        'hand_joints_weights', 'face_joints_weights', 'body_pose_prior_weights',
        'shape_weights', 'expr_weights', 'hand_pose_prior_weights',
        'jaw_pose_prior_weights', 'coll_loss_weights')}
    need(schedules['hand_joints_weights'] == [0., 0., 0., .1, 2.]
         and schedules['face_joints_weights'] == [0., 0., 0., 0., 2.],
         'The pinned activation schedule has changed.')
    plan = dict(state='preregistered', pairs=original['pairs'],
        protected_frames=original['protected_frames'], arms=list(ARMS),
        source_pins={str(path): digest for path, digest in pins.items()},
        stage_schedule=schedules, collision_active=False, shared_body_weight=30.,
        shared_body_target_sha256=original['shared_body_target_sha256'],
        scope='Observe original CPU camera and five body stages; same frozen data and A/B latent variants.',
        co_changes='The final body stage raises both hand and face data weights. '
                   'Any observed regression localizes chronology only. The configured collision weight '
                   'also changes, with collisions disabled by the frozen integration.',
        limits=dict(fit_calls=6, workers=3, threads_per_worker=1, gpu_calls=0, gaussian_updates=0))
    return plan, pairs, checked, beta, pose, official, assets


def observed_job(item):
    """Use the original monitor and exact closure object, copying CPU state after return."""
    snapshots = []
    original_loader = seq.official.load_official

    def observed_loader(path):
        utils, prior, fitting, optimizer_module = original_loader(path)
        monitor = fitting.FittingMonitor
        create = monitor.create_fitting_closure
        run = monitor.run_fitting
        last = {}

        def create_observed(self, *args, **kwargs):
            closure = create(self, *args, **kwargs)
            camera = kwargs['camera'] if 'camera' in kwargs else args[2]
            last.update(closure=closure, camera=camera, loss=kwargs.get('loss'),
                        weights=kwargs.get('joint_weights'), confidence=kwargs.get('joints_conf'))
            return closure

        def run_observed(self, optimizer, closure, params, body_model, **kwargs):
            need(closure is last['closure'], 'Original closure identity differs.')
            value = run(self, optimizer, closure, params, body_model, **kwargs)
            camera = last['camera']
            clone = lambda tensor: tensor.detach().cpu().numpy().copy()
            weights, confidence, loss = last['weights'], last['confidence'], last['loss']
            stage = len(snapshots) - 1
            need((stage == -1) == (weights is None and confidence is None and loss is None),
                 'Unexpected camera/body phase boundary.')
            snapshots.append(dict(stage=stage, parameters={name: clone(param)
                for name, param in body_model.named_parameters()},
                pose_embedding=clone(kwargs['pose_embedding']), camera_translation=clone(camera.translation),
                K=clone(camera.K), joint_weights=None if weights is None else clone(weights),
                joints_confidence=None if confidence is None else clone(confidence),
                weights={} if loss is None else {name: clone(value).tolist()
                    for name, value in loss.named_buffers(recurse=False)},
                rho=None if loss is None else float(loss.robustifier.rho),
                use_joints_conf=None if loss is None else bool(loss.use_joints_conf)))
            return value

        monitor.create_fitting_closure = create_observed
        monitor.run_fitting = run_observed
        return utils, prior, fitting, optimizer_module

    with patch.object(seq.official, 'load_official', new=observed_loader):
        arrays, receipt = ab.fit_job(item)
    need([state['stage'] for state in snapshots] == [-1, 0, 1, 2, 3, 4],
         'Expected camera stage and exactly one orientation with five body stages.')
    return arrays, receipt, snapshots


def replay(snapshots, case, body, vposer, gmof, arrays, output):
    import torch

    rows, saved = [], {}
    target = case['target']
    for state in snapshots:
        stage = state['stage']
        need(ab.exact(state['K'], arrays['K_compute_float32'])
             and ab.exact(state['parameters']['betas'], arrays['betas']), 'Stage shape or K drifted.')
        body.reset_params(**state['parameters'])
        with torch.no_grad():
            axis = vposer.decode(torch.from_numpy(state['pose_embedding']))['pose_body'].reshape(1, 63)
            body_out = body(body_pose=axis, return_verts=True, return_full_pose=True)
            translation = torch.from_numpy(state['camera_translation'])[:, None]
            joints = body_out.joints + translation
            q = joints @ torch.from_numpy(state['K']).T
            uv = q[..., :2] / q[..., 2:]
            joints, uv = joints.numpy(), uv.numpy()
            if stage == 4:
                need(ab.exact(axis.numpy(), arrays['body_pose_axis_angle'])
                     and ab.exact(joints, arrays['joints_camera'])
                     and ab.exact(uv, arrays['joints_projected_native'])
                     and ab.exact((body_out.vertices + translation).numpy(), arrays['vertices_camera']),
                     'Isolated last-stage replay differs from the exact official terminal output.')
            weighted_data = None
            if state['joint_weights'] is not None:
                weights = state['joint_weights']
                need(weights.shape == (1, 118) and state['joints_confidence'].shape == (1, 118)
                     and np.array_equal(state['joints_confidence'][0], target[:, 2]),
                     'Actual stage confidence differs from the original target.')
                effective = weights * state['joints_confidence'] if state['use_joints_conf'] else weights
                residual = torch.from_numpy(target[None, :, :2] - uv)
                weighted_data = (torch.from_numpy(effective[..., None]).square() *
                                 gmof(rho=state['rho'])(residual) * state['weights']['data_weight'] ** 2)
                weighted_data = weighted_data[0].sum(dim=-1).numpy()
        errors = np.linalg.norm(uv[0] - target[:, :2], axis=1)
        regions = {}
        for name, indices in REGIONS.items():
            supported = indices[target[indices, 2] > 0]
            residuals = errors[supported]
            regions[name] = dict(count=len(residuals),
                median_px=float(np.median(residuals)) if len(residuals) else None,
                max_px=float(residuals.max()) if len(residuals) else None,
                robust_data_component=float(weighted_data[indices].sum()) if weighted_data is not None else None,
                actual_joint_weights=np.unique(weights[0, indices]).tolist() if weighted_data is not None else None)
        arm = {name: dict(target_xy=target[index, :2].tolist(), confidence=float(target[index, 2]),
                          fitted_xy=uv[0, index].tolist(), residual_px=float(errors[index]))
               for name, index in (('shoulder', 2), ('elbow', 3), ('body_wrist', 4), ('hand_root', 46))}
        height = case['row']['initial_box_xyxy'][3] - case['row']['initial_box_xyxy'][1]
        rows.append(dict(stage='camera_init' if stage == -1 else stage,
            phase_has_body_data=weighted_data is not None, weights=state['weights'], regions=regions, right_arm=arm,
            elbow_minus_body_wrist_native_y_px=float(uv[0, 3, 1] - uv[0, 4, 1]),
            target_elbow_minus_body_wrist_native_y_px=float(target[3, 1] - target[4, 1]),
            fitted_wrist_minus_shoulder_camera_z_model_m=float(joints[0, 4, 2] - joints[0, 2, 2]),
            fitted_elbow_minus_shoulder_camera_z_model_m=float(joints[0, 3, 2] - joints[0, 2, 2]),
            original_body_max_over_height=regions['body']['max_px'] / height))
        tensors = dict(state['parameters'], K=state['K'], pose_embedding=state['pose_embedding'],
                       camera_translation=state['camera_translation'], joints_camera=joints, joints_projected_native=uv)
        if weighted_data is not None:
            tensors.update(joint_weights=state['joint_weights'], joints_confidence=state['joints_confidence'])
        saved.update({f'stage_{stage}_{key}': value for key, value in tensors.items()})
    np.savez_compressed(output, **saved)
    return rows


def probe(prepared, output):
    plan, pairs, checked, beta, pose, official, assets = prepared
    completed = {}
    for arm in ARMS:
        folder = output / arm
        folder.mkdir()
        jobs = [((checked[frame]['row'], folder / frame, checked[frame]['target'][None],
                  checked[frame]['K'], beta, True, pose),
                 None if arm == 'stock-zero' else checked[donor]['old']['pose_embedding'])
                for frame, donor in pairs]
        with ProcessPoolExecutor(max_workers=3, mp_context=get_context('spawn'),
                                 initializer=seq._init_cpu_fit, initargs=(official, assets)) as pool:
            results = list(pool.map(observed_job, jobs))
        for (frame, _), (arrays, receipt, snapshots) in zip(pairs, results, strict=True):
            previous = ab.OUTPUT / arm / frame
            archived = ab.arrays_at(previous / 'winner-camera-space.npz')
            report = seq.rt.read(receipt['path'])
            seq.check_result(report, checked[frame]['row'])
            need(report == seq.rt.read(previous / 'result.json') and arrays.keys() == archived.keys()
                 and all(ab.exact(value, archived[key]) for key, value in arrays.items()),
                 f'{arm}/{frame}: observer changed exact archived outputs.')
            completed[arm, frame] = arrays, snapshots
    body, vposer, _ = seq.official.create_models(official_root=official, assets=assets,
                                                device='cpu', static_topology=False)
    utils, _, _, _ = seq.official.load_official(official)
    records = {}
    for (arm, frame), (arrays, snapshots) in completed.items():
        path = output / arm / frame / 'stage-state.npz'
        stages = replay(snapshots, checked[frame], body.eval(), vposer, utils.GMoF, arrays, path)
        records[f'{arm}/{frame}'] = dict(exact_archived_terminal=True, stages=stages,
                                        stage_arrays=str(path), stage_arrays_sha256=seq.rt.sha(path))
    seq.check_bound(plan['source_pins'])
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'probe'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    need(output.parent == ab.LAB and not output.exists(), 'Use a new isolated output.')
    prepared = prepare()
    plan = prepared[0]
    if args.mode == 'check':
        print(json.dumps(dict(state='005_observer_inputs_verified', frames=[p['target'] for p in plan['pairs']],
                              pins=len(plan['source_pins']), fit_calls=0, gpu_calls=0)))
        return
    with ab.LOCK.open('rb') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        available = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
        need(available >= 8 * 1024**2 and len(os.sched_getaffinity(0)) >= 3,
             'Need 8 GiB free WSL memory and three CPU cores.')
        output.mkdir()
        seq.rt.save(output / 'preregistration.json', plan)
        start = perf_counter()
        try:
            records = probe(prepared, output)
            seq.rt.save(output / 'result.json', dict(state='stage_observation_complete', records=records,
                exact_stock_and_seed_parity=True, fit_calls=6, gaussian_updates=0, gpu_calls=0,
                source_unchanged=True, objective_changed=False, physical_depth_validated=False,
                elapsed_seconds=perf_counter()-start, quality_accepted=False))
            print('005_STAGE_OBSERVATION_COMPLETE', len(records), flush=True)
        except BaseException as exc:
            seq.rt.save(output / 'failure.json', dict(state='failed', error=str(exc),
                                                     elapsed_seconds=perf_counter()-start))
            raise


if __name__ == '__main__':
    main()
