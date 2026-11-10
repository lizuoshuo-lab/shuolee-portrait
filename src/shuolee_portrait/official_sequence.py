"""Named-frame I/O around the official single-frame solver; no temporal optimizer."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from multiprocessing import get_context
from pathlib import Path
import hashlib
import io
import json
import math
import re
import time

import numpy as np
from PIL import Image

from . import official_smplifyx as official, runtime as rt
from .camera_world_units import normalize_camera_w2c


ANKLE_FOOT_TARGETS = {'left_foot': (14, 19, 20, 21), 'right_foot': (11, 22, 23, 24)}


def _source_visibility(spec, manifest, observations_sha256, bound):
    """Bind independently reviewed ankle/foot absence to exact native input rows."""
    if (not isinstance(spec, dict) or set(spec) != {'path', 'sha256'} or
            not isinstance(spec['path'], str) or not Path(spec['path']).is_absolute() or
            not isinstance(spec['sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', spec['sha256'])):
        raise ValueError('Visibility annotations require an absolute path and SHA-256.')
    path = Path(spec['path']).resolve()
    digest = spec['sha256']
    if not path.is_file() or rt.sha(path) != digest:
        raise ValueError('Visibility annotation file hash differs.')
    evidence = rt.read(path)
    if (not isinstance(evidence, dict) or evidence.get('actual_visual_review') is not True or
            evidence.get('source_video_sha256') != manifest['video_sha256'] or
            evidence.get('observations_sha256') != observations_sha256):
        raise ValueError('Visibility annotations must match the reviewed source video and detector observations.')
    source = {row['frame']: row['source_sha256'] for row in manifest['records']}
    records = evidence.get('records')
    if not isinstance(records, list) or len(records) != len(source):
        raise ValueError('Visibility annotations require a complete one-to-one native frame roster.')
    absent = {}
    for record in records:
        frame = record.get('frame') if isinstance(record, dict) else None
        if (not isinstance(frame, str) or frame not in source or frame in absent or
                record.get('source_sha256') != source[frame]):
            raise ValueError('Visibility annotations require a complete one-to-one native frame roster and source hashes.')
        groups = record.get('absent_groups')
        if (not isinstance(groups, list) or
                any(not isinstance(group, str) or group not in ANKLE_FOOT_TARGETS for group in groups) or
                len(set(groups)) != len(groups)):
            raise ValueError('Visibility annotations accept only unique left_foot and right_foot groups.')
        absent[frame] = [slot for group in groups for slot in ANKLE_FOOT_TARGETS[group]]
    bound[str(path)] = digest
    return absent


def checked_image(row):
    image = None
    for key, digest in [('image', 'source_sha256'), ('mask', 'mask_sha256')]:
        path = Path(row[key])
        if not path.is_absolute():
            raise ValueError('Image and mask paths must be absolute: ' + row['frame'])
        raw = path.read_bytes()
        if __import__('hashlib').sha256(raw).hexdigest() != row[digest]:
            raise ValueError(key + ' changed: ' + row['frame'])
        with Image.open(io.BytesIO(raw)) as im:
            if list(im.size) != row['wh']:
                raise ValueError(key + ' dimensions differ: ' + row['frame'])
            im.load()
            if key == 'image':
                image = np.asarray(im.convert('RGB'), dtype=np.float32) / 255
    return image


def load_inputs(frames_path, observations_path, *, camera_length_contract, reference_frame, capture_layout=None):
    """Join observations to the entire caller-owned roster by unique frame IDs."""
    frames_path, observations_path = map(lambda p: Path(p).resolve(), (frames_path, observations_path))
    bound = {str(p): rt.sha(p) for p in (frames_path, observations_path, Path(__file__).resolve(),
        Path(official.__file__).resolve(), Path(__file__).with_name('smplifyx_runtime.py').resolve(),
        Path(__file__).with_name('camera_world_units.py').resolve())}
    manifest = rt.read(frames_path)
    rows = manifest['records']
    ids = [r['frame'] for r in rows]
    if (not ids or len(set(ids)) != len(ids) or reference_frame not in ids or
            any(not isinstance(v, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', v) for v in ids)):
        raise ValueError('Expected unique safe frame IDs and an explicit reference in the roster.')
    with np.load(observations_path, allow_pickle=False) as f:
        obs = {key: f[key].copy() for key in ('frame_ids', 'keypoints', 'scores', 'boxes', 'flip_disagreement')}
    obs_ids = obs['frame_ids'].astype(str).tolist()
    if len(set(obs_ids)) != len(obs_ids) or len(obs_ids) != len(ids) or set(obs_ids) != set(ids):
        raise ValueError('Observation frame IDs must match the complete unique input roster.')
    order = [obs_ids.index(frame) for frame in ids]
    scope = None
    fit_slots = list(range(len(rows)))
    if capture_layout is not None:
        from . import stable_selection
        bound[str(Path(stable_selection.__file__).resolve())] = rt.sha(stable_selection.__file__)
        scope = stable_selection.selection_scope(dict(rows=rows, source_manifest=manifest),
            arc=capture_layout['arc'], capture_layout=capture_layout)
        fit_slots = [ids.index(frame) for frame in scope['fit_frame_ids']]
    fit_order = [order[i] for i in fit_slots]
    fitted_targets, confidence = official.adapt_rtmw_observations(
        *(obs[key][fit_order] for key in ('keypoints', 'scores', 'boxes', 'flip_disagreement')))
    if capture_layout is not None and 'visibility_annotations' in capture_layout:
        absent = _source_visibility(capture_layout['visibility_annotations'], manifest,
                                    bound[str(observations_path)], bound)
        for i, slot in enumerate(fit_slots):
            fitted_targets[i, absent[ids[slot]], 2] = 0
    targets = np.zeros((len(rows), *fitted_targets.shape[1:]), dtype=fitted_targets.dtype)
    targets[fit_slots] = fitted_targets
    K = np.asarray([r['K_integer_centers'] for r in rows], np.float64)
    if (K.shape != (len(rows), 3, 3) or not np.isfinite(K).all() or
            np.any(K[:, (0, 1), (0, 1)] <= 0) or np.any(K[:, 1, 0] != 0) or
            not np.array_equal(K[:, 2], np.tile([0., 0., 1.], (len(rows), 1)))):
        raise ValueError('Expected complete finite upper-triangular native K.')
    units = normalize_camera_w2c([r['w2c'] for r in rows], camera_length_contract)
    for i, row in enumerate(rows):
        if (len(row['wh']) != 2 or any(type(v) is not int or v <= 0 for v in row['wh']) or
                type(row['source_frame_index']) is not int or row['source_frame_index'] < 0):
            raise ValueError('Invalid frame dimensions or source index: ' + row['frame'])
        corner = K[i].copy(); corner[:2, 2] += .5
        if not np.array_equal(corner, row['K']):
            raise ValueError('Integer/corner K differs: ' + row['frame'])
        checked_image(row)
        if scope is not None:
            x, y = targets[i, ..., 0], targets[i, ..., 1]
            outside = (x < -.5) | (x >= row['wh'][0]-.5) | (y < -.5) | (y >= row['wh'][1]-.5)
            targets[i, ..., 2][outside] = 0
    check_bound(bound)
    return dict(scope=scope, rows=rows, ids=ids, targets=targets, K=K, units=units,
                reference_frame=reference_frame, bound=bound, confidence=confidence,
                frames_path=frames_path, source_manifest=manifest,
                source_reliability=obs['scores'][order, :23].sum(1) /
                (1. + obs['flip_disagreement'][order, :23].mean(1)))


def check_bound(bound):
    for path, digest in bound.items():
        if rt.sha(path) != digest:
            raise ValueError('Input or source changed: ' + str(path))


def read_shape_policy(policy):
    if not isinstance(policy, dict) or policy.get('mode') not in ('fit_reference', 'supplied'):
        raise ValueError('Explicit shape policy must be fit_reference or supplied.')
    if policy['mode'] == 'fit_reference':
        if set(policy) != {'mode'}:
            raise ValueError('fit_reference uses the official zero/mean initialization.')
        return None, {}
    if set(policy) != {'mode', 'path', 'sha256', 'key'}:
        raise ValueError('supplied shape requires path, sha256 and key.')
    path = Path(policy['path']).resolve()
    if rt.sha(path) != policy['sha256']:
        raise ValueError('Shared shape file hash differs.')
    with np.load(path, allow_pickle=False) as archive:
        shared = np.asarray(archive[policy['key']], dtype=np.float32).copy()
    if shared.shape not in ((10,), (1, 10)) or not np.isfinite(shared).all():
        raise ValueError('Expected one finite 10-vector for the entire sequence.')
    return shared.reshape(1, 10), {str(path): policy['sha256']}


def pelvis_basis(body):
    cpu = lambda value: value.detach().cpu().numpy().astype(np.float64)
    reg = cpu(body.J_regressor)[0]
    return (reg @ cpu(body.v_template), np.einsum('v,vck->ck', reg, cpu(body.shapedirs)),
            np.einsum('v,vck->ck', reg, cpu(body.expr_dirs)))


def root_to_world(arrays, E, root_template, root_shape_dirs, root_expr_dirs):
    """LBS pelvis pivot, not a guessed identity or origin translation.

    SMPL-X rotates its articulated body about shape-dependent J0. The old
    export field world_translation denotes that pelvis in world coordinates:
    world=(root_zero_vertices-J0) @ root_rotation.T + world_translation.
    """
    from scipy.spatial.transform import Rotation
    beta = arrays['betas'].reshape(10)
    expression = arrays['expression'].reshape(10)
    J0 = root_template + root_shape_dirs @ beta + root_expr_dirs @ expression
    global_R = Rotation.from_rotvec(arrays['global_orient'].reshape(3)).as_matrix()
    camera_t = arrays['camera_translation'].reshape(3)
    R, t = E[:3, :3], E[:3, 3]
    return (R.T @ global_R, (J0 + camera_t - t) @ R, J0)


STACK_FIELDS = ('vertices_camera', 'vertices_world', 'joints_camera', 'joints_projected_native',
    'root_rotation', 'world_translation', 'root_pivot_model', 'global_orient', 'camera_translation',
    'body_pose_axis_angle', 'pose_embedding', 'expression', 'jaw_pose', 'left_hand_pose',
    'right_hand_pose', 'leye_pose', 'reye_pose')


def assemble(inputs, winners, shared, basis):
    """Pure CPU camera-to-world conversion and the existing training array layout."""
    if set(winners) != set(inputs['ids']):
        raise ValueError('Exactly one official winner per named frame is required.')
    shared = np.asarray(shared, dtype=np.float32).reshape(1, 10)
    values, training_rows, faces = [], [], None
    for i, row in enumerate(inputs['rows']):
        a = {key: np.asarray(value).copy() for key, value in winners[row['frame']].items()}
        if 'frame_id' in a and str(a['frame_id'].item()) != row['frame']:
            raise ValueError('Winner frame identity differs.')
        if (not np.array_equal(a['betas'], shared) or
                not np.array_equal(a['K_integer_centers'], inputs['K'][i])):
            raise ValueError('Winner shared beta or complete K differs: ' + row['frame'])
        if (a['vertices_camera'].ndim != 3 or a['vertices_camera'].shape[0] != 1 or
                a['vertices_camera'].shape[2] != 3 or a['faces'].ndim != 2 or
                a['faces'].shape[1] != 3 or a['faces'].dtype.kind not in 'iu' or
                a['faces'].min() < 0 or a['faces'].max() >= a['vertices_camera'].shape[1]):
            raise ValueError('Invalid winner mesh geometry.')
        if faces is None:
            faces = a['faces'].copy()
        if not np.array_equal(faces, a['faces']):
            raise ValueError('Frame topology changed.')
        E = inputs['units'].w2c_meters[i]
        a['vertices_world'] = ((a['vertices_camera'][0].astype(np.float64)-E[:3, 3]) @ E[:3, :3])[None]
        rotation, translation, pivot = root_to_world(a, E, *basis)
        a.update(root_rotation=rotation[None], world_translation=translation[None], root_pivot_model=pivot[None])
        if any(a[key].shape[0] != 1 or not np.isfinite(a[key]).all() for key in STACK_FIELDS):
            raise ValueError('Invalid/nonfinite frame output.')
        values.append(a)
        corner = inputs['K'][i].copy(); corner[:2, 2] += .5
        training_rows.append(dict(row, K=corner.tolist(), K_integer_centers=inputs['K'][i].tolist(),
            w2c=E.tolist(), source_w2c=inputs['units'].source_w2c[i].tolist()))
    contract = dict(translation_unit='meters', meters_per_camera_unit=1.,
        basis=inputs['units'].contract['basis'], evidence='Already converted once from explicit source camera convention. '
        + inputs['units'].contract['evidence'])
    arrays = {key: np.concatenate([a[key] for a in values], axis=0) for key in STACK_FIELDS}
    arrays.update(faces=faces, frame_ids=np.asarray(inputs['ids']), reference_frame_id=np.asarray(inputs['reference_frame']),
        shared_betas=shared, betas=shared, source_frames=np.asarray([r['source_frame_index'] for r in training_rows], np.int64),
        image_sha256=np.asarray([r['source_sha256'] for r in training_rows]),
        mask_sha256=np.asarray([r['mask_sha256'] for r in training_rows]), K_integer_centers=inputs['K'],
        w2c=inputs['units'].w2c_meters, source_w2c=inputs['units'].source_w2c,
        observed_native_keypoints=inputs['targets'], camera_length_contract_json=np.asarray(json.dumps(contract)),
        source_camera_length_contract_json=np.asarray(json.dumps(dict(inputs['units'].contract))), world_scale=np.asarray(1.),
        root_translation_semantics=np.asarray('shape-dependent LBS pelvis in world; not affine origin'))
    return arrays, dict(records=training_rows, K_convention='K_integer_centers plus exactly 0.5 on cx/cy; renderer subtracts 0.5',
        camera_length_contract=contract, source_camera_length_contract=dict(inputs['units'].contract),
        pose_mode='named_official_per_frame_shared_shape', formal_calibration_accepted=False, quality_accepted=False)


def check_result(report, row):
    prior = report.get('shared_body_pose_soft') is True
    loss_valid = (report.get('upstream_loss_monitor_optimizer_unchanged') is True if not prior else
                  report.get('original_optimizer_and_monitor') is True and
                  report.get('shared_body_weight') == 30. and report.get('shared_root_weight') == 0. and
                  type(report.get('shared_body_loss_calls')) is int and report['shared_body_loss_calls'] > 0 and
                  isinstance(report.get('last_evaluated_body_loss'), (int,float)) and
                  np.isfinite(report['last_evaluated_body_loss']) and re.fullmatch(r'[0-9a-f]{64}', str(report.get('shared_body_target_sha256'))) is not None)
    if (report.get('state') != 'complete' or report.get('frame_id') != row['frame'] or
            report.get('source_frame_index') != row['source_frame_index'] or
            report.get('source_image_sha256') != row['source_sha256'] or
            report.get('official_commit') != official.COMMIT or report.get('gvhmr_used') is not False or
            not loss_valid or any(report.get(key) is not True for key in ('five_stages',
                                                       'translation_applied_once', 'full_K_integer_centers'))):
        raise ValueError('Unbound or incomplete official result: ' + row['frame'])


def write_results(root, inputs, winners, shared, basis, descriptors, *, execution_kind, extra=None):
    arrays, training = assemble(inputs, winners, shared, basis)
    check_bound(inputs['bound'])
    for row in inputs['rows']:
        checked_image(row)
    reports = []
    for item, row in zip(descriptors, inputs['rows']):
        if item['frame_id'] != row['frame'] or rt.sha(item['path']) != item['sha256']:
            raise ValueError('Official result descriptor differs.')
        report = rt.read(item['path']); check_result(report, row); reports.append(report)
    if len(descriptors) != len(inputs['rows']):
        raise ValueError('Missing official frame receipts.')
    target = root/'data/training-inputs.json'
    if target.exists() and rt.sha(target) != inputs['bound'][str(inputs['frames_path'])]:
        raise FileExistsError('Existing training inputs are not the unchanged preprocessing manifest.')
    with (root/'fit/fitted.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    rt.save(target, training)
    sources = [dict(path=str(p.resolve()), sha256=rt.sha(p)) for p in
               (Path(__file__), Path(official.__file__), Path(__file__).with_name('smplifyx_runtime.py'))]
    receipt = dict(state='complete', frames=len(inputs['rows']), execution_kind=execution_kind,
        source=dict(input_files=inputs['bound'], confidence=inputs['confidence'], gvhmr_used=False),
        frame_results=reports, official_frame_results=descriptors, sources=sources,
        fit_sha256=rt.sha(root/'fit/fitted.npz'), training_inputs_sha256=rt.sha(target),
        full_roster=True, reference_in_subset=True, shared_identity=True,
        world_export='(camera_vertices - normalized_E.translation) @ E.rotation; one source-E conversion only',
        new_fitting_started=execution_kind == 'fresh_official_sequence', quality_accepted=False,
        physical_quality_accepted=False, gaussian_training_started=False, **(extra or {}))
    rt.save(root/'fit/official-fit-result.json', receipt)
    return receipt


def _frame_fit(job, official_root, body, vposer, device):
    row, destination, keypoints, K, beta, freeze, shared_pose, *seed = job
    destination.mkdir()
    kwargs = {'static_topology': False} if device == 'cpu' else {}
    if shared_pose is not None:
        kwargs['shared_body_pose'] = shared_pose
    if seed:
        kwargs['initial_body_latent'], = seed
    with (destination/'fit.log').open('x', encoding='utf-8', buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            arrays, report = official.fit_frame(official_root=official_root, image=checked_image(row),
                keypoints=keypoints, K_integer_centers=K, body_model=body,
                vposer2=vposer, result_path=destination/'official-result.pkl', shared_betas=beta,
                freeze_betas=freeze, device=device, diff_path=destination/'input-output-adapter.diff', **kwargs)
    frame = row['frame']
    arrays['frame_id'] = np.asarray(frame)
    np.savez_compressed(destination/'winner-camera-space.npz', **arrays)
    report.update(state='complete', frame_id=frame, source_frame_index=row['source_frame_index'],
                  source_image_sha256=row['source_sha256'], quality_accepted=False)
    rt.save(destination/'result.json', report)
    return arrays, dict(frame_id=frame, path=str(destination/'result.json'), sha256=rt.sha(destination/'result.json'))


def _init_cpu_fit(official_root, assets):
    import torch
    global _CPU_FIT
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    body, vposer, _ = official.create_models(official_root=official_root, assets=assets,
                                            device='cpu', static_topology=False)
    _CPU_FIT = (official_root, body.eval(), vposer, 'cpu')


def _fit_cpu_job(job):
    try:
        return _frame_fit(job, *_CPU_FIT)
    except Exception as exc:
        raise RuntimeError(f'Official fit failed on {job[0]["frame"]}: {exc}') from exc


def _usable_latent(pose):
    latent = pose.get('pose_embedding')
    return (isinstance(latent, np.ndarray) and latent.shape == (1, 32) and
            latent.dtype == np.float32 and np.isfinite(latent).all() and np.any(latent))


def _missing_bin_seeds(inputs, selection, winners, shape_anchors):
    from . import stable_selection as stable

    reasons = selection['reasons']
    holdouts = set(selection['retained_holdouts'])
    rows = {row['frame']: row for row in inputs['rows']}
    admitted = [frame for frame, reason in reasons.items() if frame not in holdouts and
                reason['status'] in ('technically_valid', 'selected_bin_leader', 'selected_quota_fill')
                and _usable_latent(winners[frame])]
    for side, coverage in selection['coverage'].items():
        for bin_id in coverage['missing_source_bins']:
            candidates = [frame for frame, reason in reasons.items() if frame not in holdouts and
                          reason['status'] == 'pose_screen_rejection' and reason['side'] == side and
                          reason['source_bin'] == bin_id]
            item = dict(side=side, source_bin=bin_id)
            if not candidates:
                yield dict(item, status='no_pose_rejected_candidate')
                continue
            frame = min(candidates, key=lambda frame: (stable._body_rank(reasons[frame]), frame))
            item.update(frame_id=frame, source_frame_index=rows[frame]['source_frame_index'])
            if frame in shape_anchors:
                yield dict(item, seed_kind='same_frame_shape_anchor', seed_frame=frame)
                continue
            donors = [name for name in admitted if reasons[name]['side'] == side and
                      abs(reasons[name]['source_bin'] - bin_id) <= 2]
            if not donors:
                yield dict(item, status='no_nearby_admitted_seed')
                continue
            times = {name: rows[name].get('time_seconds') for name in [frame, *donors]}
            use_pts = all(type(value) in (int, float) and math.isfinite(value) for value in times.values())
            positions = times if use_pts else {name: rows[name]['source_frame_index'] for name in times}
            donor = min(donors, key=lambda name: (abs(positions[name]-positions[frame]),
                        rows[name]['source_frame_index'], name))
            yield dict(item, seed_kind='nearby_baseline_admitted', seed_frame=donor,
                       seed_distance_axis='native_pts_seconds' if use_pts else 'original_source_index')


_RECOVERY_MAX_GROWTH_PX = dict(head=.5, nonhead=1.)


def _recovery_protection(observed, before, after):
    from . import stable_selection as stable

    head_body = [0, 15, 16, 17, 18]
    regions = dict(head=np.r_[head_body, stable.FACE],
                   nonhead=np.r_[np.setdiff1d(stable.BODY, head_body), stable.FEET, 25:67])
    before_error, after_error = (np.linalg.norm(
        pose['joints_projected_native'][0].astype(np.float64) - observed[:, :2], axis=1)
        for pose in (before, after))
    result = {}
    for name, indices in regions.items():
        supported = indices[observed[indices, 2] > 0]
        growth = float(np.max(after_error[supported] - before_error[supported])) if len(supported) else None
        result[name] = dict(points=len(supported), max_growth_px=growth,
                            limit_px=_RECOVERY_MAX_GROWTH_PX[name],
                            passed=growth is None or growth <= _RECOVERY_MAX_GROWTH_PX[name])
    return result


def _recover_missing_bins(root, inputs, scope, selection, winners, descriptors, reports,
                          shared, shared_pose, shape_anchors, pool):
    from . import stable_selection as stable

    destination = root/'fit/recovery'
    destination.mkdir()
    rt.save(destination/'baseline-selection.json', selection)
    check_bound(inputs['bound'])
    planned = list(_missing_bin_seeds(inputs, selection, winners, shape_anchors))
    jobs = {}
    for item in planned:
        if 'seed_kind' not in item:
            continue
        frame, donor = item['frame_id'], item['seed_frame']
        anchor = item['seed_kind'] == 'same_frame_shape_anchor'
        seed, seed_receipt = shape_anchors[donor] if anchor else (winners[donor]['pose_embedding'], descriptors[donor])
        item.update(seed_receipt=seed_receipt, baseline_receipt=descriptors[frame],
                    seed_sha256=hashlib.sha256(seed.tobytes()).hexdigest())
        index = inputs['ids'].index(frame)
        jobs[frame] = (inputs['rows'][index], destination/frame, inputs['targets'][index:index+1],
                       inputs['K'][index], shared, True, shared_pose, seed)
    rt.save(destination/'plan.json', dict(one_attempt_per_missing_bin=True, protected_holdouts=scope['retained_holdouts'],
            supported_point_growth_limits_native_px=_RECOVERY_MAX_GROWTH_PX, bins=planned))

    futures = {frame: pool.submit(_fit_cpu_job, job) for frame, job in jobs.items()}
    promoted = []
    for item in planned:
        if 'seed_kind' not in item:
            continue
        frame = item['frame_id']
        row, _, keypoints, K, *_ = jobs[frame]
        try:
            candidate, descriptor = futures[frame].result()
            item['recovery_receipt'] = descriptor
            report = rt.read(descriptor['path'])
            check_result(report, row)
            stock = reports[frame]
            frozen_fields = ('shared_body_target_sha256', 'body_placement_initial_m', 'orientation_count')
            if (report.get('initial_body_latent_sha256') != item['seed_sha256'] or
                    any(report.get(field) != stock.get(field) for field in frozen_fields)):
                raise ValueError('Recovery seed or frozen fit initialization differs.')
            before = winners[frame]
            topology = before['vertices_camera'].shape, before['faces']
            scores, flags, _, _ = stable._candidate(row, candidate, keypoints[0], K, shared, topology, report, True)
            protection = _recovery_protection(keypoints[0], before, candidate)
            item.update(scores=scores, screen_flags=flags, protection=protection)
            if not flags and all(region['passed'] for region in protection.values()):
                winners[frame], descriptors[frame], reports[frame] = candidate, descriptor, report
                item['status'] = 'promoted'
                promoted.append(frame)
            else:
                item['status'] = 'rejected'
        except (RuntimeError, OSError, KeyError, TypeError, ValueError) as error:
            item.update(status='recovery_error', detail=str(error))
    check_bound(inputs['bound'])
    if promoted:
        selection = stable.select_frames(inputs, winners, scope=scope, reports=reports, shared_betas=shared,
                                         allow_empty=True)
    receipt = dict(attempted=len(jobs), promoted_frames=promoted, bins=planned,
                   remaining_missing_bins={side: details['missing_source_bins']
                                           for side, details in selection['coverage'].items()},
                   protected_holdouts=scope['retained_holdouts'], quality_accepted=False,
                   plan_sha256=rt.sha(destination/'plan.json'),
                   baseline_selection_sha256=rt.sha(destination/'baseline-selection.json'))
    path = destination/'result.json'
    rt.save(path, receipt)
    return selection, dict(path=str(path), sha256=rt.sha(path), attempted=len(jobs), promoted_frames=promoted)


def _selection_inputs(inputs, ordered, reference):
    slots = [inputs['ids'].index(frame) for frame in ordered]
    rows = [inputs['rows'][i] for i in slots]
    units = normalize_camera_w2c([row['w2c'] for row in rows], inputs['units'].contract)
    return dict(inputs, rows=rows, ids=ordered, units=units, reference_frame=reference,
                K=inputs['K'][slots], targets=inputs['targets'][slots])


def fit_named_sequence(*, frames_path, observations_path, root, official_root, assets,
                       camera_length_contract, reference_frame, shape_policy, capture_layout=None):
    """One reusable model pair, independent official fits, one explicitly chosen shape."""
    root = Path(root).resolve()
    if (root/'fit').exists():
        raise FileExistsError('Use a fresh run for a new shared-shape pass.')
    initial_reference = reference_frame
    if capture_layout is not None and initial_reference is None:
        initial_reference = rt.read(frames_path)['records'][0]['frame']
    inputs = load_inputs(frames_path, observations_path, camera_length_contract=camera_length_contract,
                         reference_frame=initial_reference, capture_layout=capture_layout)
    shared, bound = read_shape_policy(shape_policy)
    inputs['bound'].update(bound)
    scope = inputs['scope']
    if scope is not None:
        from . import stable_selection
    official.adapted_entry_source(official_root)
    (root/'fit/frames').mkdir(parents=True)
    started = time.monotonic()
    active = None
    stack = ExitStack()
    try:
        device = 'cpu' if scope is not None else 'cuda'
        model_options = {'device': 'cpu', 'static_topology': False} if scope is not None else {}
        body, vposer, meta = official.create_models(official_root=official_root, assets=assets, **model_options)
        basis = pelvis_basis(body)
        pool = None if scope is None else stack.enter_context(ProcessPoolExecutor(
            max_workers=6, mp_context=get_context('spawn'), initializer=_init_cpu_fit,
            initargs=(official_root, assets)))

        def fits(frame_ids, destination, beta, freeze, shared_pose=None):
            nonlocal active
            check_bound(inputs['bound'])
            jobs = []
            for frame in frame_ids:
                i = inputs['ids'].index(frame)
                jobs.append((rows[frame], destination/frame, inputs['targets'][i:i+1],
                             inputs['K'][i], beta, freeze, shared_pose))
            work = pool.map(_fit_cpu_job, jobs) if pool is not None else (
                _frame_fit(job, official_root, body, vposer, device) for job in jobs)
            for count, (arrays, desc) in enumerate(work, 1):
                active = desc['frame_id']
                rt.save(root/'fit/progress.json', dict(state='fitting', frame_id=active,
                    freeze_betas=freeze, phase=destination.name, frames_completed=count, total=len(jobs)))
                yield active, arrays, desc

        rows = {row['frame']: row for row in inputs['rows']}
        shared_pose = None
        extra = {'shape_policy': shape_policy, 'vposer': meta['vposer']}
        if scope is not None:
            samples = stable_selection.shape_anchors(inputs, scope)
            anchors = list(samples['front']) + list(samples['rear'])
            anchor_root = root/'fit/shape-anchors'
            anchor_root.mkdir()
            accepted, anchor_receipts, shape_anchor_seeds = [], [], {}
            for frame, arrays, desc in fits(anchors, anchor_root, shared, shared is not None):
                anchor_receipts.append(desc)
                if np.isfinite(arrays['vertices_camera']).all() and (arrays['vertices_camera'][..., 2] > .05).all():
                    accepted.append(arrays)
                    if _usable_latent(arrays):
                        shape_anchor_seeds[frame] = (arrays['pose_embedding'], desc)
            if not accepted:
                raise ValueError('No positive-depth official shape anchor.')
            if shared is None:
                shared = np.median([a['betas'] for a in accepted], axis=0).astype(np.float32).reshape(1, 10)
            latent = np.median([a['pose_embedding'] for a in accepted], axis=0).astype(np.float32).reshape(1, 32)
            if not np.isfinite(shared).all() or not np.isfinite(latent).all():
                raise ValueError('Invalid shared shape or pose target.')
            shared_pose = official.decode_shared_body_pose(vposer, latent, device=device)
            rt.save(root/'fit/shared-body.json', dict(betas=shared.tolist(), pose_latent=latent.tolist(),
                    anchor_groups=samples, anchor_receipts=anchor_receipts, valid_anchors=len(accepted)))
            extra['shared_body_sha256'] = rt.sha(root/'fit/shared-body.json')
        elif shared is None:
            i = inputs['ids'].index(reference_frame)
            active = reference_frame
            job = (rows[reference_frame], root/'fit/reference-initialization',
                   inputs['targets'][i:i+1], inputs['K'][i], None, False, None)
            initial, _ = _frame_fit(job, official_root, body, vposer, device)
            shared = np.asarray(initial['betas'], dtype=np.float32).reshape(1, 10).copy()
        winners, descriptors = {}, {}
        fit_ids = scope['fit_frame_ids'] if scope is not None else inputs['ids']
        for frame, arrays, desc in fits(fit_ids, root/'fit/frames', shared, True, shared_pose):
            if not np.array_equal(arrays['betas'], shared):
                raise ValueError('Official fit changed the shared frozen shape.')
            winners[frame], descriptors[frame] = arrays, desc
        fitted_count = len(winners)
        if scope is not None:
            reports = {frame: rt.read(desc['path']) for frame, desc in descriptors.items()}
            hashes = {report.get('shared_body_target_sha256') for report in reports.values()}
            if len(hashes) != 1 or None in hashes:
                raise ValueError('Final fits did not use one shared body target.')
            selection = stable_selection.select_frames(inputs, winners, scope=scope, reports=reports,
                                                        shared_betas=shared, allow_empty=True)
            if not selection['full_bin_coverage']:
                selection, extra['pose_recovery'] = _recover_missing_bins(
                    root, inputs, scope, selection, winners, descriptors, reports,
                    shared, shared_pose, shape_anchor_seeds, pool)
            selected = selection['selected_frames']
            if not selected:
                raise ValueError('No technically valid non-holdout poses for automatic training.')
            reference = reference_frame or selection['reference_frame']
            if reference not in selected:
                raise ValueError('Requested reference did not pass the automatic training selection.')
            selection['reference_frame'] = reference
            ordered = [reference] + [frame for frame in selected if frame != reference]
            if len(ordered) != len(set(ordered)) or set(ordered) & set(scope['retained_holdouts']):
                raise ValueError('Automatic training selection changed the protected roster.')
            rt.save(root/'fit/selection.json', selection)
            inputs = _selection_inputs(inputs, ordered, reference)
            winners = {frame: winners[frame] for frame in ordered}
            extra.update(selection=selection, fitted_candidate_frames=fitted_count)
        receipt = write_results(root, inputs, winners, shared, basis,
            [descriptors[frame] for frame in inputs['ids']], execution_kind='fresh_official_sequence',
            extra=dict(extra, elapsed_seconds=time.monotonic()-started))
        rt.save(root/'fit/progress.json', dict(state='complete', frames_completed=fitted_count))
        return receipt
    except BaseException as error:
        rt.save(root/'fit/progress.json', dict(state='failed', frame_id=active, error=str(error)))
        raise
    finally:
        stack.close()


def assemble_saved_winners(*, inputs, root, winners, shared_betas, basis):
    """Read-only historical verification: never initializes a model or fits a frame."""
    root = Path(root).resolve()
    (root/'fit').mkdir(parents=True, exist_ok=False)
    if set(winners) != set(inputs['ids']):
        raise ValueError('Saved winners must match the full named roster.')
    arrays, descriptors = {}, []
    for frame in inputs['ids']:
        item = winners[frame]
        for key in ('npz', 'result'):
            if rt.sha(item[key]) != item[key+'_sha256']:
                raise ValueError('Saved winner changed: ' + frame)
            inputs['bound'][str(Path(item[key]).resolve())] = item[key+'_sha256']
        with np.load(item['npz'], allow_pickle=False) as f:
            arrays[frame] = {key: f[key].copy() for key in f.files}
        descriptors.append(dict(frame_id=frame, path=str(Path(item['result']).resolve()), sha256=item['result_sha256']))
    return write_results(root, inputs, arrays, shared_betas, basis, descriptors,
                         execution_kind='historical_readonly_assembly')
