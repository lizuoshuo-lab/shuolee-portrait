"""Named-frame I/O around the official single-frame solver; no temporal optimizer."""
from pathlib import Path
import io
import json
import re
import time

import numpy as np
from PIL import Image

from . import official_smplifyx as official, runtime as rt
from .camera_world_units import normalize_camera_w2c


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


def load_inputs(frames_path, observations_path, *, camera_length_contract, reference_frame):
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
    targets, confidence = official.adapt_rtmw_observations(
        *(obs[key][order] for key in ('keypoints', 'scores', 'boxes', 'flip_disagreement')))
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
    check_bound(bound)
    return dict(rows=rows, ids=ids, targets=targets, K=K, units=units,
                reference_frame=reference_frame, bound=bound, confidence=confidence,
                frames_path=frames_path, source_manifest=manifest)


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
    if (report.get('state') != 'complete' or report.get('frame_id') != row['frame'] or
            report.get('source_frame_index') != row['source_frame_index'] or
            report.get('source_image_sha256') != row['source_sha256'] or
            report.get('official_commit') != official.COMMIT or report.get('gvhmr_used') is not False or
            any(report.get(key) is not True for key in ('five_stages', 'upstream_loss_monitor_optimizer_unchanged',
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


def fit_named_sequence(*, frames_path, observations_path, root, official_root, assets,
                       camera_length_contract, reference_frame, shape_policy):
    """One reusable model pair, independent official fits, one explicitly chosen shape."""
    root = Path(root).resolve()
    if (root/'fit').exists():
        raise FileExistsError('Use a fresh run for a new shared-shape pass.')
    inputs = load_inputs(frames_path, observations_path, camera_length_contract=camera_length_contract,
                         reference_frame=reference_frame)
    shared, bound = read_shape_policy(shape_policy); inputs['bound'].update(bound)
    official.adapted_entry_source(official_root)
    (root/'fit/frames').mkdir(parents=True)
    started = time.monotonic()
    active = None
    try:
        body, vposer, meta = official.create_models(official_root=official_root, assets=assets)
        basis = pelvis_basis(body)
        def fit(row, destination, beta, freeze):
            nonlocal active
            active = row['frame']; i = inputs['ids'].index(active)
            check_bound(inputs['bound']); destination.mkdir()
            rt.save(root/'fit/progress.json', dict(state='fitting', frame_id=active, freeze_betas=freeze))
            arrays, report = official.fit_frame(official_root=official_root, image=checked_image(row),
                keypoints=inputs['targets'][i:i+1], K_integer_centers=inputs['K'][i], body_model=body,
                vposer2=vposer, result_path=destination/'official-result.pkl', shared_betas=beta,
                freeze_betas=freeze, device='cuda', diff_path=destination/'input-output-adapter.diff')
            arrays['frame_id'] = np.asarray(active)
            np.savez_compressed(destination/'winner-camera-space.npz', **arrays)
            report.update(state='complete', frame_id=active, source_frame_index=row['source_frame_index'],
                          source_image_sha256=row['source_sha256'], quality_accepted=False)
            rt.save(destination/'result.json', report)
            return arrays, dict(frame_id=active, path=str(destination/'result.json'), sha256=rt.sha(destination/'result.json'))
        if shared is None:
            ref = inputs['rows'][inputs['ids'].index(reference_frame)]
            initial, _ = fit(ref, root/'fit/reference-initialization', None, False)
            shared = np.asarray(initial['betas'], dtype=np.float32).reshape(1, 10).copy()
        winners, descriptors = {}, []
        for row in inputs['rows']:
            arrays, descriptor = fit(row, root/'fit/frames'/row['frame'], shared, True)
            if not np.array_equal(arrays['betas'], shared):
                raise ValueError('Official fit changed the shared frozen shape.')
            winners[row['frame']] = arrays; descriptors.append(descriptor)
        receipt = write_results(root, inputs, winners, shared, basis, descriptors,
            execution_kind='fresh_official_sequence', extra=dict(shape_policy=shape_policy,
                vposer=meta['vposer'], elapsed_seconds=time.monotonic()-started))
        rt.save(root/'fit/progress.json', dict(state='complete', frames_completed=len(winners)))
        return receipt
    except BaseException as error:
        rt.save(root/'fit/progress.json', dict(state='failed', frame_id=active, error=str(error)))
        raise


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
