"""Constrain native head and ear alignment to the frozen SMPL-X identity and camera."""

import hashlib
import io
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from . import official_smplifyx as official
from . import runtime as rt
from .face_supplement import REGIONS, _front_depth, _load_inputs, _pixel_support
from .stable_selection import BODY, FEET


HEAD_COLUMNS = np.r_[33:36, 42:45]
FACE_POINTS = np.unique(np.concatenate(list(REGIONS.values())))
NON_HEAD = np.r_[1:15, 19:67]
REPLAY_FIXED = ('left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression')
CONFIG = dict(max_joint_delta_rad=.12, finite_difference_rad=.001, linearization_passes=2,
              backtracks=(1., .5, .25, .125), minimum_region_gain_px=.05,
              face_region_rms_growth_px=.25, face_region_peak_growth_px=.25,
              body_landmark_max_delta_px=.08, body_surface_max_delta_m=.001,
              reliable_confidence=.50, min_raw_score=.35, max_flip_px=6.,
              max_flip_over_span=.08, max_occlusion_m=.012, min_face_span_px=32.,
              min_eye_points=3, min_nose_points=4, min_mouth_points=6, min_ear_points=1,
              near_ear_min_depth_difference_m=.005, silhouette_mask_alpha_max=.05,
              silhouette_margin_px=2., max_new_head_exterior_samples=0,
              max_head_exterior_distance_growth_px=1., pose_prior_weight=2.)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def load_head_model(assets, official_root):
    """Construct the exact official native observation mapper without loading VPoser."""
    import smplx
    from smplx.vertex_ids import vertex_ids
    import torch

    source = Path(official_root)/'smplifyx/utils.py'
    _require(rt.sha(source) == official.PINNED['smplifyx/utils.py'], 'Official SMPL-X mapper changed.')
    utils = official.module(source, '_head_alignment_smplifyx_utils')
    mapping = np.asarray(utils.smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True,
                                               use_face_contour=False, openpose_format='coco25'))
    path = Path(assets)/'smplx/SMPLX_NEUTRAL.npz'
    model = smplx.SMPLX(str(path), batch_size=1, num_betas=10, num_expression_coeffs=10,
                      use_pca=True, num_pca_comps=12, flat_hand_mean=False, use_face_contour=False,
                      create_body_pose=False, create_transl=False,
                      joint_mapper=utils.JointMapper(mapping), dtype=torch.float32).eval().requires_grad_(False)
    right, left = [vertex_ids['smplx'][name] for name in ('rear', 'lear')]
    _require(mapping.shape == (118,), 'SMPL-X needs the official 118-slot native landmark mapping.')
    landmarks = dict(right_ear=dict(observation_slot=17, rtmw_slot=int(np.where(official.COCO == 17)[0][0]),
                                    smplx_joint=int(mapping[17]), vertex=int(right)),
                     left_ear=dict(observation_slot=18, rtmw_slot=int(np.where(official.COCO == 18)[0][0]),
                                   smplx_joint=int(mapping[18]), vertex=int(left)))
    proof = dict(asset_sha256=rt.sha(path), mapper_sha256=rt.sha(source), joint_mapping=mapping.tolist(),
                 ears=landmarks, verified_exact_vertex_identity=True, independent_physical_ear_identity=False)
    return model, proof


def _project(points, K):
    transformed = points.astype(np.float64) @ np.swapaxes(K, -1, -2)
    if np.any(transformed[..., 2] <= .05):
        raise ValueError('Head-alignment projection crosses the native camera.')
    return transformed[..., :2]/transformed[..., 2:]


def _render(model, fit, slots, pose):
    import torch

    values = {name: torch.from_numpy(np.ascontiguousarray(fit[name][slots], dtype=np.float32))
              for name in (*REPLAY_FIXED, 'global_orient')}
    betas = torch.from_numpy(np.ascontiguousarray(fit['betas'], dtype=np.float32)).expand(len(slots), -1)
    delta = torch.from_numpy(np.ascontiguousarray(fit['camera_translation'][slots], dtype=np.float32))[:, None]
    with torch.no_grad():
        result = model(**values, betas=betas, body_pose=torch.from_numpy(np.ascontiguousarray(pose, dtype=np.float32)),
                       return_verts=True)
        vertices, joints = (result.vertices+delta).numpy(), (result.joints+delta).numpy()
    return vertices, joints, _project(joints, fit['K_integer_centers'][slots])


def _batch_render(model, fit, slots, pose):
    batches = [_render(model, fit, slots[start:start+8], pose[start:start+8]) for start in range(0, len(slots), 8)]
    return tuple(np.concatenate(values) for values in zip(*batches, strict=True))


def _read_mask(row):
    raw = Path(row['mask']).read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == row['mask_sha256'], 'Source alpha changed: '+row['frame'])
    with Image.open(io.BytesIO(raw)) as image:
        _require(list(image.size) == row['wh'], 'Source alpha dimensions changed: '+row['frame'])
        return np.asarray(image.convert('L'))


def _occlusion(points, joints, vertices, faces, K):
    front = _front_depth(points[FACE_POINTS], vertices, faces, K)
    delta = np.full(len(points), np.inf)
    delta[FACE_POINTS] = np.where(np.isfinite(front), joints[FACE_POINTS, 2]-front, np.inf)
    return delta


def _screen_observations(row, observed, raw, flip, vertices, joints, projected, faces, K):
    mask = _read_mask(row)
    confidence = (observed[:, 2] >= CONFIG['reliable_confidence']) & (raw >= CONFIG['min_raw_score'])
    inner = observed[67:, :2][confidence[67:]]
    span = float(np.ptp(inner, axis=0).max()) if len(inner) else 0.
    supported = confidence & (flip <= min(CONFIG['max_flip_px'], CONFIG['max_flip_over_span']*span))
    supported &= _pixel_support(observed[:, :2], mask)
    for left, top, right, bottom in row.get('other_person_boxes_xyxy', ()):
        xy = observed[:, :2]
        supported &= ~((xy >= [left, top]) & (xy < [right, bottom])).all(1)
    depth = _occlusion(projected, joints, vertices, faces, K)
    supported &= depth <= CONFIG['max_occlusion_m']
    gap = abs(float(joints[17, 2]-joints[18, 2]))
    if gap >= CONFIG['near_ear_min_depth_difference_m']:
        far = 17 if joints[17, 2] > joints[18, 2] else 18
        supported[far] = False
    regions, evidence = {}, {}
    for name, indices in REGIONS.items():
        slots = indices[supported[indices]]
        minimum = CONFIG['min_'+name.split('_')[-1]+'_points']
        if len(slots) >= minimum and span >= CONFIG['min_face_span_px']:
            regions[name] = slots
        evidence[name] = dict(available=name in regions, supported_points=slots.tolist(),
                              occlusion_mm=[float(depth[i]*1000) if np.isfinite(depth[i]) else None for i in indices],
                              rejected_or_unobserved_points=indices[~supported[indices]].tolist())
    return mask, regions, evidence, span


def _region_errors(projected, observed, regions):
    result = {}
    for name, indices in regions.items():
        residual = np.linalg.norm(projected[indices]-observed[indices, :2], axis=1)
        weights = observed[indices, 2].astype(np.float64)**2
        result[name] = dict(rms_px=float(np.sqrt(np.average(residual**2, weights=weights))),
                            peak_px=float(residual.max()))
    return result


def _metrics_gain(before, after, ears):
    keys = [name for name in before if name.endswith('_ear') == ears]
    return float(np.mean([before[name]['rms_px']-after[name]['rms_px'] for name in keys])) if keys else None


def _linear_step(start, center, projected, jacobian, observed, regions, before):
    residual = projected-observed[:, :2]
    rms_tol, peak_tol = CONFIG['face_region_rms_growth_px'], CONFIG['face_region_peak_growth_px']
    weights = {}
    for key, ids in regions.items():
        confidence = observed[ids, 2].astype(np.float64)**2
        strength = 1. if key.endswith('_ear') else .4
        weights[key] = strength*confidence/confidence.sum()

    def candidate(delta):
        return residual+np.einsum('ijk,k->ij', jacobian, delta-center)

    def objective(delta):
        error = candidate(delta)
        return (sum(np.sum(weights[key]*np.square(error[ids]).sum(1)) for key, ids in regions.items())
                +CONFIG['pose_prior_weight']*np.square(delta).sum())

    def objective_jac(delta):
        error = candidate(delta)
        return (sum(2*np.einsum('n,ni,nik->k', weights[key], error[ids], jacobian[ids])
                    for key, ids in regions.items())+2*CONFIG['pose_prior_weight']*delta)

    def margins(delta):
        errors = np.square(candidate(delta)).sum(1)
        values = []
        for key, ids in regions.items():
            if key.endswith('_ear'):
                continue
            values.append((before[key]['rms_px']+rms_tol)**2-np.average(errors[ids], weights=weights[key]))
            values.extend((before[key]['peak_px']+peak_tol)**2-errors[ids])
        values.extend(CONFIG['max_joint_delta_rad']**2-np.square(delta.reshape(2, 3)).sum(1))
        return np.asarray(values)

    def margin_jac(delta):
        gradient = 2*np.einsum('ni,nik->nk', candidate(delta), jacobian)
        rows = []
        for key, ids in regions.items():
            if not key.endswith('_ear'):
                rows.append(-np.average(gradient[ids], axis=0, weights=weights[key]))
                rows.extend(-gradient[ids])
        for block in (slice(0, 3), slice(3, 6)):
            row = np.zeros(6)
            row[block] = -2*delta[block]
            rows.append(row)
        return np.asarray(rows)

    solution = minimize(objective, start, jac=objective_jac,
                        constraints=[dict(type='ineq', fun=margins, jac=margin_jac)],
                        method='SLSQP', bounds=[(-CONFIG['max_joint_delta_rad'], CONFIG['max_joint_delta_rad'])]*6,
                        options=dict(maxiter=32, ftol=1e-7))
    if np.isfinite(solution.x).all():
        # At an active inequality SLSQP can stop a few ulps outside.
        # Retract toward the same endpoint under the registered limits.
        for fraction in (1., .95, .8, .5, .25):
            step = start+fraction*(solution.x-start)
            if margins(step).min() >= -1e-7 and objective(step) < objective(start):
                return step
    return start


def _exterior_evidence(vertices, K, mask, head, distance=None):
    from scipy.ndimage import distance_transform_edt

    height, width = mask.shape
    image = _project(vertices[head], K)
    if distance is None:
        distance = distance_transform_edt(mask <= 255*CONFIG['silhouette_mask_alpha_max']).astype(np.float32)
    x, y = np.rint(image).astype(int).T
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    outside = np.full(len(x), 1e3, np.float32)
    outside[valid] = distance[y[valid], x[valid]]
    return outside, distance


def _rotation_deltas(before, after):
    start = Rotation.from_rotvec(before[HEAD_COLUMNS].reshape(2, 3))
    stop = Rotation.from_rotvec(after[HEAD_COLUMNS].reshape(2, 3))
    return (start.inv()*stop).magnitude()


def _native_gate(before, after, observed, regions, old_vertices, new_vertices, old_pixels, new_pixels,
                 new_joints, pose0, pose1, K, faces, mask, head, body, exterior0, distance,
                 body_screen_height, is_original):
    failures = []
    for name in before:
        if any(after[name][metric] > before[name][metric]+CONFIG['face_region_'+metric.split('_')[0]+'_growth_px']
               for metric in ('rms_px', 'peak_px')):
            failures.append('regressed_native_'+name)
    depth = _occlusion(new_pixels, new_joints, new_vertices, faces, K)
    if any(np.any(depth[indices] > CONFIG['max_occlusion_m']) for indices in regions.values()):
        failures.append('lost_frozen_mesh_visibility')
    body_pixels = float(np.linalg.norm(new_pixels[NON_HEAD]-old_pixels[NON_HEAD], axis=1).max())
    body_surface = float(np.linalg.norm(new_vertices[body]-old_vertices[body], axis=1).max())
    if body_pixels > CONFIG['body_landmark_max_delta_px'] or body_surface > CONFIG['body_surface_max_delta_m']:
        failures.append('body_geometry_moved')
    rotation = _rotation_deltas(pose0, pose1)
    if rotation.max() > CONFIG['max_joint_delta_rad']+1e-5:
        failures.append('rotation_trust_region')
    exterior, _ = _exterior_evidence(new_vertices, K, mask, head, distance)
    newly_outside = (exterior0 <= CONFIG['silhouette_margin_px']) & (exterior > CONFIG['silhouette_margin_px'])
    if (int(newly_outside.sum()) > CONFIG['max_new_head_exterior_samples'] or
            float(exterior.max()-exterior0.max()) > CONFIG['max_head_exterior_distance_growth_px']):
        failures.append('regressed_independent_source_silhouette')
    screen = np.r_[BODY, FEET]
    valid = observed[screen, 2] > 0
    screen_error = float(np.linalg.norm(new_pixels[screen][valid]-observed[screen][valid, :2], axis=1).max())
    if is_original and screen_error > .03*body_screen_height:
        failures.append('would_fail_existing_full_body_screen')
    return failures, dict(body_max_landmark_delta_px=body_pixels, body_max_surface_delta_m=body_surface,
                          neck_head_relative_rotation_rad=rotation.tolist(), new_exterior_head_samples=int(newly_outside.sum()),
                          head_exterior_max_before_px=float(exterior0.max()), head_exterior_max_after_px=float(exterior.max()),
                          existing_body_feet_peak_px=screen_error)


def refine_local_head(model, fit, rows, slots, native_raw, native_flip, *, original_slots=()):
    """Recompute admissible local pose updates for the provided, independently fitted rows."""
    slots = np.asarray(slots, dtype=int)
    original_slots = set(original_slots)
    poses = fit['body_pose_axis_angle'][slots].copy()
    masks, regions, evidence, spans = zip(*[
        _screen_observations(rows[slot], fit['observed_native_keypoints'][slot], native_raw[slot], native_flip[slot],
                             fit['vertices_camera'][slot], fit['joints_camera'][slot],
                             fit['joints_projected_native'][slot], fit['faces'], fit['K_integer_centers'][slot])
        for slot in slots], strict=True)
    first = [_region_errors(fit['joints_projected_native'][slot], fit['observed_native_keypoints'][slot], visible)
             for slot, visible in zip(slots, regions, strict=True)]
    eligible = np.asarray([('nose' in region and 'mouth' in region and
                           ('right_eye' in region or 'left_eye' in region)) for region in regions])
    baseline_v, baseline_j, baseline_p = _batch_render(model, fit, slots, poses)
    _require(np.max(np.abs(baseline_v-fit['vertices_camera'][slots])) < 2e-6
             and np.max(np.abs(baseline_j-fit['joints_camera'][slots])) < 2e-6
             and np.max(np.abs(baseline_p-fit['joints_projected_native'][slots])) < .002,
             'Archived native fit does not reproduce with the pinned official SMPL-X mapper.')
    delta = np.zeros((len(slots), 6))
    for _ in range(CONFIG['linearization_passes']):
        center = poses.copy()
        center[:, HEAD_COLUMNS] += delta.astype(np.float32)
        _, _, projected = _batch_render(model, fit, slots, center)
        jacobian = []
        for column in HEAD_COLUMNS:
            perturbed = center.copy()
            perturbed[:, column] += CONFIG['finite_difference_rad']
            _, _, prediction = _batch_render(model, fit, slots, perturbed)
            jacobian.append((prediction-projected)/CONFIG['finite_difference_rad'])
        jacobian = np.stack(jacobian, axis=-1)
        for i in np.flatnonzero(eligible):
            old = delta[i].copy()
            delta[i] = _linear_step(old, old, projected[i], jacobian[i], fit['observed_native_keypoints'][slots[i]],
                                    regions[i], first[i])

    # Skinning support determines which neck/head vertices can move and
    # which body surface samples must stay within the original metric pose.
    skin = model.lbs_weights.detach().cpu().numpy()
    influence = skin[:, [12, 15, 22, 23, 24]].sum(1)
    head, body = influence >= .5, influence < .01
    exterior = [_exterior_evidence(fit['vertices_camera'][slot], fit['K_integer_centers'][slot], mask, head)
                for slot, mask in zip(slots, masks, strict=True)]
    evaluated = []
    changed, trials = {}, [[] for _ in slots]
    for fraction in CONFIG['backtracks']:
        pending = np.asarray([i for i in np.flatnonzero(eligible) if i not in changed and np.linalg.norm(delta[i]) > 1e-5],
                             dtype=int)
        if not len(pending):
            break
        trial_pose = poses[pending].copy()
        trial_pose[:, HEAD_COLUMNS] += (fraction*delta[pending]).astype(np.float32)
        vertices, joints, pixels = _batch_render(model, fit, slots[pending], trial_pose)
        for out, i in enumerate(pending):
            slot = slots[i]
            observed = fit['observed_native_keypoints'][slot]
            after = _region_errors(pixels[out], observed, regions[i])
            gain, ear_gain = (_metrics_gain(first[i], after, kind) for kind in (False, True))
            before_p = fit['joints_projected_native'][slot]
            box = rows[slot]['initial_box_xyxy']
            failures, geometry = _native_gate(first[i], after, observed, regions[i], fit['vertices_camera'][slot],
                vertices[out], before_p, pixels[out], joints[out], poses[i], trial_pose[out], fit['K_integer_centers'][slot],
                fit['faces'], masks[i], head, body, *exterior[i], box[3]-box[1], slot in original_slots)
            if ear_gain is not None and ear_gain < CONFIG['minimum_region_gain_px']:
                failures.append('no_measurable_ear_gain')
            if ear_gain is None and gain < CONFIG['minimum_region_gain_px']:
                failures.append('no_measurable_face_gain')
            trials[i].append(dict(fraction=fraction, failures=failures, after=after, inner_face_gain_px=gain,
                                  ear_gain_px=ear_gain, geometry=geometry))
            if not failures:
                changed[i] = dict(body_pose_axis_angle=trial_pose[out], vertices_camera=vertices[out],
                                  joints_camera=joints[out], joints_projected_native=pixels[out].astype(np.float32))

    for i, slot in enumerate(slots):
        observed = fit['observed_native_keypoints'][slot]
        final = changed.get(i)
        before_p = fit['joints_projected_native'][slot]
        after_p = final['joints_projected_native'] if final is not None else before_p
        after = _region_errors(after_p, observed, regions[i])
        evaluated.append(dict(frame=rows[slot]['frame'], source_index=rows[slot]['source_frame_index'],
                              source_image_sha256=rows[slot]['source_sha256'], independent_mask_sha256=rows[slot]['mask_sha256'],
                              state='committed' if final is not None else ('kept_baseline' if eligible[i] else 'unsupported'),
                              native_face_span_px=spans[i], visibility=evidence[i], before=first[i], after=after,
                              before_projected_native=before_p[FACE_POINTS].tolist(),
                              after_projected_native=after_p[FACE_POINTS].tolist(),
                              native_observed=observed[FACE_POINTS].tolist(), trials=trials[i]))
    return {int(slots[i]): value for i, value in changed.items()}, evaluated


def fit_head_alignment(root, *, assets, official_root, auxiliary_frames):
    """Return fit archives with named, protected local head changes and a native evidence ledger."""
    import torch

    started = perf_counter()
    root = Path(root)
    base_rows = rt.read(root/'data/training-inputs.json')['records']
    _, rows, selection, fit, units, raw, flip, source_hashes = _load_inputs(root, base_rows)
    names = fit['frame_ids'].astype(str).tolist()
    original = [row['frame'] for row in base_rows]
    auxiliary = list(auxiliary_frames)
    roster = original+auxiliary
    _require(len(set(roster)) == len(roster) and set(roster) <= set(names)
             and not set(roster) & set(selection['retained_holdouts']), 'Invalid frozen head fitting roster.')
    slots = [names.index(name) for name in roster]
    model, semantics = load_head_model(assets, official_root)
    _require(np.array_equal(model.faces, fit['faces']), 'Official frozen head topology differs.')
    ear_vertices = [value['vertex'] for value in semantics['ears'].values()]
    _require(np.array_equal(fit['joints_camera'][:, 17:19], fit['vertices_camera'][:, ear_vertices]),
             'Archived left/right ear landmark semantics differ from the official vertices.')
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(min(previous_threads, 4))
        updates, frames = refine_local_head(model, fit, rows, slots, raw, flip, original_slots=slots[:len(original)])
    finally:
        torch.set_num_threads(previous_threads)
    with np.load(root/'fit/fitted.npz', allow_pickle=False) as archive:
        base = dict(archive)
    base_index = {str(name): i for i, name in enumerate(base['frame_ids'])}
    fixed = dict(shared_betas=_digest(base['shared_betas']), original_targets=_digest(base['observed_native_keypoints']),
                 all_candidate_targets=_digest(fit['observed_native_keypoints']), full_K=_digest(base['K_integer_centers']),
                 all_candidate_K=_digest(fit['K_integer_centers']), original_w2c=_digest(base['w2c']),
                 topology=_digest(base['faces']), original_pose_embedding=_digest(base['pose_embedding']))
    for slot, delta in updates.items():
        for field, value in delta.items():
            fit[field][slot] = value
        if names[slot] in base_index:
            i = base_index[names[slot]]
            for field, value in delta.items():
                base[field][i] = value
            E = units.w2c_meters[slot]
            base['vertices_world'][i] = (delta['vertices_camera'].astype(np.float64)-E[:3, 3]) @ E[:3, :3]
    audit = dict(schema='shuolee-native-head-alignment/1', state='completed', config=CONFIG.copy(),
                 input_sha256=source_hashes, fixed_arrays_sha256=fixed, mapping=semantics, head_columns=HEAD_COLUMNS.tolist(),
                 original_frames=original, auxiliary_frames=auxiliary, retained_holdouts=selection['retained_holdouts'],
                 fitted_frames=roster, changed_frames=[names[slot] for slot in updates], frames=frames,
                 evaluated_landmark_slots=FACE_POINTS.tolist(), shared_shape_updates=0, per_frame_scale_parameters=0,
                 pose_embedding_semantics='Archived VPoser provenance; replay from explicit body_pose_axis_angle.',
                 mask_semantics='Independent source alpha for native target support and mesh exterior checks.',
                 visibility_semantics='Frozen mesh perspective-correct front depth plus native detector and source alpha.',
                 elapsed_seconds=perf_counter()-started, quality_accepted=False)
    return base, fit, audit
