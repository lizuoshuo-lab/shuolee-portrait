"""Jointly refine named training candidates in the fixed camera-relative model."""

import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .coverage_refinement import _clip
from .joint_shared_step import project_halfspaces, row_constraints
from .stable_selection import BODY, FACE, FEET


CONFIG = dict(max_sweeps=144, batch_size=8, raster_long_edge=448, reliable_confidence=.15,
              outside_margin_px=1, outside_distance_cap_px=12, reprojection_unit_over_height=.03,
              latent_residual_l2_limit=8., root_update_max_rad=.55, translation_update_max_m=.12,
              shared_latent_lr=.025, residual_latent_lr=.055, root_lr=.013, translation_lr=.0032,
              body_weight=1., feet_weight=.75, face_weight=.55, hand_weight=.12, screen_tail_weight=.45,
              silhouette_weight=.55, local_rotation_weight=.45, latent_residual_weight=.012,
              anchor_weight=.015, root_trust_weight=.012, translation_trust_weight=.018,
              face_unit_px=2., face_guard_confidence=.5, face_guard_tolerance_px=.25,
              face_guard_weight=1., neck_head_update_max_rad=.25, neck_head_lr=.0075,
              neck_head_trust_weight=.012, shared_every=4, stable_sweeps=12,
              objective_relative_stop=1e-5, screen_stop=1e-4,
              trust_min=.0625, replay_margin_px=.02,
              trial_fractions=(1., .5, .25, .125, .0625), projection_sweeps=4,
              shared_trust_initial=.20, shared_trust_min=.00390625, shared_trust_max=.50,
              shared_fd_steps=(.02, .005), shared_projected_stop=1e-5, shared_min_gain_ratio=.05,
              shared_stationary_checks=3)
FROZEN_FIELDS = ('left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression')
GROUPS = {'body': BODY, 'feet': FEET, 'face': FACE, 'hands': np.arange(25, 67)}
# The official adapter maps inner 68-point facial landmarks #17–67 to slots 67–117.
FACE_REGIONS = dict(face=FACE, right_eye=np.arange(86, 92), left_eye=np.arange(92, 98),
                    nose=np.arange(77, 86), mouth=np.arange(98, 118),
                    right_ear=np.array([17]), left_ear=np.array([18]))
# SMPL-X local body-pose indices, excluding the global pelvis rotation.
NECK_HEAD = (11, 14)


def _face_regions(observations, wh):
    membership = torch.zeros((len(FACE_REGIONS), observations.shape[1]), dtype=torch.bool, device=observations.device)
    for i, ids in enumerate(FACE_REGIONS.values()):
        membership[i, ids.tolist()] = True
    coords = observations[..., :2]
    reliable = ((observations[..., 2] >= CONFIG['face_guard_confidence']) & torch.isfinite(observations).all(-1)
                & ((coords >= 0) & (coords < coords.new_tensor(wh))).all(-1))
    support = reliable[:, None, :] & membership[None, :, :]
    weights = torch.where(support, observations[..., 2][:, None, :].square(), 0.)
    return weights, support


def _face_stats(error, weights, support):
    expanded = error[:, None, :].masked_fill(~support, 0.)
    rms = ((expanded.square()*weights).sum(-1)/weights.sum(-1).clamp_min(1e-12)).clamp_min(1e-12).sqrt()
    peak = expanded.amax(-1)
    return torch.stack((rms, peak), dim=-1)


def _guard_details(names, before, after, points):
    if not torch.isfinite(before).all() or not torch.isfinite(after).all():
        raise ValueError('Protected face metrics must be finite.')
    tolerance = CONFIG['face_guard_tolerance_px']
    records, failures = {}, []
    for name, base, candidate, counts in zip(names, before.tolist(), after.tolist(), points.tolist()):
        frame = records[name] = {}
        for region, old, new, count in zip(FACE_REGIONS, base, candidate, counts):
            if not count:
                frame[region] = dict(supported_points=0, status='unobserved')
                continue
            checks = ('rms_px', 'peak_px')
            reasons = [metric for metric, a, b in zip(checks, old, new) if b > a+tolerance]
            frame[region] = dict(supported_points=count, status='fail' if reasons else 'pass',
                                 before=dict(zip(checks, old)), after=dict(zip(checks, new)),
                                 delta=dict(zip(checks, (b-a for a,b in zip(old,new)))), failures=reasons)
            failures.extend(dict(frame=name, region=region, metric=metric,
                                 before_px=frame[region]['before'][metric],
                                 after_px=frame[region]['after'][metric]) for metric in reasons)
    return records, failures


def _bounded_rows(values, limit):
    return values * (limit / values.norm(dim=-1, keepdim=True).clamp_min(limit))


def _mean_joint_angle_degrees(first, second):
    delta = (first*second).sum((-1, -2))
    return torch.rad2deg(torch.acos(((delta-1.)*.5).clamp(-1., 1.))).mean(1)


def _project_feasible_step(direction, normals, slack, sweeps):
    """Project scaled row updates against the linearized supported inequalities."""
    for _ in range(sweeps):
        for normal, limit in zip(normals.unbind(1), slack.unbind(1)):
            excess = ((direction*normal).sum(-1) - limit).clamp_min(0.)
            direction = direction - normal*(excess/normal.square().sum(-1).clamp_min(1e-12))[:, None]
    return direction


def _project(points, K):
    projective = points @ K.transpose(-1, -2)
    return projective[..., :2] / projective[..., 2:].clamp_min(.05)


def _bounds_and_targets(rows, device):
    from scipy.ndimage import distance_transform_edt

    native = {tuple(row['wh']) for row in rows}
    if len(native) != 1:
        raise ValueError('Joint fitting requires one native raster size within the source clip.')
    width, height = native.pop()
    scale = CONFIG['raster_long_edge'] / max(width, height)
    # Raster dimensions are multiples of eight. Each axis uses its exact
    # native-to-raster scale in corner coordinates.
    w, h = (max(8, 8*round(d*scale/8)) for d in (width, height))
    masks, exterior = [], []
    for row in rows:
        data = Path(row['mask']).read_bytes()
        if hashlib.sha256(data).hexdigest() != row['mask_sha256']:
            raise ValueError('Joint silhouette mask hash changed: ' + row['frame'])
        with Image.open(io.BytesIO(data)) as image:
            if image.size != (width, height):
                raise ValueError('Joint silhouette mask differs from native source: ' + row['frame'])
            coverage = np.asarray(image.convert('L').resize((w, h), Image.Resampling.BOX), dtype=np.float32)/255.
        foreground = coverage >= .5
        if not foreground.any() or foreground.all():
            raise ValueError('Joint silhouette mask has no reliable person boundary: ' + row['frame'])
        distance = distance_transform_edt(coverage <= .05) - CONFIG['outside_margin_px']
        distance = np.clip(distance, 0., CONFIG['outside_distance_cap_px'])
        masks.append(foreground)
        exterior.append((distance / CONFIG['outside_distance_cap_px']).astype(np.float32))
    return (w, h), (w/width, h/height), torch.as_tensor(np.stack(masks), device=device), \
        torch.as_tensor(np.stack(exterior), device=device)


def _raster_silhouette(dr, context, vertices, K, faces, dimensions, scale, *, antialias):
    w, h = dimensions
    projected = vertices @ K.transpose(-1, -2)
    x, y, z = projected.unbind(-1)
    corner = torch.stack(((x+.5*z)*scale[0], (y+.5*z)*scale[1], z), dim=-1)
    clip = _clip(corner, w, h).contiguous()
    raster, _ = dr.rasterize(context, clip, faces, (h, w), grad_db=False)
    alpha = (raster[..., 3:] > 0).to(clip.dtype)
    if antialias:
        alpha = dr.antialias(alpha, raster, clip, faces)
    return alpha[..., 0].flip(1)


def _projection_metrics(observed, projected, box_height):
    error = np.linalg.norm(projected-observed[:, :2], axis=1)
    metrics = {}
    for group, indices in GROUPS.items():
        confidence = observed[indices, 2]
        supported = confidence > 0
        reliable = confidence >= CONFIG['reliable_confidence']
        values = error[indices]
        rms = np.sqrt(np.average(values[supported]**2, weights=confidence[supported]**2)) if supported.any() else None
        maximum = float(values[supported].max()) if supported.any() else None
        metrics[group] = dict(visible_points=int(supported.sum()), reliable_points=int(reliable.sum()),
                              rms_px=float(rms) if rms is not None else None, max_px=maximum,
                              max_over_height=maximum/box_height if maximum is not None else None)
    return metrics


def _shape_output(body, beta, fixed, root, translation, pose):
    result = body(**fixed, betas=beta, global_orient=root, body_pose=pose, return_verts=True)
    return result.vertices + translation[:, None], result.joints + translation[:, None]


def _packed_parameters(winners, names, field, device):
    return torch.as_tensor(np.concatenate([winners[name][field] for name in names], axis=0),
                           dtype=torch.float32, device=device)


def _checkpoint_coverage(names, eligible, scope):
    eligible = {name for name, passed in zip(names, eligible) if passed}
    bins = scope['source_view_bins']
    targets = scope['view_targets']
    front = bins[:targets['front']]
    rear = bins[targets['front']:]
    by_side = [sum(name in eligible for group in side for name in group) for side in (front, rear)]
    roster = min(by_side[0], targets['front']) + min(by_side[1], targets['rear'])
    covered = sum(bool(eligible.intersection(group)) for group in bins)
    return dict(pose_gate_eligible=len(eligible), pose_gate_eligible_front=by_side[0],
                pose_gate_eligible_rear=by_side[1], pose_gate_roster_capacity=roster,
                source_bins_with_pose_gate_eligible=covered)


def _evaluate_joint(*, body, vposer, winners, rows, targets, intrinsics, beta, anchor, scope, progress_path):
    import nvdiffrast.torch as dr
    from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
    from . import runtime as rt

    device = next(body.parameters()).device
    config = CONFIG
    names = [row['frame'] for row in rows]
    count = len(rows)
    height = np.asarray([row['initial_box_xyxy'][3]-row['initial_box_xyxy'][1] for row in rows])
    if not np.isfinite(height).all() or (height <= 0).any():
        raise ValueError('Joint fitting requires positive native body-box heights.')
    bbox_scale = torch.as_tensor(height*config['reprojection_unit_over_height'], dtype=torch.float32, device=device)
    K = torch.as_tensor(intrinsics, dtype=torch.float32, device=device)
    observations = torch.as_tensor(targets, dtype=torch.float32, device=device)
    dimensions, raster_scale, masks, exterior = _bounds_and_targets(rows, device)
    context = dr.RasterizeCudaContext(device=device)
    faces = body.faces_tensor.to(dtype=torch.int32).contiguous()
    beta = torch.as_tensor(beta, dtype=torch.float32, device=device)
    anchor = torch.as_tensor(anchor, dtype=torch.float32, device=device).reshape(1, 32)
    def parameters(field):
        return _packed_parameters(winners, names, field, device)

    original_root = parameters('global_orient')
    original_root_matrix = axis_angle_to_matrix(original_root)
    original_translation = parameters('camera_translation')
    original_latent = parameters('pose_embedding')
    fixed = {field: parameters(field) for field in FROZEN_FIELDS}
    independent_pose = parameters('body_pose_axis_angle')
    original_geometry = parameters('vertices_camera')
    original_joints = parameters('joints_camera')
    original_pixels = parameters('joints_projected_native')
    chunks = [slice(i, min(count, i+config['batch_size'])) for i in range(0, count, config['batch_size'])]
    anchor_matrix = vposer.decode(anchor)['pose_body_matrot'].reshape(1, 21, 3, 3)
    group_tensors = {name: torch.as_tensor(ids, device=device) for name, ids in GROUPS.items()}
    screened = torch.as_tensor(np.concatenate([BODY, FEET]), device=device)
    face_weights, face_support = _face_regions(observations, rows[0]['wh'])
    face_baseline = _face_stats((original_pixels-observations[..., :2]).norm(dim=-1), face_weights, face_support)
    face_points = face_support.sum(-1)
    neck_head_ids = torch.as_tensor(NECK_HEAD, device=device)

    # Store the actual per-frame decode inputs. Recomputing the mean is only
    # a gauge change: it never clips or moves an unedited frame's latent.
    initial = torch.cat((original_latent, original_root.new_zeros(count, 12)), dim=1)
    scale = initial.new_tensor([config['residual_latent_lr']]*32 + [config['root_lr']]*3
                               + [config['translation_lr']]*3 + [config['neck_head_lr']]*6)
    observed_regions = face_points > 0
    face_limit = face_baseline + config['face_guard_tolerance_px'] - config['replay_margin_px']

    def unpack(state, slots):
        latent, root_delta, trans_delta, neck_delta = state.split((32, 3, 3, 6), dim=-1)
        rotation = _bounded_rows(root_delta, config['root_update_max_rad'])
        root = matrix_to_axis_angle(axis_angle_to_matrix(rotation) @ original_root_matrix[slots])
        displacement = _bounded_rows(trans_delta, config['translation_update_max_m'])
        decoded = vposer.decode(latent)
        angles = decoded['pose_body'].reshape(-1, 21, 3)
        matrices = decoded['pose_body_matrot'].reshape(-1, 21, 3, 3)
        compensation = _bounded_rows(neck_delta.reshape(-1, 2, 3), config['neck_head_update_max_rad'])
        corrections = axis_angle_to_matrix(compensation) @ matrices[:, neck_head_ids]
        pose = angles.index_copy(1, neck_head_ids, matrix_to_axis_angle(corrections)).reshape(-1, 63)
        return root, original_translation[slots]+displacement, pose, matrices, rotation, displacement, compensation

    def pose_penalty(state, matrices, basis):
        shared_matrix = vposer.decode(basis)['pose_body_matrot'].reshape(1, 21, 3, 3)
        return (config['local_rotation_weight']*(matrices-shared_matrix).square().mean((1, 2, 3))
                + config['latent_residual_weight']*(state[:, :32]-basis).square().mean(1)
                + config['anchor_weight']*(shared_matrix-anchor_matrix).square().mean())

    def energy(state, slots, basis):
        root, translation, body_pose, body_matrices, rotation, displacement, compensation = unpack(state, slots)
        vertices, joints = _shape_output(body, beta, {k: v[slots] for k, v in fixed.items()},
                                       root, translation, body_pose)
        projected = _project(joints, K[slots])
        pixel_delta = projected-observations[slots, :, :2]
        pixel_error = pixel_delta.square().sum(-1).clamp_min(1e-12).sqrt()
        distance2 = (pixel_delta/bbox_scale[slots, None, None]).square().sum(-1)
        robust = torch.sqrt(1.+distance2)-1.
        confidence = observations[slots, :, 2]
        reliable = confidence >= config['reliable_confidence']
        weights = confidence.square()*reliable
        reprojection = robust.new_zeros(len(root))
        for group, field in (('body', 'body_weight'), ('feet', 'feet_weight'), ('hands', 'hand_weight')):
            ids = group_tensors[group]
            reprojection += config[field]*(robust[:, ids]*weights[:, ids]).sum(1)/weights[:, ids].sum(1).clamp_min(.01)
        # Two native pixels is on the order of the paired original/flip
        # detector repeatability. The guard separately limits regression.
        unit = config['face_unit_px']
        region_weights = face_weights[slots]
        face_robust = (torch.sqrt(1.+(pixel_error/unit).square())-1.)[:, None, :]
        face_region_loss = (face_robust*region_weights).sum(-1)/region_weights.sum(-1).clamp_min(1e-12)
        face_loss = face_region_loss.sum(-1)/observed_regions[slots].sum(-1).clamp_min(1)
        face_stats = _face_stats(pixel_error, region_weights, face_support[slots])
        face_excess = (face_stats-face_baseline[slots]).masked_fill(~observed_regions[slots, :, None], 0.)
        # The worst protected region per frame contributes a gradient as soon as
        # it starts to regress, before reaching the hard checkpoint tolerance.
        face_barrier = (face_excess.clamp_min(0.)/unit).amax((1, 2)).square()
        # A bounded-influence penalty on the worst supported body/foot error
        # complements regional RMS. Selection still applies its native gate.
        excess = (torch.sqrt(distance2[:, screened]+1e-8)-.75).clamp_min(0.)
        tail = ((torch.sqrt(1.+excess.square())-1.)*confidence[:, screened]*reliable[:, screened]).amax(1)
        alpha = _raster_silhouette(dr, context, vertices, K[slots], faces, dimensions, raster_scale, antialias=True)
        silhouette = (alpha*exterior[slots]).sum((1, 2))/masks[slots].sum((1, 2))
        # One-sided silhouette allows clothing to extend beyond the bare mesh.
        trust = (config['root_trust_weight']*(rotation/config['root_update_max_rad']).square().sum(1)
                 +config['translation_trust_weight']*(displacement/config['translation_update_max_m']).square().sum(1)
                 +config['neck_head_trust_weight']*(compensation/config['neck_head_update_max_rad']).square().sum((1, 2)))
        penalty = pose_penalty(state, body_matrices, basis)
        depth = vertices[..., 2].amin(1)
        depth_penalty = 40.*(.10-depth).clamp_min(0.).square()
        screened_positive = confidence[:, screened] > 0
        max_screen = (distance2[:, screened].masked_fill(~screened_positive, 0.).amax(1)+1e-12).sqrt()
        eligible = screened_positive[:, :len(BODY)].any(1) & (max_screen <= 1.)
        terms = torch.stack([reprojection, config['face_weight']*face_loss, config['face_guard_weight']*face_barrier,
                             config['screen_tail_weight']*tail, config['silhouette_weight']*silhouette,
                             penalty, trust, depth_penalty], dim=1)
        return terms, depth, eligible, face_stats, max_screen

    @torch.no_grad()
    def replay_check():
        geometry_error, joint_error, projection_error = 0., 0., 0.
        for slots in chunks:
            vertices, joints = _shape_output(body, beta, {k: v[slots] for k, v in fixed.items()},
                original_root[slots], original_translation[slots], independent_pose[slots])
            geometry_error = max(geometry_error, float((vertices-original_geometry[slots]).abs().max()))
            joint_error = max(joint_error, float((joints-original_joints[slots]).abs().max()))
            projection_error = max(projection_error, float((_project(joints, K[slots])-original_pixels[slots]).abs().max()))
        if geometry_error > 1e-5 or joint_error > 1e-5 or projection_error > .015:
            raise ValueError(f'Joint SMPL-X preflight differs from named winner: {geometry_error:.6g} m, '
                             f'{joint_error:.6g} m joints, {projection_error:.6g} px.')
        return dict(winner_vertices_max_abs_m=geometry_error, winner_joints_max_abs_m=joint_error,
                    winner_projection_max_abs_px=projection_error, mapped_keypoints=original_pixels.shape[1])

    def latent_basis(state):
        return state[:, :32].mean(0, keepdim=True)

    def residual_bound(state):
        return (state[:, :32]-latent_basis(state)).norm(dim=-1).max() <= config['latent_residual_l2_limit']

    def feasible(part, slots, eligible):
        _, depth, passes_body, face, _ = part
        return ((depth > .05) & torch.isfinite(part[0]).all(1) & torch.isfinite(face).all((1, 2))
                & ((face <= face_limit[slots]) | ~observed_regions[slots, :, None]).all((1, 2))
                & (~eligible | passes_body))

    @torch.no_grad()
    def evaluate(state):
        basis = latent_basis(state)
        parts = [energy(state[slots], slots, basis) for slots in chunks]
        return tuple(torch.cat(values) for values in zip(*parts))

    def checkpoint(step, part):
        terms, depth, eligible, face, _ = part
        values = terms.mean(0)
        coverage = _checkpoint_coverage(names, eligible.cpu().tolist(), scope)
        _, failures = _guard_details(names, face_baseline.cpu(), face.cpu(), face_points.cpu())
        components = dict(zip(('reprojection', 'face_native_reprojection', 'face_guard_barrier', 'pose_screen_tail',
                               'silhouette_exterior', 'shared_pose', 'pose_trust', 'minimum_depth'), values.tolist()))
        return dict(step=step, fraction=1., objective=float(values.sum()), components=components,
                    minimum_depth_m=float(depth.min()), face_guard_passed=not failures,
                    face_guard_failures=failures, **coverage)

    def direction(gradient, units):
        scaled = gradient*units
        if not torch.isfinite(scaled).all():
            raise RuntimeError('Nonfinite block gradient in joint SMPL-X fitting.')
        return -scaled/scaled.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def local_trial(old, slots, basis, eligible, objective_limit, radius, repair):
        parameter = old.detach().requires_grad_(True)
        current = energy(parameter, slots, basis)
        base_score = current[0].sum(1).detach()
        base_screen = current[4].detach()
        body_priority = repair & ~eligible & (base_screen > 1.)
        loss = torch.where(body_priority, current[4], current[0].sum(1))
        gradient, = torch.autograd.grad(loss.sum(), parameter, retain_graph=True)
        step = direction(gradient, scale)*radius[:, None]
        take = torch.zeros_like(eligible, dtype=torch.int64)
        chosen = old.clone()
        metrics = [value.detach().clone() for value in current]

        @torch.no_grad()
        def try_step(update, fraction, code):
            proposal = old + update*scale*fraction
            part = energy(proposal, slots, basis)
            improvement = torch.where(body_priority,
                (part[4] < base_screen-1e-6) & (part[0].sum(1) <= objective_limit+1e-6),
                part[0].sum(1) < base_score-1e-7)
            accept = (take == 0) & feasible(part, slots, eligible) & improvement
            take[accept] = code
            chosen[accept] = proposal[accept]
            for result, value in zip(metrics, part):
                result[accept] = value[accept]
            return part

        full = try_step(step, 1., 1)
        active = ((full[3] > face_limit[slots]) | (current[3].detach() >= face_limit[slots]-.08))
        active &= observed_regions[slots, :, None] & (take == 0)[:, None, None]
        constraints = torch.nonzero(active.flatten(1).any(0), as_tuple=False).flatten().tolist()
        if constraints:
            stats = current[3].flatten(1)
            threatened = active.flatten(1)
            normals = []
            for metric in constraints:
                normal, = torch.autograd.grad(stats[:, metric].masked_fill(~threatened[:, metric], 0.).sum(),
                                               parameter, retain_graph=True)
                normals.append(normal*scale)
            slack = (face_limit[slots].flatten(1)[:, constraints]-stats[:, constraints].detach()).clamp_min(0.)
            step_projected = _project_feasible_step(step, torch.stack(normals, 1), slack, config['projection_sweeps'])
            try_step(step_projected, 1., 2)
        for fraction in config['trial_fractions'][1:]:
            if not bool((take == 0).any()):
                break
            if constraints:
                try_step(step_projected, fraction, 4)
            if bool((take == 0).any()):
                try_step(step, fraction, 3)
        full_rejections = torch.stack((
            ((full[3] > face_limit[slots]) & observed_regions[slots, :, None]).any((1, 2)),
            eligible & ~full[2], full[1] <= .05), dim=1)
        return chosen, tuple(metrics), take, gradient.detach().norm(dim=-1), full_rejections

    @torch.no_grad()
    def commit_local(state, current, proposed, metrics, accepted, start_objective):
        # Local candidates were scored against a fixed basis. Refresh exactly
        # the only cross-frame term without re-rendering any silhouette.
        new = state.clone()
        chosen = torch.zeros_like(accepted)
        result = tuple(value.clone() for value in current)

        def commit(rows):
            new[rows] = proposed[rows]
            if bool(residual_bound(new)):
                terms = result[0].clone()
                terms[rows] = metrics[0][rows]
                matrices = vposer.decode(new[:, :32])['pose_body_matrot'].reshape(count, 21, 3, 3)
                terms[:, 5] = pose_penalty(new, matrices, latent_basis(new))
                objective = terms.sum()/count
                screen_gain = ((~result[2][rows]) & (metrics[4][rows] < result[4][rows]-1e-6)).any()
                if bool((objective <= result[0].sum()/count+1e-7) |
                        (screen_gain & (objective <= start_objective+1e-7))):
                    result[0].copy_(terms)
                    for before, candidate in zip(result[1:], metrics[1:]):
                        before[rows] = candidate[rows]
                    chosen[rows] = True
                    return
            new[rows] = state[rows]
            if len(rows) > 1:
                half = len(rows)//2
                commit(rows[:half])
                commit(rows[half:])

        rows = torch.nonzero(accepted, as_tuple=False).flatten()
        if len(rows):
            commit(rows)
        return new, result, chosen

    # The metric gives a common step and the RMS per-frame compensation
    # comparable scales. A common step leaves centered residuals unchanged.
    shared_units = initial.new_full((32,), config['shared_latent_lr'])
    compensation_units = scale[32:]*count**.5
    units = torch.cat((shared_units, compensation_units))
    face_columns = len(FACE_REGIONS)*2
    smooth_terms = [0, 1, 5, 6]  # Reprojection, shared prior, pose trust. The other terms use a max/min or raster.
    metric_names = [f'{region}_{stat}' for region in FACE_REGIONS for stat in ('rms_px', 'peak_px')]
    metric_names += ['body_feet_max_over_limit', 'negative_depth_m']

    def constraint_values(part):
        return torch.cat((part[3].flatten(1), part[4][:, None], -part[1][:, None]), dim=1)

    def shared_linearization(state, supported, audit):
        common = torch.zeros_like(shared_units)
        common_smooth = torch.zeros_like(shared_units)
        local, local_smooth, jacobians = [], [], []
        for slots in chunks:
            rows = state[slots].detach().requires_grad_(True)
            basis = latent_basis(state).detach().requires_grad_(True)
            part = energy(rows, slots, basis)
            loss = part[0].sum()/count
            row_grad, basis_grad = torch.autograd.grad(loss, (rows, basis), retain_graph=True)
            common += (row_grad[:, :32].sum(0)+basis_grad[0])*shared_units
            local.append(row_grad[:, 32:]*compensation_units)
            if audit:
                smooth = part[0][:, smooth_terms].sum()/count
                rg, bg = torch.autograd.grad(smooth, (rows, basis), retain_graph=True)
                common_smooth += (rg[:, :32].sum(0)+bg[0])*shared_units
                local_smooth.append(rg[:, 32:]*compensation_units)
            stats = constraint_values(part)
            columns = []
            for metric, visible in enumerate(supported[slots].unbind(1)):
                if bool(visible.any()):
                    derivative, = torch.autograd.grad(stats[:, metric].masked_fill(~visible, 0.).sum(),
                                                       rows, retain_graph=True)
                    columns.append(derivative*units)
                else:
                    columns.append(torch.zeros_like(rows))
            jacobians.append(torch.stack(columns, dim=1))
        def packed(common, rows):
            return torch.cat((common, torch.cat(rows).flatten())).double().cpu().numpy()
        gradient = packed(common, local)
        smooth_gradient = packed(common_smooth, local_smooth) if audit else None
        jacobian = torch.cat(jacobians).cpu().numpy()
        if not np.isfinite(gradient).all() or not np.isfinite(jacobian).all():
            raise RuntimeError('The shared objective gradient or constraint linearization is nonfinite.')
        return gradient, smooth_gradient, jacobian

    def apply_shared(state, step):
        scaled = state.new_tensor(step)
        return state + torch.cat(((scaled[:32]*shared_units).expand(count, -1),
                                  scaled[32:].reshape(count, -1)*compensation_units), dim=1)

    def reduction(part):
        terms = part[0].double().mean(0)
        full = float(terms.sum())
        return full, float(terms[smooth_terms].sum())

    @torch.no_grad()
    def derivative_audit(state, current, gradient, smooth, matrix, owners, metrics, projected):
        common = gradient.copy()
        common[32:] = 0.
        baseline = constraint_values(current).cpu().numpy()[owners, metrics]
        records = []
        for name, raw in (('common_only', -common), ('coupled', projected)):
            size = np.linalg.norm(raw)
            if size <= 1e-12:
                records.append(dict(kind=name, direction_norm=float(size), samples=[], nonzero=False))
                continue
            probe = raw/size
            prediction = matrix @ probe
            analytic, smooth_analytic = float(gradient @ probe), float(smooth @ probe)
            samples = []
            for h in config['shared_fd_steps']:
                plus = evaluate(apply_shared(state, h*probe))
                minus = evaluate(apply_shared(state, -h*probe))
                a, ap = reduction(plus)
                b, bp = reduction(minus)
                measured = (a-b)/(2*h)
                smooth_measured = (ap-bp)/(2*h)
                p = constraint_values(plus).cpu().numpy()[owners, metrics].astype(np.float64)
                m = constraint_values(minus).cpu().numpy()[owners, metrics].astype(np.float64)
                slopes = (p-m)/(2*h)
                error = np.abs(slopes-prediction)
                detail = {}
                for label, included in (('face_px', metrics < face_columns),
                    ('body_fraction_of_screen_limit', metrics == face_columns),
                    ('depth_m', metrics == face_columns+1)):
                    if np.any(included):
                        expected, observed = prediction[included], slopes[included]
                        delta = error[included]
                        rms = float(np.linalg.norm(delta)/np.sqrt(len(delta)))
                        expected_rms = float(np.linalg.norm(expected)/np.sqrt(len(expected)))
                        noise_floor = {'face_px': .04, 'body_fraction_of_screen_limit': .0002, 'depth_m': .00003}[label]
                        detail[label] = dict(observations=int(included.sum()), max_abs_error=float(delta.max()),
                            p90_abs_error=float(np.quantile(delta, .9)), rms_abs_error=rms,
                            linearization_passed=rms <= max(noise_floor, .15*expected_rms),
                            relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(expected), 1e-5)),
                            predicted_slope_norm=float(np.linalg.norm(expected)),
                            actual_slope_norm=float(np.linalg.norm(observed)),
                            max_forward_prediction_error=float(np.abs(
                                p[included]-baseline[included]-h*expected).max()))
                worst = int(error.argmax())
                samples.append(dict(h=h, finite_difference=measured, smooth_finite_difference=smooth_measured,
                    smooth_gradient_abs_error=abs(smooth_measured-smooth_analytic),
                    smooth_gradient_passed=abs(smooth_measured-smooth_analytic) <=
                        max(5e-5, .08*abs(smooth_analytic)),
                    linearization=detail, worst_frame=names[owners[worst]], worst_metric=metric_names[metrics[worst]]))
            records.append(dict(kind=name, direction_norm=float(size), analytic_derivative=analytic,
                smooth_analytic_derivative=smooth_analytic, samples=samples, nonzero=True))
        return dict(checked_frames=count, directions=records,
                    smooth_derivative_passed=all(record['samples'] and any(
                        sample['smooth_gradient_passed'] for sample in record['samples']) for record in records),
                    constraint_linearization_passed=all(record['samples'] and any(all(
                        value['linearization_passed'] for value in sample['linearization'].values())
                        for sample in record['samples']) for record in records),
                    smooth_components=['reprojection', 'face_native_reprojection', 'shared_pose', 'pose_trust'],
                    full_gradient_includes_antialiased_silhouette_and_extrema=True)

    def shared_trial(state, current, radius, audit):
        # Use paired forwards for small objective differences. Do not let a
        # recomputation of a boundary row silently remove its body protection.
        retained_eligible = current[2]
        current = evaluate(state)
        if bool((retained_eligible & ~current[2]).any()):
            raise RuntimeError('Shared pretrial replay lost a protected body screen result.')
        supported = torch.cat((observed_regions.repeat_interleave(2, dim=1),
                               current[2][:, None], torch.ones_like(current[2][:, None])), dim=1)
        limits = torch.cat((face_limit.flatten(1), torch.ones_like(current[4][:, None]),
                            torch.full_like(current[1][:, None], -.05)), dim=1)
        slack_full = (limits-constraint_values(current)).clamp_min(0.)
        gradient, smooth, jacobian = shared_linearization(state, supported, audit)
        matrix, owners, metrics, norm = row_constraints(jacobian, supported.cpu().numpy(), 32)
        slack = slack_full.cpu().numpy()[owners, metrics].astype(np.float64)
        normalized = matrix.multiply(1/np.maximum(norm[:, None], 1e-12)).tocsr()
        slack_normalized = slack/np.maximum(norm, 1e-12)
        gradient_norm = float(np.linalg.norm(gradient))
        target = -gradient/max(gradient_norm, config['shared_projected_stop'])
        working = (norm > 1e-12) & (slack_normalized <= radius+1e-6)
        def projected_trial(target):
            return project_halfspaces(target*radius, normalized[working], slack_normalized[working])

        step, projection = projected_trial(target)
        common_target = np.zeros_like(gradient)
        common_target[:32] = -gradient[:32]/max(np.linalg.norm(gradient[:32]), config['shared_projected_stop'])
        common_step, common_projection = projected_trial(common_target)
        # Use genuinely near-active rows for the cone check. Positive-slack
        # constraints still enter the trial projection above.
        thresholds = np.where(metrics < face_columns, .002, np.where(metrics == face_columns, .0002, .00002))
        active = (norm > 1e-12) & (slack <= thresholds)
        tangent, tangent_proof = project_halfspaces(target, normalized[active], np.zeros(int(active.sum())))
        tangent_norm = float(np.linalg.norm(tangent))*max(gradient_norm, config['shared_projected_stop'])
        stationarity_rows = (norm > 1e-12) & (slack_normalized <= gradient_norm+1e-6)
        projected_gradient, stationarity_proof = project_halfspaces(
            -gradient, normalized[stationarity_rows], slack_normalized[stationarity_rows])
        projected_norm = float(np.linalg.norm(projected_gradient))
        active_counts = {label:int(np.count_nonzero(active & (metrics == i)))
                         for i,label in enumerate(metric_names)}
        diagnostic = dict(trust_radius_before=radius, gradient_norm=gradient_norm,
            projected_gradient_norm=projected_norm, tangent_gradient_norm=tangent_norm, step_norm=float(np.linalg.norm(step)),
            common_gradient_norm=float(np.linalg.norm(gradient[:32])),
            common_direction_norm=float(np.linalg.norm(step[:32])),
            compensation_direction_norm=float(np.linalg.norm(step[32:])),
            common_predicted_decrease=-float(gradient[:32] @ step[:32]),
            compensation_predicted_decrease=-float(gradient[32:] @ step[32:]),
            projection=projection, common_guided_projection=common_projection, tangent_projection=tangent_proof,
            stationarity_projection=stationarity_proof, active_constraints=active_counts)
        fd = derivative_audit(state, current, gradient, smooth, matrix, owners, metrics, step) if audit else None
        attempts = []
        accepted = False
        base, _ = reduction(current)
        with torch.no_grad():
            for kind, step_direction in (('common_guided', common_step), ('objective_guided', step)):
                parts = (-float(gradient[:32] @ step_direction[:32]), -float(gradient[32:] @ step_direction[32:]))
                prediction = sum(parts)
                if kind == 'common_guided' and prediction <= 1e-12:
                    diagnostic['common_guided_skipped_predicted_decrease'] = prediction
                    continue
                for fraction in config['trial_fractions']:
                    candidate = apply_shared(state, step_direction*fraction)
                    part = evaluate(candidate)
                    passes = feasible(part, slice(None), current[2])
                    stats = constraint_values(part)
                    failed = (stats > limits) & supported
                    residual_ok = bool(residual_bound(candidate))
                    value, _ = reduction(part)
                    gain = base-value
                    predicted = prediction*fraction
                    ratio = gain/predicted if predicted > 1e-12 else None
                    descent = gain > 1e-9
                    error = np.abs((stats-constraint_values(current)).cpu().numpy()[owners, metrics]
                                   - fraction*(matrix @ step_direction))
                    attempts.append(dict(direction_kind=kind, fraction=fraction, checked_frames=count,
                        failed_frames=int((~passes).sum()), failed_face_frames=int(failed[:, :face_columns].any(1).sum()),
                        failed_body_frames=int(failed[:, face_columns].sum()),
                        failed_depth_frames=int((part[1] <= .05).sum()), residual_bound_passed=residual_ok,
                        objective=value, descent=descent, predicted_decrease=predicted, actual_decrease=gain, ratio=ratio,
                        common_predicted_decrease=parts[0]*fraction, compensation_predicted_decrease=parts[1]*fraction,
                        common_latent_step_l2=float(np.linalg.norm(step_direction[:32])*config['shared_latent_lr']*fraction),
                        compensation_max_step=float((candidate[:, 32:]-state[:, 32:]).norm(dim=1).max()),
                        max_face_linearization_error_px=float(error[metrics < face_columns].max(initial=0.)),
                        max_body_linearization_error=float(error[metrics == face_columns].max(initial=0.)),
                        max_depth_linearization_error_m=float(error[metrics == face_columns+1].max(initial=0.))))
                    if (bool(passes.all()) and residual_ok and descent and ratio is not None
                            and ratio >= config['shared_min_gain_ratio']):
                        state, current, accepted = candidate, part, True
                        break
                if accepted:
                    break
        if accepted:
            outcome = 'accepted_exact_descent'
            multiplier = 1.35 if fraction == 1. and ratio >= .75 else max(.5, fraction)
        else:
            outcome = 'unresolved_no_accepted_trial'
            multiplier = .5
        radius = float(np.clip(radius*multiplier, config['shared_trust_min'], config['shared_trust_max']))
        stationary = (not accepted and tangent_proof['certified'] and projection['certified']
                      and max(stationarity_proof['primal_residual'], stationarity_proof['dual_residual']) <= 1e-6
                      and projected_norm <= config['shared_projected_stop']
                      and all(trial['actual_decrease'] <= 1e-9 for trial in attempts))
        if stationary:
            outcome = 'small_projected_gradient_and_no_trial_descent'
        diagnostic.update(trust_radius_after=radius, outcome=outcome, stationarity_check_passed=stationary)
        return state, current, accepted, attempts, diagnostic, radius, fd

    preflight = replay_check()
    state = initial
    current = evaluate(state)
    trace = [checkpoint(0, current)]
    initial_eligible = current[2].tolist()
    if not trace[0]['face_guard_passed'] or not bool(residual_bound(state)) or trace[0]['minimum_depth_m'] <= .05:
        raise RuntimeError('The named official initializers are outside the joint block domain.')
    start_objective = current[0].sum()/count
    start_scores = current[0].sum(1)
    trust = state.new_ones(count)
    local_counts = torch.zeros((count, 9), dtype=torch.int64, device=device)
    sweep_records = []
    stable = 0
    stop_reason = 'maximum_sweeps_unconverged'
    shared_radius, shared_preflight = config['shared_trust_initial'], None
    recent_shared_stationary = 0
    for sweep in range(1, config['max_sweeps']+1):
        previous = current
        repair = sweep % 4 == 3
        basis = latent_basis(state)
        trials = [local_trial(state[slots], slots, basis, current[2][slots], start_scores[slots],
                              trust[slots], repair) for slots in chunks]
        proposed, metrics, codes, gradients, full_rejections = zip(*trials)
        codes = torch.cat(codes)
        basis_before = latent_basis(state)
        state, current, accepted = commit_local(state, current, torch.cat(proposed),
            tuple(torch.cat(values) for values in zip(*metrics)), codes > 0, start_objective)
        local_counts[:, 0] += accepted
        local_counts[:, 1] += accepted & ((codes == 2) | (codes == 4))
        local_counts[:, 2] += accepted & (codes >= 3)
        local_counts[:, 3] += codes == 0
        local_counts[:, 4] += ~accepted & (codes > 0)
        local_counts[:, 5] += accepted & repair & ~previous[2]
        local_counts[:, 6:] += torch.cat(full_rejections)
        trust = torch.where(accepted & (codes <= 2), trust*1.2, trust*.7).clamp(config['trust_min'], 1.)
        local_basis_shift = float((latent_basis(state)-basis_before).norm())
        shared_accepted, shared_checks, shared_diagnostics = False, [], None
        if sweep % config['shared_every'] == 0:
            state, current, shared_accepted, shared_checks, shared_diagnostics, shared_radius, fd = shared_trial(
                state, current, shared_radius, audit=shared_preflight is None)
            if fd is not None:
                shared_preflight = fd
            recent_shared_stationary = (recent_shared_stationary+1 if
                shared_diagnostics['stationarity_check_passed'] else 0)
        delta = float((previous[0].sum()-current[0].sum())/previous[0].sum().abs().clamp_min(1.))
        screen_delta = float((previous[4]-current[4]).masked_fill(previous[2], 0.).abs().max())
        new_eligible = int((current[2] & ~previous[2]).sum())
        row = dict(sweep=sweep, repair_direction=repair, local_accepted=int(accepted.sum()),
                   local_projected_accepted=int((accepted & ((codes == 2) | (codes == 4))).sum()),
                   shared_accepted=shared_accepted, shared_checks=shared_checks, shared_diagnostics=shared_diagnostics,
                   new_eligible=new_eligible, local_basis_shift_l2=local_basis_shift,
                   objective=float(current[0].sum()/count), objective_relative_change=delta,
                   screen_change=screen_delta, max_local_gradient=float(torch.cat(gradients).max()),
                   max_face_delta_px=float((current[3]-face_baseline).masked_fill(~observed_regions[..., None], 0.).max()))
        sweep_records.append(row)
        if new_eligible or abs(delta) > config['objective_relative_stop'] or screen_delta > config['screen_stop']:
            stable = 0
        else:
            stable += 1
        converged = (stable >= config['stable_sweeps'] and
                     recent_shared_stationary >= config['shared_stationary_checks'] and
                     shared_preflight is not None and shared_preflight['smooth_derivative_passed'] and
                     shared_preflight['constraint_linearization_passed'])
        if sweep % 4 == 0 or sweep == config['max_sweeps'] or converged:
            record = checkpoint(sweep, current)
            trace.append(record)
            if not record['face_guard_passed']:
                raise RuntimeError('Accepted joint block crossed the protected native face limit.')
            if progress_path is not None:
                rt.save(Path(progress_path), dict(state='joint_fit', phase='block_feasible_joint_smplx',
                    completed_steps=sweep, total=config['max_sweeps'], candidate_frames=count,
                    checkpoint_objective=record['objective'], eligible_frames=record['pose_gate_eligible'],
                    accepted_local=int(local_counts[:, 0].sum()),
                    accepted_shared=sum(value['shared_accepted'] for value in sweep_records)))
        if converged:
            stop_reason = 'shared_stationary_local_progress'
            break

    best_record = trace[-1]
    changed = (state != initial).any(1)
    block_solver = dict(max_sweeps=config['max_sweeps'], completed_sweeps=sweep, stop_reason=stop_reason,
        converged=stop_reason == 'shared_stationary_local_progress', shared_preflight=shared_preflight,
        convergence_scope='Shared projected stationarity with sustained small local objective and screen changes.',
        local_accepted=int(local_counts[:, 0].sum()), local_projected_accepted=int(local_counts[:, 1].sum()),
        shared_attempted=sum(bool(row['shared_checks']) for row in sweep_records),
        shared_accepted=sum(row['shared_accepted'] for row in sweep_records), sweeps=sweep_records,
        per_frame={name: dict(zip(('local_accepted', 'projected_accepted', 'backtracked_accepted', 'no_feasible_descent',
                                 'joint_objective_or_bound_rejection', 'body_repair_accepted',
                                 'full_trial_face_violation', 'full_trial_body_violation', 'full_trial_depth_violation'),
                                values), changed=bool(flag), initial_body_eligible=initial_ok, final_body_eligible=final_ok)
                   for name, values, flag, initial_ok, final_ok in zip(
                       names, local_counts.tolist(), changed.tolist(), initial_eligible, current[2].tolist())})

    @torch.no_grad()
    def summarize():
        latent = state[:, :32]
        shared = latent_basis(state)
        pose_residual = latent-shared
        shared_pose = vposer.decode(shared)
        root, translation, pose, _, rotation, displacement, compensation = unpack(state, slice(None))
        root = torch.where(changed[:, None], root, original_root)
        translation = torch.where(changed[:, None], translation, original_translation)
        pose = torch.where(changed[:, None], pose, independent_pose)
        previous_rotation = axis_angle_to_matrix(independent_pose.reshape(count, 21, 3))
        final_rotation = axis_angle_to_matrix(pose.reshape(count, 21, 3))
        anchor_rotation = shared_pose['pose_body_matrot'].reshape(1, 21, 3, 3)

        before_angle = _mean_joint_angle_degrees(previous_rotation, anchor_rotation).cpu().numpy()
        after_angle = _mean_joint_angle_degrees(final_rotation, anchor_rotation).cpu().numpy()
        change_angle = _mean_joint_angle_degrees(final_rotation, previous_rotation).cpu().numpy()
        before_silhouette, after_silhouette = [], []

        def silhouette_metrics(vertices, slots):
            predicted = _raster_silhouette(dr, context, vertices, K[slots], faces, dimensions, raster_scale,
                                          antialias=False) > .5
            truth = masks[slots]
            iou = (predicted & truth).sum((1, 2))/(predicted | truth).sum((1, 2)).clamp_min(1)
            outside = (predicted & ~truth).sum((1, 2))/predicted.sum((1, 2)).clamp_min(1)
            return [dict(raster_wh=list(dimensions), iou=float(i), exterior_fraction=float(o))
                    for i, o in zip(iou.tolist(), outside.tolist())]

        for slots in chunks:
            mesh = _shape_output(body, beta, {k: v[slots] for k, v in fixed.items()},
                                 root[slots], translation[slots], pose[slots])[0]
            before_silhouette += silhouette_metrics(original_geometry[slots], slots)
            after_silhouette += silhouette_metrics(mesh, slots)
        summary = dict(state='complete', config=config.copy(), candidate_frames=names,
            optimizer_updates=sweep, block_solver=block_solver,
            shared_betas=beta.cpu().tolist(), shared_latent_anchor=anchor.cpu().tolist(),
            shared_latent_final=shared.cpu().tolist(), shared_body_pose_axis_angle_final=shared_pose['pose_body'].reshape(1, 63).cpu().tolist(),
            pose_residual_mean_abs_max=float(pose_residual.mean(0).abs().max()),
            pose_residual_l2_max=float(pose_residual.norm(dim=1).max()),
            objective_initial=trace[0]['objective'], objective_final=best_record['objective'],
            objective_min=min(record['objective'] for record in trace),
            objective_final_step=best_record['step'], objective_trace=trace, preflight=preflight,
            chosen_checkpoint=best_record,
            face_guard=dict(reference='official_initializer', threshold_px=config['face_guard_tolerance_px'],
                confidence_threshold=config['face_guard_confidence'], metric='confidence-squared native RMS and peak',
                regions={name:indices.tolist() for name,indices in FACE_REGIONS.items()},
                fallback=None),
            pose_embedding_semantics='Effective VPoser latent. body_pose_axis_angle includes bounded neck/head composition.',
            checkpoint_choice='Final feasible state. Local exact face/depth, preserve eligible rows, exact shared-basis prior. '
                              'Body repair can trade objective only within the original budget. Shared moves check all frames.',
            silhouette_supervision='one-sided exterior of observed mask; reduced aspect-aware corner raster',
            silhouette_raster_wh=list(dimensions), silhouette_native_to_raster_scale_xy=list(raster_scale),
            frames={frame:dict(before=dict(silhouette=before_silhouette[i]),
                after=dict(silhouette=after_silhouette[i]), body_mean_joint_change_deg=float(change_angle[i]),
                local_pose_mean_distance_to_final_shared_before_deg=float(before_angle[i]),
                local_pose_mean_distance_to_final_shared_after_deg=float(after_angle[i]),
                latent_residual_l2=float(pose_residual[i].norm()),
                root_change_deg=float(torch.rad2deg(rotation[i].norm())),
                neck_head_compensation_deg=torch.rad2deg(compensation[i].norm(dim=1)).tolist(),
                translation_change_m=float(displacement[i].norm())) for i, frame in enumerate(names)})
        values = (root, translation, pose, latent)
        return [a.detach().cpu().numpy() for a in values], summary, height

    return summarize()


def refine_joint_body(*, inputs, winners, body, vposer, shared_betas, shared_latent, progress_path=None):
    """Preserve official shape and calibration while jointly fitting train-only body poses."""
    scope = inputs['scope']
    names = scope['fit_frame_ids']
    held = scope['retained_holdouts']
    if len(names) < 2 or len(names) != len(set(names)) or set(names) != set(winners) or set(names) & set(held):
        raise ValueError('Joint fitting requires the complete and disjoint training candidate roster.')
    slots = [inputs['ids'].index(name) for name in names]
    rows = [inputs['rows'][i] for i in slots]
    K, targets = inputs['K'][slots], inputs['targets'][slots]
    beta = np.asarray(shared_betas, dtype=np.float32).reshape(1, 10)
    anchor = np.asarray(shared_latent, dtype=np.float32).reshape(1, 32)
    if not np.isfinite(beta).all() or not np.isfinite(anchor).all():
        raise ValueError('Joint fitting requires a finite shared shape and VPoser anchor.')
    for i, name in enumerate(names):
        fit = winners[name]
        if (str(np.asarray(fit['frame_id']).item()) != name
                or not np.array_equal(fit['betas'], beta) or not np.array_equal(fit['K_integer_centers'], K[i])
                or not np.array_equal(fit['faces'], body.faces)):
            raise ValueError('Named joint winner changed the frozen shape, native K, or topology: '+name)
    body_device = next(body.parameters()).device
    vposer_device = next(vposer.parameters()).device
    models = (body, vposer)
    training = [model.training for model in models]
    trainable = [{name: param.requires_grad for name, param in model.named_parameters()} for model in models]
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        for model in models:
            model.to('cuda').eval().requires_grad_(False)
        (roots, translations, poses, latents), report, heights = _evaluate_joint(
            body=body, vposer=vposer, winners=winners, rows=rows, targets=targets, intrinsics=K,
            beta=beta, anchor=anchor, scope=scope, progress_path=progress_path)
        # Re-evaluate each delivered mesh with the same single-frame body call
        # and original device used by the official fit and named mesh replay.
        body.to(body_device)
        tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device=body_device)
        refined = {}
        with torch.no_grad():
            for i, name in enumerate(names):
                source = winners[name]
                fixed = {key: tensor(source[key]) for key in FROZEN_FIELDS}
                root, translation, pose = roots[i:i+1], translations[i:i+1], poses[i:i+1]
                mesh, joints = _shape_output(body, tensor(beta), fixed, tensor(root), tensor(translation), tensor(pose))
                projected = _project(joints, tensor(K[i]))
                if not torch.isfinite(mesh).all() or not torch.isfinite(projected).all() or float(mesh[..., 2].min()) <= .05:
                    raise ValueError('Joint candidate has invalid native geometry: ' + name)
                pixels = projected.cpu().numpy()
                refined[name] = dict(source, global_orient=root, camera_translation=translation,
                    body_pose_axis_angle=pose, pose_embedding=latents[i:i+1], vertices_camera=mesh.cpu().numpy(),
                    joints_camera=joints.cpu().numpy(), joints_projected_native=pixels)
                if not report['block_solver']['per_frame'][name]['changed']:
                    refined[name] = source
                    pixels = source['joints_projected_native']
                frame = report['frames'][name]
                frame['before']['projection'] = _projection_metrics(targets[i], source['joints_projected_native'][0], heights[i])
                frame['after']['projection'] = _projection_metrics(targets[i], pixels[0], heights[i])
        observed = torch.as_tensor(targets, dtype=torch.float32)
        face_weights, face_support = _face_regions(observed, rows[0]['wh'])

        def measured(poses):
            pixels = _packed_parameters(poses, names, 'joints_projected_native', 'cpu')
            return _face_stats((pixels-observed[..., :2]).norm(dim=-1), face_weights, face_support)

        original_face = measured(winners)
        final_face, failures = _guard_details(names, original_face, measured(refined), face_support.sum(-1))
        report['face_guard']['exact_reprojection_rejected_failures'] = failures
        if failures:
            first = failures[0]
            raise RuntimeError(f"Exact original-device replay failed {len(failures)} native face checks; first: {first}")
        for name, frame in report['frames'].items():
            frame['face_guard'] = final_face[name]
        # Use the same native body/feet test as the actual frame selector on
        # the original-device results, and expose numerical boundary changes.
        def screen(poses):
            slots = np.concatenate((BODY, FEET))
            observed_slots = targets[:, slots]
            supported = observed_slots[..., 2] > 0
            projected = np.concatenate([poses[name]['joints_projected_native'] for name in names], axis=0)[:, slots]
            error = np.linalg.norm(projected-observed_slots[..., :2], axis=-1)
            return (supported[:, :len(BODY)].any(1) &
                    (np.where(supported, error, 0.).max(1)/heights <= CONFIG['reprojection_unit_over_height']))

        before_body, final_body = screen(winners), screen(refined)
        exact = report['block_solver']['exact_body_screen'] = dict(
            initial_eligible=[name for name, passed in zip(names, before_body) if passed],
            final_eligible=[name for name, passed in zip(names, final_body) if passed],
            gpu_cpu_disagreements=[name for name, passed in zip(names, final_body)
                if passed != report['block_solver']['per_frame'][name]['final_body_eligible']])
        if np.any(before_body & ~final_body):
            raise RuntimeError('Original-device replay lost initially eligible body poses: '
                               + ', '.join(name for name, lost in zip(names, before_body & ~final_body) if lost))
        exact['initial_coverage'] = _checkpoint_coverage(names, before_body, scope)
        exact['final_coverage'] = _checkpoint_coverage(names, final_body, scope)
        report['face_guard']['exact_reprojection_passed'] = True
        report['face_guard']['supported_frame_count'] = int(face_support.any((1, 2)).sum())
        report['retained_holdouts'] = list(held)
        report['shared_betas_frozen'] = True
        report['native_K_frozen'] = True
        report['per_frame_scale_parameters'] = 0
        return refined, report
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        for model, device, was_training, flags in zip(models, (body_device, vposer_device), training, trainable):
            model.to(device).train(was_training)
            for name, param in model.named_parameters():
                param.requires_grad_(flags[name])

