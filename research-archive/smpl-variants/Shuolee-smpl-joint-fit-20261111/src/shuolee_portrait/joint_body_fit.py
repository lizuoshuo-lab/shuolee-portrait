"""Jointly refine named training candidates in the fixed camera-relative model."""

import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .coverage_refinement import _clip
from .stable_selection import BODY, FACE, FEET


CONFIG = dict(steps=80, batch_size=8, raster_long_edge=448, reliable_confidence=.15,
              outside_margin_px=1, outside_distance_cap_px=12, reprojection_unit_over_height=.03,
              latent_residual_l2_limit=8., root_update_max_rad=.55, translation_update_max_m=.12,
              shared_latent_lr=.025, residual_latent_lr=.055, root_lr=.013, translation_lr=.0032,
              body_weight=1., feet_weight=.75, face_weight=.55, hand_weight=.12, screen_tail_weight=.45,
              silhouette_weight=.55, local_rotation_weight=.45, latent_residual_weight=.012,
              anchor_weight=.015, root_trust_weight=.012, translation_trust_weight=.018,
              checkpoints=[0, 20, 40, 60, 80])
FROZEN_FIELDS = ('left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression')
GROUPS = {'body': BODY, 'feet': FEET, 'face': FACE, 'hands': np.arange(25, 67)}


def _bounded_rows(values, limit):
    return values * (limit / values.norm(dim=-1, keepdim=True).clamp_min(limit))


def _centered_residual(values, limit):
    centered = values - values.mean(dim=0, keepdim=True)
    clipped = _bounded_rows(centered, limit)
    clipped = clipped - clipped.mean(dim=0, keepdim=True)
    return clipped * (limit / clipped.norm(dim=-1).max().clamp_min(limit))


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

    shared = torch.nn.Parameter(original_latent.mean(0, keepdim=True))
    residual = torch.nn.Parameter(original_latent-shared.detach())
    root_update = torch.nn.Parameter(torch.zeros_like(original_root))
    translation_update = torch.nn.Parameter(torch.zeros_like(original_translation))
    variables = [shared, residual, root_update, translation_update]
    learning_rates = [config[key] for key in ('shared_latent_lr', 'residual_latent_lr', 'root_lr', 'translation_lr')]
    optimizer = torch.optim.Adam([{'params': [param], 'lr': lr} for param, lr in zip(variables, learning_rates)])

    def roots(slots):
        rotation = _bounded_rows(root_update[slots], config['root_update_max_rad'])
        root = matrix_to_axis_angle(axis_angle_to_matrix(rotation) @ original_root_matrix[slots])
        displacement = _bounded_rows(translation_update[slots], config['translation_update_max_m'])
        return root, original_translation[slots]+displacement, rotation, displacement

    def energy(slots):
        residual_used = _centered_residual(residual, config['latent_residual_l2_limit'])[slots]
        latent = shared+residual_used
        decoded = vposer.decode(latent)
        body_pose = decoded['pose_body'].reshape(-1, 63)
        body_matrices = decoded['pose_body_matrot'].reshape(-1, 21, 3, 3)
        shared_matrix = vposer.decode(shared)['pose_body_matrot'].reshape(1, 21, 3, 3)
        root, translation, rotation, displacement = roots(slots)
        vertices, joints = _shape_output(body, beta, {k: v[slots] for k, v in fixed.items()},
                                       root, translation, body_pose)
        projected = _project(joints, K[slots])
        delta = (projected-observations[slots, :, :2])/bbox_scale[slots, None, None]
        distance2 = delta.square().sum(-1)
        robust = torch.sqrt(1.+distance2)-1.
        confidence = observations[slots, :, 2]
        reliable = confidence >= config['reliable_confidence']
        weights = confidence.square()*reliable
        reprojection = robust.new_zeros(len(root))
        for group, field in (('body', 'body_weight'), ('feet', 'feet_weight'),
                             ('face', 'face_weight'), ('hands', 'hand_weight')):
            ids = group_tensors[group]
            reprojection += config[field]*(robust[:, ids]*weights[:, ids]).sum(1)/weights[:, ids].sum(1).clamp_min(.01)
        # A bounded-influence penalty on the worst supported body/foot error
        # complements regional RMS. Selection still applies its native gate.
        excess = (torch.sqrt(distance2[:, screened]+1e-8)-.75).clamp_min(0.)
        tail = ((torch.sqrt(1.+excess.square())-1.)*confidence[:, screened]*reliable[:, screened]).amax(1)
        alpha = _raster_silhouette(dr, context, vertices, K[slots], faces, dimensions, raster_scale, antialias=True)
        silhouette = (alpha*exterior[slots]).sum((1, 2))/masks[slots].sum((1, 2))
        # One-sided silhouette allows clothing to extend beyond the bare mesh.
        local_rotation = (body_matrices-shared_matrix).square().mean((1, 2, 3))
        trust = (config['root_trust_weight']*(rotation/config['root_update_max_rad']).square().sum(1)
                 +config['translation_trust_weight']*(displacement/config['translation_update_max_m']).square().sum(1))
        penalty = (config['local_rotation_weight']*local_rotation
                   +config['latent_residual_weight']*residual_used.square().mean(1)
                   +config['anchor_weight']*(shared_matrix-anchor_matrix).square().mean())
        depth_penalty = 40.*(.10-vertices[..., 2].amin(1)).clamp_min(0.).square()
        screened_positive = confidence[:, screened] > 0
        eligible = (screened_positive[:, :len(BODY)].any(1) &
                    (distance2[:, screened].masked_fill(~screened_positive, 0.).amax(1) <= 1.))
        terms = torch.stack([reprojection, config['screen_tail_weight']*tail,
                             config['silhouette_weight']*silhouette, penalty, trust, depth_penalty]).sum(1)/count
        return terms, vertices[..., 2].amin(), eligible

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

    preflight = replay_check()
    trace, best, best_key, best_record = [], None, None, None
    for step in range(config['steps']+1):
        assess = step in config['checkpoints']
        train = step < config['steps']
        optimizer.zero_grad(set_to_none=True)
        values, depths, eligible = [], [], []
        for slots in chunks:
            with torch.set_grad_enabled(train):
                terms, depth, passed = energy(slots)
                if train:
                    terms.sum().backward()
            values.append(terms.detach())
            depths.append(depth.detach())
            eligible.append(passed.detach())
        if assess:
            terms = torch.stack(values).sum(0)
            objective = float(terms.sum())
            depth = float(torch.stack(depths).min())
            coverage = _checkpoint_coverage(names, torch.cat(eligible).cpu().tolist(), scope)
            components = dict(zip(('reprojection', 'pose_screen_tail', 'silhouette_exterior',
                                   'shared_pose', 'root_translation_trust', 'minimum_depth'), terms.tolist()))
            record = dict(step=step, objective=objective, components=components,
                          minimum_depth_m=depth, **coverage)
            trace.append(record)
            # Only training candidates contribute. Compare whole-sequence
            # checkpoints under the existing positive-confidence pose screen.
            rank = (-coverage['pose_gate_roster_capacity'], -coverage['source_bins_with_pose_gate_eligible'],
                    objective)
            if depth > .05 and objective <= trace[0]['objective'] and (best_key is None or rank < best_key):
                best_key, best_record = rank, record
                best = [param.detach().clone() for param in variables]
            if progress_path is not None:
                rt.save(Path(progress_path), dict(state='joint_fit', phase='bounded_joint_smplx',
                        completed_steps=step, total=config['steps'], candidate_frames=count,
                        checkpoint_objective=objective, **coverage))
        if train:
            norm = torch.nn.utils.clip_grad_norm_(variables, max_norm=3., error_if_nonfinite=True)
            if not torch.isfinite(norm):
                raise RuntimeError(f'Nonfinite gradient in joint fitting at update {step+1}.')
            if step == 0:
                preflight['first_update_gradient_norm'] = float(norm)
                preflight['first_update_parameter_grad_norms'] = dict(
                    zip(('shared_latent', 'residual_latent', 'root_rotation', 'translation'),
                        (float(param.grad.norm()) for param in variables)))
            for group, lr in zip(optimizer.param_groups, learning_rates):
                group['lr'] = lr*(1. if step < 48 else .5 if step < 68 else .25)
            optimizer.step()
    if best is None or not all(np.isfinite(item['objective']) for item in trace):
        raise RuntimeError('Joint fitting has no finite positive-depth checkpoint.')

    @torch.no_grad()
    def summarize():
        for variable, value in zip(variables, best):
            variable.copy_(value)
        pose_residual = _centered_residual(residual, config['latent_residual_l2_limit'])
        latent = shared+pose_residual
        shared_pose = vposer.decode(shared)
        pose = vposer.decode(latent)['pose_body'].reshape(count, 63)
        root, translation, rotation, displacement = roots(slice(None))
        previous_rotation = axis_angle_to_matrix(independent_pose.reshape(count, 21, 3))
        final_rotation = axis_angle_to_matrix(pose.reshape(count, 21, 3))
        anchor_rotation = shared_pose['pose_body_matrot'].reshape(1, 21, 3, 3)

        def mean_angle(a, b):
            delta = (a*b).sum((-1, -2))
            return torch.rad2deg(torch.acos(((delta-1.)*.5).clamp(-1., 1.))).mean(1)

        before_angle = mean_angle(previous_rotation, anchor_rotation).cpu().numpy()
        after_angle = mean_angle(final_rotation, anchor_rotation).cpu().numpy()
        change_angle = mean_angle(final_rotation, previous_rotation).cpu().numpy()
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
            mesh, _ = _shape_output(body, beta, {k: v[slots] for k, v in fixed.items()},
                                   root[slots], translation[slots], pose[slots])
            before_silhouette += silhouette_metrics(original_geometry[slots], slots)
            after_silhouette += silhouette_metrics(mesh, slots)
        summary = dict(state='complete', config=config.copy(), candidate_frames=names,
            optimizer_updates=config['steps'],
            shared_betas=beta.cpu().tolist(), shared_latent_anchor=anchor.cpu().tolist(),
            shared_latent_final=shared.cpu().tolist(), shared_body_pose_axis_angle_final=shared_pose['pose_body'].reshape(1, 63).cpu().tolist(),
            pose_residual_mean_abs_max=float(pose_residual.mean(0).abs().max()),
            pose_residual_l2_max=float(pose_residual.norm(dim=1).max()),
            objective_initial=trace[0]['objective'], objective_final=best_record['objective'],
            objective_min=min(record['objective'] for record in trace),
            objective_final_step=best_record['step'], objective_trace=trace, preflight=preflight,
            checkpoint_choice='positive-depth and objective no worse than start, then existing pose-screen roster capacity, '
                              'eligible chronological source bins, and joint objective',
            silhouette_supervision='one-sided exterior of observed mask; reduced aspect-aware corner raster',
            silhouette_raster_wh=list(dimensions), silhouette_native_to_raster_scale_xy=list(raster_scale),
            frames={frame:dict(before=dict(silhouette=before_silhouette[i]),
                after=dict(silhouette=after_silhouette[i]), body_mean_joint_change_deg=float(change_angle[i]),
                local_pose_mean_distance_to_final_shared_before_deg=float(before_angle[i]),
                local_pose_mean_distance_to_final_shared_after_deg=float(after_angle[i]),
                latent_residual_l2=float(pose_residual[i].norm()),
                root_change_deg=float(torch.rad2deg(rotation[i].norm())),
                translation_change_m=float(displacement[i].norm())) for i, frame in enumerate(names)})
        return [a.detach().cpu().numpy() for a in (root, translation, pose, latent)], summary, height

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
                frame = report['frames'][name]
                frame['before']['projection'] = _projection_metrics(targets[i], source['joints_projected_native'][0], heights[i])
                frame['after']['projection'] = _projection_metrics(targets[i], pixels[0], heights[i])
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

