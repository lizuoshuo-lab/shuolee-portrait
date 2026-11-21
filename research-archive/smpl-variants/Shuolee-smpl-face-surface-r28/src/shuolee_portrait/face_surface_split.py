"""Guarded two-child replacement of oversized trusted skin Gaussians."""

from copy import copy
from math import exp, log

import torch

from .face_surface_geometry import FACE_SURFACE, face_surface_metrics
from .size_child_commit import MOMENTS, PARAMETERS
from .splat_surface_distance import tangent_split_seeds


SPLIT = dict(parent_budget=16, repeat_views=2, min_relative_excess=.05, tangent_shift=.50,
             mean_rgb_increase=.0025, mean_foreground_alpha_drop=.0025, opaque_affected_p99_drop=.035,
             opaque_alpha=.90, hole_alpha=.80, affected_threshold=.003,
             mean_outside_alpha_increase=.001, mean_outside_rgb_increase=.001,
             outside_affected_p99_increase=.02)
UNKNOWN_SPLIT = dict(min_principal_aspect=3., major_sigma_cap_m=.0045)


def _native_alpha(model, target, render, pipe, black):
    white = torch.ones_like(model._xyz)
    y, x = target.crop
    return render(target.cam, model, pipe, black, override_color=white)['render'][0, y, x].contiguous()


def _view_name(target):
    return str(target.row['frame'])


def _reference_and_votes(model, ids, is_skin, targets, render, pipe, black):
    votes = torch.zeros_like(ids)
    excess = model._xyz.new_zeros(len(ids))
    max_scale = torch.zeros_like(excess)
    baselines, records = [], []
    detached = copy(model)
    for name in PARAMETERS.values():
        setattr(detached, name, getattr(model, name).detach())
    major_sigma_local = model._scaling[ids].amax(dim=1).exp()
    for target in targets:
        model.select_mesh_by_timestep(target.slot)
        detached.select_mesh_by_timestep(target.slot)
        values = face_surface_metrics(model, ids)
        y, x = target.crop
        with torch.enable_grad():
            packet = render(target.cam, detached, pipe, black)
            native = packet['render'][:, y, x]
            observed = target.rgb[:, target.valid]*target.alpha[target.valid]
            photo = (native[:, target.valid]-observed).abs().mean()
            screen_gradient, = torch.autograd.grad(photo, packet['viewspace_points'])
        active = packet['visibility_filter'][ids] & screen_gradient[ids, :2].abs().any(1)
        votes += active
        relative = torch.where(is_skin, values['tangent_major_sigma']/values['tangent_cap'],
            major_sigma_local*values['support_scale']/UNKNOWN_SPLIT['major_sigma_cap_m'])-1
        excess += relative.clamp_min(0)*active
        max_scale = torch.maximum(max_scale, values['support_scale']*active)
        baselines.append((native.detach().contiguous(), _native_alpha(detached, target, render, pipe, black)))
        records.append(dict(frame=_view_name(target), slot=int(target.slot),
                            trusted_rows_with_raster_gradient=int(active.count_nonzero())))
    return votes, excess/votes.clamp_min(1), max_scale, baselines, records


def _principal_split_seeds(local, logs, quaternion, ratio):
    from roma import quat_wxyz_to_xyzw, unitquat_to_rotmat

    basis = unitquat_to_rotmat(quat_wxyz_to_xyzw(torch.nn.functional.normalize(quaternion, dim=-1)))
    axis = logs.argmax(dim=1, keepdim=True)
    major = logs.gather(1, axis).exp()
    direction = basis.gather(2, axis[:, None].expand(-1, 3, -1))[:, :, 0]
    displacement = direction*(ratio*major)
    centers = torch.stack((local-displacement, local+displacement), dim=1)
    widths = logs.clone()
    widths.scatter_add_(1, axis, logs.new_full(axis.shape, .5*log(1.-ratio**2)))
    return centers, widths[:, None].expand(-1, 2, -1), quaternion[:, None].expand(-1, 2, -1)


def _preview(model, keep, parents, centers, widths, rotations, opacity):
    preview = copy(model)
    inherited = parents.repeat_interleave(2)
    appended = dict(_xyz=centers.flatten(0, 1), _scaling=widths.flatten(0, 1),
                    _rotation=rotations.flatten(0, 1), _opacity=opacity,
                    _features_dc=model._features_dc[inherited], _features_rest=model._features_rest[inherited])
    for name, value in appended.items():
        setattr(preview, name, torch.cat((getattr(model, name)[keep], value)))
    preview.binding = torch.cat((model.binding[keep], model.binding[inherited]))
    return preview, appended, inherited


def _mean(values):
    return float(values.mean()) if values.numel() else 0.


def _p99(values):
    return float(torch.quantile(values.float(), .99)) if values.numel() else 0.


def _gate_view(target, old_rgb, old_alpha, rgb, alpha):
    if not (bool(torch.isfinite(rgb).all()) and bool(torch.isfinite(alpha).all())):
        return dict(accepted=False, reason='nonfinite_render')
    valid, outside = target.valid, target.outside
    expected = target.rgb[:, valid]*target.alpha[valid]
    rgb_increase = _mean((rgb[:, valid]-expected).abs()-(old_rgb[:, valid]-expected).abs())
    foreground_drop = _mean((old_alpha-alpha)[valid])
    changed = ((old_rgb-rgb).abs().amax(0) > SPLIT['affected_threshold']) | (
        (old_alpha-alpha).abs() > SPLIT['affected_threshold'])
    baseline_opaque = valid & (target.baseline_alpha >= SPLIT['opaque_alpha'])
    opaque_changed = baseline_opaque & changed
    opaque_incremental_drop = _p99((old_alpha-alpha)[opaque_changed].clamp_min(0))
    new_holes = baseline_opaque & (old_alpha >= SPLIT['opaque_alpha']) & (alpha < SPLIT['hole_alpha'])
    spill = (alpha-old_alpha).clamp_min(0)
    brightening = (rgb-old_rgb).clamp_min(0).amax(0)
    outside_alpha = _mean(spill[outside])
    outside_rgb = _mean(brightening[outside])
    outside_local = max(_p99(spill[outside & changed]), _p99(brightening[outside & changed]))
    holes = int(new_holes.count_nonzero())
    passed = (rgb_increase <= SPLIT['mean_rgb_increase']
              and foreground_drop <= SPLIT['mean_foreground_alpha_drop']
              and opaque_incremental_drop <= SPLIT['opaque_affected_p99_drop']
              and holes == 0 and outside_alpha <= SPLIT['mean_outside_alpha_increase']
              and outside_rgb <= SPLIT['mean_outside_rgb_increase']
              and outside_local <= SPLIT['outside_affected_p99_increase'])
    return dict(accepted=passed, mean_rgb_increase=rgb_increase, mean_foreground_alpha_drop=foreground_drop,
                opaque_affected_p99_drop=opaque_incremental_drop, newly_transparent_pixels=holes,
                affected_opaque_pixels=int(opaque_changed.count_nonzero()),
                outside_mean_alpha_increase=outside_alpha, outside_mean_rgb_increase=outside_rgb,
                outside_affected_p99_increase=outside_local)


def _commit(model, keep, parents, appended, inherited):
    if getattr(model, 'point_limits', None) is not None:
        raise ValueError('The face refinement split expects its separate per-row caps.')
    optimizer = model.optimizer
    attributes = (*PARAMETERS.values(), 'binding', 'binding_counter', 'xyz_gradient_accum', 'denom',
                  'max_radii2D', 'densification_ids', 'next_densification_id')
    saved = {name: getattr(model, name) for name in attributes}
    groups = [(group, list(group['params'])) for group in optimizer.param_groups]
    states = {parameter: state.copy() for parameter, state in optimizer.state.items()}
    added = len(inherited)
    owners = model.binding[inherited]
    to_prune = torch.cat((~keep, keep.new_zeros(added)))
    try:
        model.binding = torch.cat((model.binding, owners))
        model.binding_counter = model.binding_counter.clone()
        model.binding_counter.scatter_add_(0, owners, torch.ones_like(owners, dtype=model.binding_counter.dtype))
        model.densification_postfix(*(appended[name] for name in PARAMETERS.values()), constraint_parents=inherited)
        for name in ('xyz_gradient_accum', 'denom'):
            value = saved[name]
            setattr(model, name, torch.cat((value, value.new_zeros((added, *value.shape[1:])))))
        model.prune_points(to_prune)
        survivors = len(keep)-len(parents)
        if len(model.binding) != survivors+added or not torch.equal(model.binding[-added:], owners):
            raise RuntimeError('Skin split changed the verified owner mapping.')
        expected_counts = saved['binding_counter'].clone()
        parent_owners = saved['binding'][parents]
        expected_counts.scatter_add_(0, parent_owners, torch.ones_like(parent_owners, dtype=expected_counts.dtype))
        if not torch.equal(model.binding_counter, expected_counts):
            raise RuntimeError('Skin split changed binding counts outside the verified proposal.')
        for name in ('binding', 'max_radii2D', 'xyz_gradient_accum', 'denom', 'densification_ids', *PARAMETERS.values()):
            if not torch.equal(getattr(model, name)[:survivors], saved[name][keep]):
                raise RuntimeError(f'Skin split changed survivor data in {name}.')
        for name, value in appended.items():
            parameter = getattr(model, name)
            if not torch.equal(parameter[-added:], value):
                raise RuntimeError(f'Skin split changed the verified child in {name}.')
            old = states.get(saved[name], {})
            if old:
                new = optimizer.state[parameter]
                if not torch.equal(old['step'], new['step']) or any(
                        not torch.equal(new[key][:survivors], old[key][keep])
                        or bool(new[key][-added:].count_nonzero()) for key in MOMENTS):
                    raise RuntimeError(f'Skin split changed survivor Adam or child initialization in {name}.')
        if len(model.densification_ids.unique()) != len(model.binding):
            raise RuntimeError('Skin split produced duplicate stable IDs.')
    except BaseException:
        for name, value in saved.items():
            setattr(model, name, value)
        for group, parameters in groups:
            group['params'] = parameters
        optimizer.state.clear()
        optimizer.state.update(states)
        raise
    return dict(parent_ids=saved['densification_ids'][parents].tolist(),
                child_ids=model.densification_ids[-added:].reshape(-1, 2).tolist(),
                owner_faces=parent_owners.tolist(), unchanged_survivor_rows=survivors)


@torch.no_grad()
def try_skin_split(model, labels, spill, caps, targets, render, pipe, black, step, *, observed=None, skin=None):
    """Try trusted tangent or observed unknown principal splits against native cameras."""
    if len(model.binding) != len(labels) or labels.shape != spill.shape or caps.shape != model._scaling.shape:
        raise ValueError('Skin split requires aligned row labels, spill flags, and three-axis caps.')
    r27 = skin is not None or observed is not None
    trusted = (labels == 1) & ~spill
    observed_unknown = torch.zeros_like(trusted)
    for selection in (skin, observed):
        if selection is not None and (selection.shape != labels.shape or selection.dtype != torch.bool
                                      or selection.device != labels.device):
            raise ValueError('Skin and observed support masks must be aligned Boolean rows.')
    if skin is not None:
        trusted &= skin
    if observed is not None:
        logs = model._scaling.detach().sort(dim=1).values
        elongated = logs[:, 2]-logs[:, 1] >= log(UNKNOWN_SPLIT['min_principal_aspect'])
        observed_unknown = observed & (labels == 0) & elongated
    selected = trusted | observed_unknown
    training_support = getattr(model, 'face_surface_view_support', None)
    if training_support is not None:
        if training_support.shape != labels.shape or training_support.device != labels.device:
            raise ValueError('The previous native sweep support does not align with the live Gaussian rows.')
        selected &= training_support >= SPLIT['repeat_views']
    ids = selected.nonzero(as_tuple=True)[0]
    event = dict(step=step, policy=SPLIT.copy(), rows_before=len(labels), rows_after=len(labels),
                 trusted_nonspill_rows=int((selected & trusted).count_nonzero()), attempted_parents=0, committed_parents=0)
    if r27:
        event.update(unknown_policy=UNKNOWN_SPLIT.copy(), observed_elongated_unknown_rows=int(
            (selected & observed_unknown).count_nonzero()))
    if not len(ids):
        return labels, spill, caps, dict(event, reason='no_trusted_rows')
    previous_pose = model.timestep
    try:
        votes, excess, scale, baselines, views = _reference_and_votes(model, ids, trusted[ids], targets, render, pipe, black)
        event['views'] = views
        eligible = (votes >= SPLIT['repeat_views']) & (excess >= SPLIT['min_relative_excess'])
        candidates = eligible.nonzero(as_tuple=True)[0]
        event['eligible_rows'] = len(candidates)
        if not len(candidates):
            return labels, spill, caps, dict(event, reason='no_repeated_oversized_skin')
        rank = torch.argsort(excess[candidates]*votes[candidates].float().log1p(), descending=True, stable=True)
        picked = candidates[rank[:SPLIT['parent_budget']]]
        parents = ids[picked]
        event.update(attempted_parents=len(parents), proposed_parent_ids=model.densification_ids[parents].tolist(),
                     source_view_counts=votes[picked].tolist(), relative_tangent_excess=excess[picked].tolist())
        skin_parent = trusted[parents]
        if r27:
            event['parent_material_labels'] = labels[parents].tolist()
            event['relative_footprint_excess'] = event.pop('relative_tangent_excess')
        if training_support is not None:
            event['previous_sweep_support_counts'] = training_support[parents].tolist()
        ratio = SPLIT['tangent_shift']
        centers = model._xyz.new_empty((len(parents), 2, 3))
        widths = torch.empty_like(centers)
        rotations = model._rotation.new_empty((len(parents), 2, 4))
        if bool(skin_parent.any()):
            rows = parents[skin_parent]
            x, w, q = tangent_split_seeds(model._xyz[rows], model._scaling[rows], model._rotation[rows],
                scale[picked][skin_parent], center_clearance_m=float('inf') if r27 else FACE_SURFACE['center_clearance_m'],
                normal_sigma_cap_m=float('inf') if r27 else FACE_SURFACE['normal_sigma_clearance_m'],
                tangent_shift_ratio=ratio)
            upper = caps[rows, None].amax(dim=-1, keepdim=True) if r27 else caps[rows, None]
            centers[skin_parent], widths[skin_parent], rotations[skin_parent] = x, w.minimum(upper), q
        if bool((~skin_parent).any()):
            rows = parents[~skin_parent]
            centers[~skin_parent], widths[~skin_parent], rotations[~skin_parent] = _principal_split_seeds(
                model._xyz[rows], model._scaling[rows], model._rotation[rows], ratio)
        midpoint = exp(-.5*ratio**2/(1-ratio**2))
        alpha = model.get_opacity[parents]
        child_alpha = ((1-(1-alpha).sqrt())/midpoint).clamp(.00001, .99999)
        opacity = child_alpha.logit().repeat_interleave(2, dim=0)
        keep = torch.ones_like(labels, dtype=torch.bool)
        keep[parents] = False
        preview, appended, inherited = _preview(model, keep, parents, centers, widths, rotations, opacity)
        for target, (old_rgb, old_alpha), receipt in zip(targets, baselines, views, strict=True):
            preview.select_mesh_by_timestep(target.slot)
            y, x = target.crop
            rgb = render(target.cam, preview, pipe, black)['render'][:, y, x]
            alpha = _native_alpha(preview, target, render, pipe, black)
            receipt['gate'] = _gate_view(target, old_rgb, old_alpha, rgb, alpha)
        if not all(view['gate']['accepted'] for view in views):
            return labels, spill, caps, dict(event, reason='native_guard_declined')
        model.select_mesh_by_timestep(previous_pose)
        result = _commit(model, keep, parents, appended, inherited)
        return (torch.cat((labels[keep], labels[inherited])), torch.cat((spill[keep], spill[inherited])),
                torch.cat((caps[keep], caps[inherited])),
                dict(event, reason='native_guard_passed', rows_after=len(model.binding),
                     committed_parents=len(parents), **result))
    finally:
        model.select_mesh_by_timestep(previous_pose)
