"""Bounded native proposals driven by reliable facial Gaussian distribution distance."""

import math
from pathlib import Path

import torch

from .render_support import replacement_evaluator, source_weights
from .shape_refinement import _projection
from .size_child_commit import commit_size_children
from .size_child_optimization import _preserve_pose, optimize_size_children
from .splat_surface_distance import choose_world_split_rows, plane_distribution, tangent_split_seeds
from .surface_attachment import CONFIG as SKIN_CONFIG, _in_skin


SURFACE_DENSIFICATION = dict(
    steps=[3000, 4800, 6600, 8400],
    expected_distance_threshold_m=.004,
    parent_budget_per_step=3,
    local_cycles=4,
    min_source_bins=2,
    min_skin_parent_contribution_px=.25,
    center_clearance_m=.0025,
    normal_sigma_cap_m=.002,
    tangent_shift_ratio=.50,
    min_distance_drop_m=.0005,
    min_distance_drop_fraction=.15,
)


@torch.no_grad()
def _native_parent_support(model, attachment):
    """Collect current posed, raster-visible row votes from frozen material masks."""
    from gaussian_renderer import render

    constraints = attachment.constraints
    groups = sorted(set(constraints.bins))
    group_index = {group: index for index, group in enumerate(groups)}
    rows = len(model.binding)
    observed = torch.zeros((len(attachment.views), rows), dtype=torch.bool, device=model.binding.device)
    scales_by_bin = model._xyz.new_zeros((len(groups), rows))
    view_receipts = []
    with _preserve_pose(model):
        for i, supported in enumerate(attachment.has_supported_view):
            if supported:
                model.select_mesh_by_timestep(constraints.slots[i])
                visible = render(constraints.cameras[i], model, constraints.pipe, constraints.black)['visibility_filter']
                ids = attachment._selected(i, visible)
                observed[i, ids] = True
                group = group_index[constraints.bins[i]]
                scale = model.face_scaling[model.binding[ids], 0]
                scales_by_bin[group, ids] = torch.maximum(scales_by_bin[group, ids], scale)
            view_receipts.append(dict(frame=constraints.rows[i]['frame'], source_bin=constraints.bins[i],
                                      selected_rows=int(observed[i].count_nonzero())))
    votes = (scales_by_bin > 0).count_nonzero(dim=0)
    supported_rows = (votes >= SURFACE_DENSIFICATION['min_source_bins']).nonzero(as_tuple=True)[0]
    distance = model._xyz.new_full((rows,), -torch.inf)
    representative_scale = model._xyz.new_zeros((rows,))
    if len(supported_rows):
        chosen_scales = scales_by_bin[:, supported_rows]
        scale = chosen_scales.masked_fill(chosen_scales == 0, torch.nan).nanmedian(dim=0).values
        representative_scale[supported_rows] = scale
        _, _, distance[supported_rows] = plane_distribution(
            model._xyz[supported_rows], model._scaling[supported_rows], model._rotation[supported_rows], scale)
    return observed, votes, distance, representative_scale, view_receipts


@torch.no_grad()
def _child_skin_bins(model, attachment, owner, centers, supporting_views):
    """Check both proposed centers against the parent's native material observations."""
    constraints = attachment.constraints
    evidence = []
    with _preserve_pose(model):
        for i in supporting_views:
            view = attachment.views[i]
            model.select_mesh_by_timestep(constraints.slots[i])
            xyz = (model.face_orien_mat[owner] @ centers[..., None]).squeeze(-1)
            xyz = xyz*model.face_scaling[owner]+model.face_center[owner]
            xy, depth = _projection(xyz, *constraints.matrices[i])
            valid = _in_skin(xy, depth, (view.box, view.mask), constraints.views[i].mesh_depth,
                             SKIN_CONFIG['center_visibility_depth_slack_m'])
            if bool(valid.all()):
                evidence.append(dict(frame=constraints.rows[i]['frame'], source_bin=constraints.bins[i]))
    return evidence


def _bin_count(evidence):
    return len({item['source_bin'] for item in evidence})


@torch.no_grad()
def _try_surface_split(model, attachment, world_scale, parents, supported_views, weights, *,
                       distance, representative_scale, diagnostic_dir):
    policy = SURFACE_DENSIFICATION
    constraints = attachment.constraints
    owners = model.binding[parents]
    owner = int(owners[0])
    stable_id = int(model.densification_ids[parents[0]])
    source_indices = [i for i in supported_views
                      if float(weights[i, 0, 0]) >= policy['min_skin_parent_contribution_px']]
    positive_bins = {constraints.bins[i] for i in source_indices}
    result = dict(parent_id=stable_id, owner_face=owner, outcome='rolled_back', child_ids=[],
                  parent_expected_distance_m=distance,
                  skin_raster_source_bins=len({constraints.bins[i] for i in supported_views}),
                  contributing_skin_source_bins=len(positive_bins), local_optimizer_updates=0)
    if len(positive_bins) < policy['min_source_bins']:
        return dict(result, reason='insufficient_visible_parent_contribution')

    scale_max = world_scale.maximum_scale[owners]
    centers, log_sigma, rotations = tangent_split_seeds(
        model._xyz[parents], model._scaling[parents], model._rotation[parents], scale_max,
        center_clearance_m=policy['center_clearance_m'],
        normal_sigma_cap_m=policy['normal_sigma_cap_m'], tangent_shift_ratio=policy['tangent_shift_ratio'])
    seed_evidence = _child_skin_bins(model, attachment, owner, centers[0], source_indices)
    result['seed_both_centers_in_skin'] = seed_evidence
    if _bin_count(seed_evidence) < policy['min_source_bins']:
        return dict(result, reason='insufficient_seed_skin_support')

    world_cap = world_scale.log_cap[model.binding, None].expand(-1, 3)
    upper = torch.minimum(world_scale.limits.cap[parents], world_cap[parents])
    upper[:, 1] = torch.minimum(upper[:, 1], (policy['normal_sigma_cap_m']/scale_max).log())
    # Seed two equal lobes from the parent alpha and compensate their tangent
    # displacement at the old midpoint. Local opacity remains fixed afterwards.
    shift = policy['tangent_shift_ratio']
    midpoint_attenuation = math.exp(-.5*shift**2/(1-shift**2))
    opacity = ((1-(1-model.get_opacity[parents]).sqrt())/midpoint_attenuation).clamp(.005, .995)
    logits = opacity[:, None].expand(-1, 2, -1).logit()
    dc = model._features_dc[parents, None].expand(-1, 2, -1, -1).clone()
    options = world_scale.options['render_support']
    fit = optimize_size_children(model, constraints, parents, centers, torch.minimum(log_sigma, upper[:, None]),
                                 logits, dc, upper, weights, max_cycles=policy['local_cycles'],
                                 guard_options=options, diagnostic_dir=diagnostic_dir,
                                 child_rotation=rotations, fit_opacity=False, require_smaller_radius=False,
                                 max_normal_center=float(policy['center_clearance_m']/scale_max[0]))
    result['local_optimizer_updates'] = fit.audit['optimizer_updates']
    result['local_fit'] = fit.audit
    if fit.audit['status'] != 'fitted' or not fit.audit['checkpoint_feasible']:
        return dict(result, reason='no_feasible_native_optimization_checkpoint')

    evidence = _child_skin_bins(model, attachment, owner, fit.centers[0], source_indices)
    result['optimized_both_centers_in_skin'] = evidence
    if _bin_count(evidence) < policy['min_source_bins']:
        return dict(result, reason='insufficient_optimized_skin_support')
    metric_scale = model._xyz.new_full((2,), representative_scale)
    offset, sigma, child_distance = plane_distribution(fit.centers[0], fit.log_sigma[0],
                                                       fit.child_rotation[0], metric_scale)
    max_child_distance = float(child_distance.max())
    required_drop = max(policy['min_distance_drop_m'], policy['min_distance_drop_fraction']*distance)
    result['geometry'] = dict(child_expected_distance_m=child_distance.tolist(),
                              child_normal_center_m=offset.tolist(), child_normal_sigma_m=sigma.tolist(),
                              expected_distance_drop_m=distance-max_child_distance, required_drop_m=required_drop,
                              child_rotation_local_wxyz=fit.child_rotation[0].tolist())
    if not math.isfinite(max_child_distance) or distance-max_child_distance < required_drop:
        return dict(result, reason='insufficient_distribution_distance_reduction')

    with replacement_evaluator(model, constraints, parents, fit.centers, fit.log_sigma,
                               fit.opacity_logits.sigmoid(), weights, options, child_dc=fit.dc,
                               child_rotation=fit.child_rotation) as evaluate:
        gate = evaluate([0])
    result['native_gate'] = gate
    if not gate['accepted']:
        return dict(result, reason='native_replay_regression')
    commit = commit_size_children(model, world_scale.limits, parents, fit, parents.new_zeros(1), world_cap)
    return dict(result, outcome='committed', reason='distance_reduced_native_replay_passed',
                child_ids=commit['child_ids'], transaction=commit)


@torch.no_grad()
def refine_surface_distributions(model, attachment, world_scale, step, *, attempted_ids, diagnostic_dir=None):
    """Rank, fit, and commit only verified native tangent splits."""
    policy = SURFACE_DENSIFICATION
    if step not in policy['steps'] or attachment is None or world_scale.model is not model:
        raise ValueError('Surface densification requires a registered step and the active bound scene.')
    observed, votes, distance, scale, views = _native_parent_support(model, attachment)
    eligible = votes >= policy['min_source_bins']
    above = eligible & (distance > policy['expected_distance_threshold_m'])
    available = distance.clone()
    if attempted_ids:
        previous = model.densification_ids.new_tensor(sorted(attempted_ids))
        available[torch.isin(model.densification_ids, previous)] = -torch.inf
    parents = choose_world_split_rows(available, votes, model.densification_ids,
                                     threshold=policy['expected_distance_threshold_m'],
                                     budget=policy['parent_budget_per_step'])
    stable_ids = model.densification_ids[parents].tolist()
    supports = observed[:, parents].T
    rows_before = len(model.binding)
    trials = []
    committed = False
    initial_weights = source_weights(model, attachment.constraints, parents) if len(parents) else None
    for index, stable_id in enumerate(stable_ids):
        parent = torch.searchsorted(model.densification_ids, model.densification_ids.new_tensor([stable_id]))
        if int(model.densification_ids[parent][0]) != stable_id:
            raise RuntimeError('Surface proposal lost its registered stable parent identity.')
        weights = (source_weights(model, attachment.constraints, parent) if committed else
                   initial_weights[:, index:index+1])
        directory = None if diagnostic_dir is None else Path(diagnostic_dir)/f'parent-{stable_id:07d}'
        selected_row = parents[index]
        trial = _try_surface_split(model, attachment, world_scale, parent,
                                   supports[index].nonzero(as_tuple=True)[0].tolist(), weights,
                                   distance=float(distance[selected_row]), representative_scale=float(scale[selected_row]),
                                   diagnostic_dir=directory)
        committed |= trial['outcome'] == 'committed'
        attempted_ids.add(stable_id)
        trials.append(trial)
    commits = sum(trial['outcome'] == 'committed' for trial in trials)
    if len(model.binding) != rows_before+commits:
        raise RuntimeError('Surface densification changed topology beyond the accepted two-child splits.')
    return dict(schema='shuolee-surface-densification/1', step=step, policy=policy.copy(),
                skin_rows_with_repeated_source_support=int(eligible.count_nonzero()),
                candidates_above_threshold=int(above.count_nonzero()),
                previously_attempted_candidates=int((above & ~torch.isfinite(available)).count_nonzero()),
                selected_parent_ids=stable_ids, attempted_parents=len(trials), committed_parents=commits,
                rolled_back_parents=len(trials)-commits, rows_before=rows_before, rows_after=len(model.binding),
                local_optimizer_updates=sum(trial['local_optimizer_updates'] for trial in trials),
                views=views, trials=trials,
                policy_description='Median supported bin scale times fixed-plane folded-normal distance. '
                                   'Verified native straight RGB and alpha regression, no local opacity optimization.')
