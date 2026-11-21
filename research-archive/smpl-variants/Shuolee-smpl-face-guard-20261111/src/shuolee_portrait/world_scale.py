"""Bound SH0 covariance limits and supported principal-axis replacements."""

import torch

from .render_support import FootprintLimits, classify, replacement_evaluator, source_weights
from .replacement_selection import select_replacements
from .shape_refinement import _projection, _screen_visibility, _tangent_distance


def scale_config():
    return dict(global_extent_ratio=.025, minimum_extent_ratio=.016, triangle_scale_multiple=3.,
                active_opacity=.10, audit_thresholds_m=[.05, .10], near_limit_ratio=.90,
                axis_ratio_min=2., max_parents=128, child_sigma_ratio=.82, child_opacity_max=.995,
                unsupported_cap_ratio=.80, small_projected_sigma_px=5., split_steps=[3000, 4800, 6600, 8400],
                needle=dict(width_over_extent=.008, width_over_face=.65, gap_over_extent=.010, gap_over_face=.75,
                            max_parents=128, center_move_fraction=.50, max_move_over_extent=.010, width_trim_ratio=.75),
                render_support=dict(min_equivalent_pixels=1., min_background_bins=2, background_to_foreground=.20,
                                    coverage_slack_px=2., coverage_loss_fraction=.025, replacement_budget=16,
                                    coverage_max_trials=32, coverage_tile_px=16,
                                    coverage_local_deficit_px=1., coverage_region_deficit_px=1.,
                                    coverage_rgb_mae_increase=1/255,
                                    coverage_local_rgb_mae_increase=1/255))


def _limits(face_scale, extent, options):
    cap = (options['triangle_scale_multiple']*face_scale).clamp(
        extent*options['minimum_extent_ratio'], extent*options['global_extent_ratio'])
    return cap, (cap/face_scale).log()


def _off_surface(local, triangle, scale, extent, options):
    normal = local[:, 1].abs()
    tangent = _tangent_distance(local[:, [0, 2]], triangle)
    gap = torch.hypot(normal, tangent)*scale
    threshold = (options['gap_over_face']*scale).clamp_min(options['gap_over_extent']*extent)
    return gap, gap > threshold


def _source_conflict(fg, bg, min_background):
    conflicted = (bg >= min_background) & (bg > fg)
    unobserved = (fg == 0) & (bg == 0)
    return conflicted, unobserved


def _children(parent_sigma, alpha, options):
    """Match the parent midpoint's alpha and its principal-axis second moment when feasible."""
    child_sigma = parent_sigma*options['child_sigma_ratio']
    offset = (parent_sigma.square()-child_sigma.square()).sqrt()
    midpoint_alpha = alpha/(1+(1-alpha).sqrt())
    max_distance = child_sigma*(-2*(midpoint_alpha/options['child_opacity_max']).clamp(1e-8, 1).log()).sqrt()
    offset = torch.minimum(offset, max_distance)
    opacity = (midpoint_alpha*((offset/child_sigma).square()*.5).exp()).clamp_max(options['child_opacity_max'])
    return child_sigma, offset, opacity


def _summary(values, active, limits, options):
    def measure(sigma, limit):
        counts = (sigma > limit+1e-6).count_nonzero()
        if not len(sigma):
            return dict(points=0, p50_m=None, p95_m=None, p99_m=None, max_m=None, over_005m=0,
                        over_010m=0, over_limit=0)
        p50, p95, p99 = torch.quantile(sigma, sigma.new_tensor([.50, .95, .99])).tolist()
        a, b = options['audit_thresholds_m']
        return dict(points=len(sigma), p50_m=p50, p95_m=p95, p99_m=p99, max_m=float(sigma.max()),
                    over_005m=int((sigma > a).count_nonzero()), over_010m=int((sigma > b).count_nonzero()),
                    over_limit=int(counts))
    return dict(all=measure(values, limits), active=measure(values[active], limits[active]))


def _projected_widths(xy, depth, K, w2c, world_axes, major):
    """One-sigma image widths before the renderer's screen low-pass filter."""
    jacobian = (K[:2]-xy[..., None]*K[2])/depth[:, None, None].clamp_min(.01)
    projected = jacobian @ w2c[:3, :3] @ world_axes
    major_axis = projected.gather(2, major[:, None, None].expand(-1, 2, 1)).squeeze(2).norm(dim=1)
    u, v = projected.unbind(1)
    a, b, c = u.square().sum(1), (u*v).sum(1), v.square().sum(1)
    ellipse = ((a+c+((a-c).square()+4*b.square()).sqrt())*.5).clamp_min(0).sqrt()
    return major_axis, ellipse


class WorldScale:
    def __init__(self, model, extent, reference, options):
        self.model, self.extent, self.reference, self.options = model, extent, reference, options
        self.projections = model._xyz.new_zeros(3, dtype=torch.int64)
        self.checkpoints, self.splits = [], []
        self.support_history, self.support_prunes = [], []
        self.limits = FootprintLimits(model._scaling)
        model.point_limits = self.limits
        self.refresh()

    def refresh(self):
        from utils.graphics_utils import compute_face_orientation

        verts = self.model.posed_vertices(slice(None))
        _, scales = compute_face_orientation(verts, self.model.faces, return_scale=True)
        scales = scales.squeeze(-1).clamp_min(1e-8)
        self.reference_scale = scales[self.reference]
        self.maximum_scale = scales.amax(0)
        self.cap, self.log_cap = _limits(self.maximum_scale, self.extent, self.options)

    def project(self):
        """Enforce the all-pose metric cap on only the offending local axes."""
        param = self.model._scaling
        limit = self.log_cap[self.model.binding, None]
        hit = param > limit
        self.projections += torch.stack((hit.count_nonzero(), hit.any(1).count_nonzero(), hit.any().long()))
        point_limits = getattr(self.model, 'point_limits', None)
        if point_limits is not None and point_limits.next_lineage:
            point_limits.apply(param, self.model.optimizer)
        param.copy_(torch.minimum(param, limit))
        state = self.model.optimizer.state.get(param)
        if state is not None:
            outward = state['exp_avg']
            outward.masked_fill_(hit & (outward < 0), 0)

    def checkpoint(self, step, stage):
        model = self.model
        local = model._scaling.amax(1).exp()
        if not torch.isfinite(local).all() or not torch.isfinite(self.maximum_scale).all():
            raise RuntimeError('Nonfinite bound Gaussian covariance or triangle scale.')
        owner = model.binding
        active = model.get_opacity[:, 0] >= self.options['active_opacity']
        axis_hits, point_hits, calls = self.projections.tolist()
        self.projections.zero_()
        row = dict(step=step, stage=stage,
                   reference=_summary(local*self.reference_scale[owner], active, self.cap[owner], self.options),
                   all_poses_max=_summary(local*self.maximum_scale[owner], active, self.cap[owner], self.options),
                   hard_projection=dict(axis_hits=axis_hits, point_hits=point_hits, calls_with_hits=calls))
        if stage in ('initial', 'terminal'):
            active_log_axes = model._scaling[active].topk(2, dim=1).values
            ratio = (active_log_axes[:, 0]-active_log_axes[:, 1]).exp()
            row['needle'] = dict(active_ratio_max=float(ratio.max()) if len(ratio) else None,
                                 active_ratio_over_20=int((ratio > 20).count_nonzero()),
                                 active_ratio_over_40=int((ratio > 40).count_nonzero()))
        if stage == 'terminal':
            row['render_support'] = self.limits.summary()
        if stage != 'initial' and row['all_poses_max']['all']['over_limit']:
            raise RuntimeError('World covariance limit did not hold in all registered poses.')
        if stage not in ('initial', 'terminal'):
            self.checkpoints.append(row)
        return row

    def _votes(self, local, owner, rotation, local_sigma, axis, constraints):
        model, options = self.model, self.options
        previous = model.timestep
        foreground = torch.zeros(local.shape[:2], device=local.device, dtype=torch.int16)
        background = torch.zeros_like(foreground)
        small_axis = small_ellipse = visible_count = 0
        try:
            for group in sorted(set(constraints.bins)):
                fg, bg = torch.zeros_like(foreground, dtype=torch.bool), torch.zeros_like(foreground, dtype=torch.bool)
                for i, view in enumerate(constraints.views):
                    if constraints.bins[i] != group:
                        continue
                    model.select_mesh_by_timestep(constraints.slots[i])
                    orient = model.face_orien_mat[owner]
                    scale = model.face_scaling[owner]
                    world = (local @ orient.transpose(1, 2))*scale[:, None]+model.face_center[owner, None]
                    K, w2c = constraints.matrices[i]
                    xy, depth = _projection(world.flatten(0, 1), K, w2c)
                    xy, depth = xy.reshape(-1, 3, 2), depth.reshape(-1, 3)
                    visible, x, y = _screen_visibility(xy, depth, view, constraints.options)
                    fg |= visible & view.inside[y, x]
                    bg |= visible & view.outside[y, x]
                    axes = orient @ (rotation*(local_sigma*scale)[:, None])
                    directional, ellipse = _projected_widths(xy[:, 0], depth[:, 0], K, w2c, axes, axis)
                    visible_count += visible[:, 0].count_nonzero()
                    small_axis += (visible[:, 0] & (directional < options['small_projected_sigma_px'])).count_nonzero()
                    small_ellipse += (visible[:, 0] & (ellipse < options['small_projected_sigma_px'])).count_nonzero()
                foreground += fg
                background += bg
        finally:
            model.select_mesh_by_timestep(previous)
        return foreground, background, dict(visible_parent_view_pairs=int(visible_count),
             small_projected_world_axis_pairs=int(small_axis), small_projected_ellipse_pairs=int(small_ellipse))

    def _replace(self, parents, centers, log_sigma, opacity):
        model = self.model
        binding = model.binding[parents].repeat(2)
        prune = torch.zeros(len(model.binding)+len(binding), device=binding.device, dtype=torch.bool)
        prune[parents] = True
        model.binding = torch.cat((model.binding, binding))
        model.binding_counter.scatter_add_(0, binding, torch.ones_like(binding, dtype=torch.int32))
        model.densification_postfix(centers.transpose(0, 1).reshape(-1, 3),
            model._features_dc[parents].repeat(2, 1, 1), model._features_rest[parents].repeat(2, 1, 1),
            opacity[:, None].repeat(2, 1).logit(), log_sigma.repeat(2, 1), model._rotation[parents].repeat(2, 1),
            constraint_parents=parents.repeat(2))
        model.prune_points(prune)
        if len(model.binding) != len(prune)-len(parents) or int(model.binding_counter.min()) < 1:
            raise RuntimeError('Principal-axis replacement lost a bound parent or child.')

    def confirm_prune(self, mask, constraints, step):
        ids = mask.nonzero(as_tuple=True)[0]
        allowed = torch.zeros_like(mask)
        if len(ids):
            weights = source_weights(self.model, constraints, ids)
            fg, _, conflict = classify(weights, constraints.bins, self.options['render_support'])
            allowed[ids] = conflict & ~fg
        self.support_prunes.append(dict(step=step, proposed=len(ids), approved=int(allowed.count_nonzero())))
        return allowed

    def remeasure(self, constraints, step, phase):
        ids = (self.limits.lineage >= 0).nonzero(as_tuple=True)[0]
        tracks = {}
        if len(ids):
            model = self.model
            weights = source_weights(model, constraints, ids).sum(0).tolist()
            sigma = (model._scaling[ids].amax(1).exp()*self.maximum_scale[model.binding[ids]]).tolist()
            overshoot = self.limits.overshoot[ids].exp().tolist()
            for lineage, mass, width, growth in zip(self.limits.lineage[ids].tolist(), weights, sigma, overshoot, strict=True):
                record = tracks.setdefault(lineage, dict(points=0, foreground=0., background=0., total=0.,
                                                         max_sigma_m=0., max_preprojection_growth_ratio=1.))
                record['points'] += 1
                for key, value in zip(('foreground', 'background', 'total'), mass, strict=True):
                    record[key] += value
                record['max_sigma_m'] = max(record['max_sigma_m'], width)
                record['max_preprojection_growth_ratio'] = max(record['max_preprojection_growth_ratio'], growth)
        self.support_history.append(dict(step=step, phase=phase, tracks=tracks))

    def _verified_subset(self, constraints, parents, centers, scales, opacity, weights):
        options = self.options['render_support']
        if not len(parents):
            return parents, None
        with replacement_evaluator(self.model, constraints, parents, centers, scales, opacity, weights, options) as evaluate:
            chosen, trial = select_replacements(evaluate, parents.tolist(), max_trials=options['coverage_max_trials'])
        trial['initial_proposed'] = trial['proposed']
        if chosen:
            trial['proposed'] = len(chosen)
        return parents.new_tensor(chosen), trial

    def _repair_displaced(self, constraints):
        """Measure raster support before adjusting broad off-surface rows."""
        from utils.general_utils import build_rotation

        model, options = self.model, self.options
        cfg = options['needle']
        owner = model.binding
        scale = self.reference_scale[owner]
        high = model._scaling.amax(1).exp()*scale
        min_width = (cfg['width_over_face']*scale).clamp_min(cfg['width_over_extent']*self.extent)
        opacity = model.get_opacity[:, 0]
        ids = ((high >= min_width) & (opacity >= options['active_opacity'])).nonzero(as_tuple=True)[0]
        ref_view = constraints.views[constraints.slots.index(self.reference)]
        gap, displaced = _off_surface(model._xyz[ids], ref_view.triangle[owner[ids]], scale[ids], self.extent, cfg)
        priority = high[ids]*gap*opacity[ids]
        ids, priority = ids[displaced], priority[displaced]
        count = len(ids)
        if count > cfg['max_parents']:
            ids = ids[priority.argsort(descending=True, stable=True)[:cfg['max_parents']]]
        event = dict(displaced_candidates=count, assessed=len(ids), conflicted=0, unobserved=0,
                     support_replaced=0, children=0, width_trimmed=0, preserved=len(ids))
        audit = dict(foreground_supported=0, center_conflicted=0, center_unobserved_with_foreground=0, ambiguous=0,
                     interventions=[], foreground_equivalent_pixels=0., background_equivalent_pixels=0.,
                     total_equivalent_pixels=0., coverage_trial=None)
        if not len(ids):
            return event, audit

        log_sigma = model._scaling[ids]
        wide_logs, wide_axes = log_sigma.topk(2, dim=1)
        axis = wide_axes[:, 0]
        rotation = build_rotation(model._rotation[ids])
        direction = rotation.gather(2, axis[:, None, None].expand(-1, 3, 1)).squeeze(2)
        child_sigma, shift, child_opacity = _children(wide_logs[:, 0].exp(), opacity[ids], options)
        original = model._xyz[ids]
        travel = original.norm(dim=1)*scale[ids]
        move = (cfg['max_move_over_extent']*self.extent/travel.clamp_min(1e-8)).clamp_max(cfg['center_move_fraction'])
        anchor = original*(1-move[:, None])
        displacement = direction*shift[:, None]
        proposals = torch.stack((original, anchor-displacement, anchor+displacement), 1)
        fg, bg, _ = self._votes(proposals, owner[ids], rotation, log_sigma.exp(), axis, constraints)
        support = options['render_support']
        weights = source_weights(model, constraints, ids)
        has_fg, unobserved, conflicted = classify(weights, constraints.bins, support)
        geometry = (fg[:, 1:] >= constraints.options['required_source_bins']).all(1) & (bg[:, 1:] == 0).all(1)
        shortlist = (has_fg & conflicted & geometry).nonzero(as_tuple=True)[0][:support['replacement_budget']]
        children = log_sigma[shortlist].clone()
        children.scatter_(1, axis[shortlist, None], child_sigma[shortlist, None].log())
        children = torch.minimum(children, self.limits.cap[ids[shortlist]])
        chosen, trial = self._verified_subset(constraints, ids[shortlist], proposals[shortlist, 1:],
                                              children, child_opacity[shortlist], weights[:, shortlist])
        replace, children = shortlist[chosen], children[chosen]
        trim = (~has_fg) & (conflicted | unobserved)

        # Retain the center, opacity and smallest width; cap all axes against regrowth.
        trim_ids, trim_axes = ids[trim], wide_axes[trim]
        limit = log_sigma[trim].clone()
        trimmed = wide_logs[trim]+wide_logs.new_tensor(cfg['width_trim_ratio']).log()
        widest_trimmed = trimmed.amax(1, keepdim=True)
        limit = torch.maximum(limit, widest_trimmed)
        limit.scatter_(1, trim_axes, trimmed)
        parents = ids[replace]
        lineage = torch.cat((self.limits.set(trim_ids, limit), self.limits.set(parents, children)))
        rows = torch.cat((trim.nonzero(as_tuple=True)[0], replace))
        original_sigma = log_sigma[rows].amax(1).exp()*self.maximum_scale[owner[ids[rows]]]
        mass = weights[:, rows].sum(0)
        audit['interventions'] = [
            dict(lineage=key, action='trim' if i < len(trim_ids) else 'replace',
                 foreground=values[0], background=values[1], total=values[2], max_sigma_m=width)
            for i, (key, values, width) in enumerate(zip(lineage.tolist(), mass.tolist(), original_sigma.tolist(), strict=True))]
        if len(parents):
            self._replace(parents, proposals[replace, 1:], children, child_opacity[replace])
        self.limits.apply(model._scaling, model.optimizer, measure=False)
        event.update(conflicted=int(conflicted.count_nonzero()), unobserved=int(unobserved.count_nonzero()),
                     support_replaced=len(parents), children=2*len(parents), width_trimmed=len(trim_ids),
                     preserved=len(ids)-len(parents)-len(trim_ids))
        totals = weights.sum((0, 1)).tolist()
        center_conflict, center_unobserved = _source_conflict(fg[:, 0], bg[:, 0], constraints.options['prune_bg_bins'])
        audit.update(foreground_supported=int(has_fg.count_nonzero()),
                     center_conflicted=int(center_conflict.count_nonzero()),
                     center_unobserved_with_foreground=int((has_fg & center_unobserved).count_nonzero()),
                     ambiguous=int((~has_fg & ~conflicted & ~unobserved).count_nonzero()),
                     foreground_equivalent_pixels=totals[0], background_equivalent_pixels=totals[1],
                     total_equivalent_pixels=totals[2], coverage_trial=trial)
        return event, audit

    def repair(self, step, constraints):
        """Split only near-cap long axes whose two centers have multi-view foreground support."""
        from utils.general_utils import build_rotation

        model, options = self.model, self.options
        support = options['render_support']
        self.remeasure(constraints, step, 'before_repair')
        high, axes = model._scaling.max(1)
        relative = high-self.log_cap[model.binding]
        eligible = (relative >= high.new_tensor(options['near_limit_ratio']).log()) & (
            (high-model._scaling.amin(1)).exp() >= options['axis_ratio_min']) & (
            model.get_opacity[:, 0] >= options['active_opacity'])
        ids = eligible.nonzero(as_tuple=True)[0]
        if len(ids) > options['max_parents']:
            ranked = relative[ids].argsort(descending=True, stable=True)
            ids = ids[ranked[:options['max_parents']]]
        event = dict(step=step, near_limit_candidates=int(eligible.count_nonzero()), assessed=len(ids),
                     split_parents=0, children=0, exterior_trimmed=0, retained_without_child_support=0,
                     visible_parent_view_pairs=0, small_projected_world_axis_pairs=0, small_projected_ellipse_pairs=0)
        near_trial, near_interventions = None, []
        if len(ids):
            axis = axes[ids]
            local_sigma = model._scaling[ids].exp()
            rotations = build_rotation(model._rotation[ids])
            direction = rotations.gather(2, axis[:, None, None].expand(-1, 3, 1)).squeeze(2)
            child_sigma, shift, opacity = _children(high[ids].exp(), model.get_opacity[ids, 0], options)
            center = model._xyz[ids]
            displacement = direction*shift[:, None]
            probes = torch.stack((center, center-displacement, center+displacement), 1)
            fg, bg, observation = self._votes(probes, model.binding[ids], rotations, local_sigma, axis, constraints)
            supported = (fg[:, 1:] >= constraints.options['required_source_bins']).all(1) & (bg[:, 1:] == 0).all(1)
            shortlist = supported.nonzero(as_tuple=True)[0][:support['replacement_budget']]
            if len(shortlist):
                weights = source_weights(model, constraints, ids[shortlist])
                has_fg, _, _ = classify(weights, constraints.bins, support)
                shortlist, weights = shortlist[has_fg], weights[:, has_fg]
                parents = ids[shortlist]
                log_sigma = model._scaling[parents].clone()
                log_sigma.scatter_(1, axis[shortlist, None], child_sigma[shortlist, None].log())
                log_sigma = torch.minimum(log_sigma, self.limits.cap[parents])
                chosen, near_trial = self._verified_subset(constraints, parents, probes[shortlist, 1:],
                                                           log_sigma, opacity[shortlist], weights)
                parents, children = parents[chosen], log_sigma[chosen]
                if len(parents):
                    lineage = self.limits.set(parents, children)
                    mass = weights[:, chosen].sum(0).tolist()
                    sigma = (model._scaling[parents].amax(1).exp()*self.maximum_scale[model.binding[parents]]).tolist()
                    near_interventions = [
                        dict(lineage=key, action='near_limit_replace', foreground=values[0], background=values[1],
                             total=values[2], max_sigma_m=width)
                        for key, values, width in zip(lineage.tolist(), mass, sigma, strict=True)]
                    self._replace(parents, probes[shortlist[chosen], 1:], children, opacity[shortlist[chosen]])
                    event.update(split_parents=len(parents), children=2*len(parents))
            event.update(observation, retained_without_child_support=len(ids)-event['split_parents'])
        event['needle'], event['render_support'] = self._repair_displaced(constraints)
        event['render_support']['near_limit_trial'] = near_trial
        event['render_support']['near_limit_interventions'] = near_interventions
        self.project()
        self.remeasure(constraints, step, 'after_repair')
        self.splits.append(event)

    def report(self, initial, terminal=None):
        return dict(state='completed' if terminal else 'running', config=self.options,
                    units='model-world meters, principal-axis one-sigma; not radius or diameter',
                    reference_pose_index=self.reference, pose_count=self.model.num_timesteps, extent_m=self.extent,
                    limits_m=dict(min=float(self.cap.min()), max=float(self.cap.max())),
                    initial=initial, checkpoints=self.checkpoints, splits=self.splits, terminal=terminal,
                    support_history=self.support_history, support_prunes=self.support_prunes)

