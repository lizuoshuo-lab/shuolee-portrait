"""Bound SH0 covariance limits and supported principal-axis replacements."""

import torch

from .shape_refinement import _projection, _screen_visibility


def scale_config():
    return dict(global_extent_ratio=.025, minimum_extent_ratio=.016, triangle_scale_multiple=3.,
                active_opacity=.10, audit_thresholds_m=[.05, .10], near_limit_ratio=.90,
                axis_ratio_min=2., max_parents=128, child_sigma_ratio=.82, child_opacity_max=.995,
                unsupported_cap_ratio=.80, small_projected_sigma_px=5., split_steps=[3000, 4800, 6600, 8400])


def _limits(face_scale, extent, options):
    cap = (options['triangle_scale_multiple']*face_scale).clamp(
        extent*options['minimum_extent_ratio'], extent*options['global_extent_ratio'])
    return cap, (cap/face_scale).log()


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

    def repair(self, step, constraints):
        """Split only near-cap long axes whose two centers have multi-view foreground support."""
        from utils.general_utils import build_rotation

        model, options = self.model, self.options
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
            # Strong exterior evidence warrants tightening only the offending axis.
            trim = (~supported) & (fg[:, 0] == 0) & (bg[:, 0] >= constraints.options['prune_bg_bins'])
            trim_ids, trim_axis = ids[trim], axis[trim]
            trimmed_limit = self.log_cap[model.binding[trim_ids]]+high.new_tensor(options['unsupported_cap_ratio']).log()
            model._scaling[trim_ids, trim_axis] = torch.minimum(model._scaling[trim_ids, trim_axis], trimmed_limit)
            state = model.optimizer.state.get(model._scaling)
            if state is not None:
                state['exp_avg'][trim_ids, trim_axis] = 0
            parents = ids[supported]
            count = len(parents)
            if count:
                children = probes[supported, 1:].transpose(0, 1).reshape(-1, 3)
                log_sigma = model._scaling[parents].clone()
                log_sigma.scatter_(1, axis[supported, None], child_sigma[supported, None].log())
                binding = model.binding[parents].repeat(2)
                prune = torch.zeros(len(model.binding)+len(binding), device=local_sigma.device, dtype=torch.bool)
                prune[parents] = True
                model.binding = torch.cat((model.binding, binding))
                model.binding_counter.scatter_add_(0, binding, torch.ones_like(binding, dtype=torch.int32))
                model.densification_postfix(children, model._features_dc[parents].repeat(2, 1, 1),
                    model._features_rest[parents].repeat(2, 1, 1), opacity[supported, None].repeat(2, 1).logit(),
                    log_sigma.repeat(2, 1), model._rotation[parents].repeat(2, 1))
                model.prune_points(prune)
                if len(model.binding) != len(prune)-count or int(model.binding_counter.min()) < 1:
                    raise RuntimeError('Principal-axis replacement lost a bound parent or child.')
            event.update(observation, split_parents=count, children=2*count, exterior_trimmed=len(trim_ids),
                         retained_without_child_support=len(ids)-count-len(trim_ids))
        self.splits.append(event)

    def report(self, initial, terminal=None):
        return dict(state='completed' if terminal else 'running', config=self.options,
                    units='model-world meters, principal-axis one-sigma; not radius or diameter',
                    reference_pose_index=self.reference, pose_count=self.model.num_timesteps, extent_m=self.extent,
                    limits_m=dict(min=float(self.cap.min()), max=float(self.cap.max())),
                    initial=initial, checkpoints=self.checkpoints, splits=self.splits, terminal=terminal)

