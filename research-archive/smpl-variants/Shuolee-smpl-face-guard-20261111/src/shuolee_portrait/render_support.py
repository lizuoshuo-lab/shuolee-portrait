"""Measure sorted Gaussian contributions and verify native replacement coverage."""

from contextlib import contextmanager
from types import SimpleNamespace

import torch
from torch.nn import functional as F


def color_weights(render_color, inside, outside, count):
    """Return equivalent opaque pixels per point for foreground, background and total."""
    if inside.ndim != 2 or inside.shape != outside.shape:
        raise ValueError('Contribution masks must share the native raster shape.')
    with torch.enable_grad():
        color = torch.zeros((count, 3), device=inside.device, dtype=torch.float32, requires_grad=True)
        image = render_color(color)
        if image.shape != (3, *inside.shape):
            raise ValueError('Expected native CHW RGB from the Gaussian renderer.')
        objective = (image[0]*inside).sum() + (image[1]*outside).sum() + image[2].sum()
        weight, = torch.autograd.grad(objective, color)
    return weight.detach()


def _scene(model):
    return SimpleNamespace(get_xyz=model.get_xyz, get_scaling=model.get_scaling,
                           get_rotation=model.get_rotation, get_opacity=model.get_opacity,
                           active_sh_degree=model.active_sh_degree)


@torch.no_grad()
def source_weights(model, constraints, ids):
    """Probe the full posed scene, returning weights for only the requested rows."""
    from gaussian_renderer import render

    previous = model.timestep
    rows = []
    try:
        for camera, slot, view in zip(constraints.cameras, constraints.slots, constraints.views, strict=True):
            model.select_mesh_by_timestep(slot)
            scene = _scene(model)

            def rgb(color):
                return render(camera, scene, constraints.pipe, constraints.black, override_color=color)['render']

            rows.append(color_weights(rgb, view.inside, view.outside, len(scene.get_xyz))[ids])
    finally:
        model.select_mesh_by_timestep(previous)
    return torch.stack(rows)


def classify(weights, bins, options):
    """Separate lack of raster contribution from reliable background conflict."""
    foreground, background, total = weights.sum(0).unbind(1)
    minimum = options['min_equivalent_pixels']
    foreground_supported = foreground >= minimum
    unobserved = total < minimum
    background_bins = sum((weights[[i for i, value in enumerate(bins) if value == group], :, 1].sum(0)
                           >= minimum).long() for group in sorted(set(bins)))
    conflict = (background_bins >= options['min_background_bins']) & (
        background >= options['background_to_foreground']*foreground)
    return foreground_supported, unobserved, conflict


@torch.no_grad()
def coverage_trial(model, constraints, parents, centers, log_sigma, opacity, weights, options):
    """Verify a bounded replacement batch against the exact native foreground alpha."""
    from gaussian_renderer import render

    if not len(parents):
        return dict(proposed=0, accepted=False, max_foreground_loss_px=0., max_background_gain_px=0.)
    previous = model.timestep
    keep = torch.ones(len(model.binding), dtype=torch.bool, device=parents.device)
    keep[parents] = False
    owners = model.binding[parents].repeat(2)
    local = centers.transpose(0, 1).reshape(-1, 3)
    scales, alpha = log_sigma.exp().repeat(2, 1), opacity[:, None].repeat(2, 1)
    accepted, foreground_loss, background_gain = True, 0., 0.
    try:
        for i, (camera, slot, view) in enumerate(zip(constraints.cameras, constraints.slots, constraints.views, strict=True)):
            model.select_mesh_by_timestep(slot)
            scene = _scene(model)
            face_scale = model.face_scaling[owners]
            xyz = (model.face_orien_mat[owners] @ local[..., None]).squeeze(-1)*face_scale+model.face_center[owners]
            alternative = SimpleNamespace(get_xyz=torch.cat((scene.get_xyz[keep], xyz)),
                get_scaling=torch.cat((scene.get_scaling[keep], scales*face_scale)),
                get_rotation=torch.cat((scene.get_rotation[keep], scene.get_rotation[parents].repeat(2, 1))),
                get_opacity=torch.cat((scene.get_opacity[keep], alpha)), active_sh_degree=scene.active_sh_degree)
            baseline_alpha, proposed_alpha = (
                render(camera, item, constraints.pipe, constraints.black,
                       override_color=torch.ones_like(item.get_xyz))['render'][0] for item in (scene, alternative))
            delta = proposed_alpha-baseline_alpha
            missing = float((-delta[view.inside]).clamp_min(0).sum())
            spill = float(delta[view.outside].clamp_min(0).sum())
            fg, bg = weights[i, :, :2].sum(0).tolist()
            accepted &= (missing <= options['coverage_slack_px']+options['coverage_loss_fraction']*fg
                         and spill <= options['coverage_slack_px']+options['coverage_loss_fraction']*bg)
            foreground_loss, background_gain = max(foreground_loss, missing), max(background_gain, spill)
    finally:
        model.select_mesh_by_timestep(previous)
    return dict(proposed=len(parents), accepted=accepted, max_foreground_loss_px=foreground_loss,
                max_background_gain_px=background_gain)


def _rgb_error(rgb, alpha, view):
    x0, y0, x1, y1 = view.box
    target = view.rgb
    if target.dtype == torch.uint8:
        target = target.float().div(255)
    straight = rgb[:, y0:y1, x0:x1] / alpha[y0:y1, x0:x1].clamp_min(1e-4)
    return (straight-target).abs().mean(0)


def _tile_sum(image, size):
    return F.avg_pool2d(image[None, None], size, stride=size, ceil_mode=True, divisor_override=1)[0, 0]


def _worst_tile(values, box, size):
    index = int(values.argmax())
    y, x = divmod(index, values.shape[1])
    x0, y0, x1, y1 = box
    left, top = x0+x*size, y0+y*size
    return dict(value=float(values[y, x]), box=[left, top, min(x1, left+size), min(y1, top+size)])


def _coverage_metrics(baseline_alpha, proposed_alpha, baseline_error, proposed_error, view,
                      regions, target, hole_threshold, foreground_budget, background_budget, options):
    """Compare the trial to original observations, using the baseline only to measure regressions."""
    x0, y0, x1, y1 = view.box
    crop = (slice(y0, y1), slice(x0, x1))
    inside = view.inside[crop]
    old, new = baseline_alpha[crop], proposed_alpha[crop]
    failures = []

    def check(item, value, budget, region=None):
        if value > budget:
            failures.append(dict(item=item, value=value, budget=budget, region=region))

    if not all(torch.isfinite(value).all() for value in
               (old[inside], new[inside], proposed_alpha[view.outside], baseline_alpha[view.outside],
                baseline_error[inside], proposed_error[inside])):
        return dict(failures=[dict(item='nonfinite_native_observation')])

    # Count new deficits without cancellation by improvements elsewhere in the frame.
    epsilon = 1e-6
    lost = (old-new).masked_fill(~inside, 0)
    lost = torch.where(lost > epsilon, lost, 0.)
    baseline_deficit = (target-old).clamp_min(0).masked_fill(~inside, 0)
    proposed_deficit = (target-new).clamp_min(0).masked_fill(~inside, 0)
    missing = torch.minimum((proposed_deficit-baseline_deficit).clamp_min(0), lost)
    new_target_crossings = (old >= target) & (new < target) & (lost > 0)
    new_holes = (old >= hole_threshold) & (new < hole_threshold) & (lost > 0)
    spill = float((proposed_alpha-baseline_alpha)[view.outside].clamp_min(0).sum())
    rgb_delta = (proposed_error-baseline_error).masked_fill(~inside, 0)
    tile = options['coverage_tile_px']
    tile_area = _tile_sum(inside.float(), tile).clamp_min(1)
    worst_deficit = _worst_tile(_tile_sum(missing, tile), view.box, tile)
    worst_rgb = _worst_tile(_tile_sum(rgb_delta, tile)/tile_area, view.box, tile)
    pixel_loss = float(missing.sum())
    hole_count = int(new_holes.count_nonzero())
    rgb_increase = float(rgb_delta.sum()/inside.count_nonzero().clamp_min(1))
    check('foreground_target_deficit_px', pixel_loss, foreground_budget)
    check('new_low_alpha_holes', hole_count, 0)
    check('background_gain_px', spill, background_budget)
    check('foreground_rgb_mae_increase', rgb_increase, options['coverage_rgb_mae_increase']+epsilon)
    check('worst_tile_target_deficit_px', worst_deficit['value'],
          options['coverage_local_deficit_px'], worst_deficit['box'])
    check('worst_tile_rgb_mae_increase', worst_rgb['value'],
          options['coverage_local_rgb_mae_increase']+epsilon, worst_rgb['box'])

    def describe_region(y, x):
        d = dict(pixels=len(y), foreground_target_deficit_px=float(missing[y, x].sum()),
                 new_target_crossings=int(new_target_crossings[y, x].count_nonzero()),
                 new_low_alpha_holes=int(new_holes[y, x].count_nonzero()),
                 original_rgb_mae_before=float(baseline_error[y, x].mean()),
                 original_rgb_mae_after=float(proposed_error[y, x].mean()),
                 rgb_mae_increase=float(rgb_delta[y, x].mean()))
        return d

    protected = {}
    for name, pixels in regions.items():
        y, x = pixels
        valid = (y >= y0) & (y < y1) & (x >= x0) & (x < x1)
        y, x = y[valid]-y0, x[valid]-x0
        valid = inside[y, x]
        y, x = y[valid], x[valid]
        if not len(y):
            continue
        observed = describe_region(y, x)
        protected[name] = observed
        check('region_target_deficit_px', observed['foreground_target_deficit_px'],
              options['coverage_region_deficit_px'], name)
        check('region_new_low_alpha_holes', observed['new_low_alpha_holes'], 0, name)
        check('region_rgb_mae_increase', observed['rgb_mae_increase'],
              options['coverage_rgb_mae_increase']+epsilon, name)

    return dict(failures=failures, foreground_target_deficit_px=pixel_loss,
                baseline_target_deficit_px=float(baseline_deficit.sum()),
                proposed_target_deficit_px=float(proposed_deficit.sum()),
                redundant_alpha_loss_px=float((lost-missing).sum()),
                new_target_crossings=int(new_target_crossings.count_nonzero()),
                new_low_alpha_holes=hole_count, background_gain_px=spill,
                original_rgb_mae_before=float(baseline_error[inside].mean()),
                original_rgb_mae_after=float(proposed_error[inside].mean()),
                rgb_mae_increase=rgb_increase, worst_deficit_tile=worst_deficit,
                worst_rgb_tile=worst_rgb, protected_regions=protected)


@contextmanager
def replacement_evaluator(model, constraints, parents, centers, log_sigma, opacity, weights, options):
    """Yield an exact posed subset evaluator; cache baseline images on the host."""
    from gaussian_renderer import render

    previous = model.timestep
    baseline = {}
    target = constraints.options['foreground_target']
    hole_threshold = constraints.options['low_alpha']
    regions = getattr(constraints, 'protected_regions', None)

    @torch.no_grad()
    def evaluate(indices):
        subset = parents.new_tensor(indices)
        selected = parents[subset]
        if not len(selected):
            return dict(proposed=0, accepted=False, frames=[])
        keep = torch.ones(len(model.binding), dtype=torch.bool, device=parents.device)
        keep[selected] = False
        owners = model.binding[selected].repeat(2)
        local = centers[subset].transpose(0, 1).reshape(-1, 3)
        scales = log_sigma[subset].exp().repeat(2, 1)
        alpha = opacity[subset, None].repeat(2, 1)
        features = model.get_features
        child_features = features[selected].repeat(2, 1, 1)
        alternative_features = torch.cat((features[keep], child_features))
        view_masses = weights[:, subset, :2].sum(1).tolist()
        frames = []

        def draw(camera, item, view):
            rgb = render(camera, item, constraints.pipe, constraints.black)['render']
            alpha_image = render(camera, item, constraints.pipe, constraints.black,
                                 override_color=torch.ones_like(item.get_xyz))['render'][0]
            return alpha_image, _rgb_error(rgb, alpha_image, view)

        try:
            for i, (camera, slot, view, row) in enumerate(zip(
                    constraints.cameras, constraints.slots, constraints.views, constraints.rows, strict=True)):
                model.select_mesh_by_timestep(slot)
                scene = _scene(model)
                scene.get_features = features
                scene.max_sh_degree = model.max_sh_degree
                face_scale = model.face_scaling[owners]
                xyz = (model.face_orien_mat[owners] @ local[..., None]).squeeze(-1)*face_scale+model.face_center[owners]
                alternative = SimpleNamespace(get_xyz=torch.cat((scene.get_xyz[keep], xyz)),
                    get_scaling=torch.cat((scene.get_scaling[keep], scales*face_scale)),
                    get_rotation=torch.cat((scene.get_rotation[keep], scene.get_rotation[selected].repeat(2, 1))),
                    get_opacity=torch.cat((scene.get_opacity[keep], alpha)), active_sh_degree=scene.active_sh_degree,
                    max_sh_degree=scene.max_sh_degree, get_features=alternative_features)
                if i not in baseline:
                    old_alpha, old_error = draw(camera, scene, view)
                    baseline[i] = old_alpha.cpu(), old_error.cpu()
                else:
                    old_alpha, old_error = (value.to(parents.device) for value in baseline[i])
                new_alpha, new_error = draw(camera, alternative, view)
                protected = {} if regions is None else dict(regions[i])
                if 'face' not in protected and view.face is not None:
                    x0, y0, _, _ = view.box
                    protected['face'] = view.face + view.face.new_tensor([[y0], [x0]])
                fg, bg = view_masses[i]
                base, fraction = options['coverage_slack_px'], options['coverage_loss_fraction']
                metrics = _coverage_metrics(old_alpha, new_alpha, old_error, new_error, view, protected,
                                            target, hole_threshold, base+fraction*fg, base+fraction*bg, options)
                frames.append(dict(frame=row['frame'], **metrics))
        finally:
            model.select_mesh_by_timestep(previous)
        return dict(proposed=len(selected), accepted=not any(frame['failures'] for frame in frames),
                    max_foreground_loss_px=max(frame.get('foreground_target_deficit_px', 0.) for frame in frames),
                    max_background_gain_px=max(frame.get('background_gain_px', 0.) for frame in frames),
                    frames=frames)

    try:
        yield evaluate
    finally:
        baseline.clear()


class FootprintLimits:
    """Point-aligned local scale constraints with lineage through topology changes."""
    def __init__(self, scaling):
        self.cap = torch.full_like(scaling, torch.inf)
        self.lineage = torch.full((len(scaling),), -1, dtype=torch.long, device=scaling.device)
        self.overshoot = torch.zeros_like(self.lineage, dtype=scaling.dtype)
        self.counts = scaling.new_zeros(3, dtype=torch.long)
        self.peak_overshoot = scaling.new_zeros(())
        self.next_lineage = 0

    def append(self, count, parents):
        if parents is None:
            cap = self.cap.new_full((count, 3), torch.inf)
            lineage = self.lineage.new_full((count,), -1)
            overshoot = self.overshoot.new_zeros(count)
        else:
            if parents.shape != (count,):
                raise ValueError('Gaussian parent roster differs from its append operation.')
            cap, lineage, overshoot = self.cap[parents], self.lineage[parents], self.overshoot[parents]
            self.counts[1] += (lineage >= 0).count_nonzero()
        self.cap = torch.cat((self.cap, cap))
        self.lineage = torch.cat((self.lineage, lineage))
        self.overshoot = torch.cat((self.overshoot, overshoot))

    def prune(self, keep):
        self.counts[2] += ((~keep) & (self.lineage >= 0)).count_nonzero()
        self.cap, self.lineage, self.overshoot = self.cap[keep], self.lineage[keep], self.overshoot[keep]

    def set(self, ids, limit):
        missing = ids[self.lineage[ids] < 0]
        self.lineage[missing] = torch.arange(self.next_lineage, self.next_lineage+len(missing),
                                             device=ids.device, dtype=torch.long)
        self.next_lineage += len(missing)
        self.cap[ids] = torch.minimum(self.cap[ids], limit)
        return self.lineage[ids]

    def apply(self, param, optimizer, *, measure=True):
        if measure:
            excess = (param-self.cap).amax(1).clamp_min(0)
            self.overshoot = torch.maximum(self.overshoot, excess)
            self.peak_overshoot = torch.maximum(self.peak_overshoot, excess.max())
            self.counts[0] += (excess > 1e-7).count_nonzero()
        hit = param > self.cap
        param.copy_(torch.minimum(param, self.cap))
        state = optimizer.state.get(param)
        if state is not None:
            state['exp_avg'].masked_fill_(hit & (state['exp_avg'] < 0), 0)

    def summary(self):
        hits, inherited, pruned = self.counts.tolist()
        tracked = self.lineage >= 0
        return dict(lineages_registered=self.next_lineage, live_points=int(tracked.count_nonzero()),
                    projected_point_updates=hits, inherited_children=inherited, pruned_points=pruned,
                    max_preprojection_growth_ratio=float(self.peak_overshoot.exp()))

