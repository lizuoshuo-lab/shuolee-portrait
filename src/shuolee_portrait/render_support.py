"""Measure sorted Gaussian contributions using the renderer's color derivative."""

from types import SimpleNamespace

import torch


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

