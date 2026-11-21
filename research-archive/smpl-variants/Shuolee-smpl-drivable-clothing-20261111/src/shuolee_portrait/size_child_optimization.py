"""Fit bounded child rows against native observations without modifying the live scene."""

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from .render_support import _coverage_metrics, _rgb_error, _tile_sum, child_world_rotation
from .shape_refinement import _projection


@dataclass
class SizeParentPriority:
    scores: torch.Tensor
    peak_radii: torch.Tensor
    residuals: torch.Tensor
    radii: torch.Tensor
    view_residuals: torch.Tensor


@dataclass
class SizeChildResult:
    centers: torch.Tensor
    log_sigma: torch.Tensor
    opacity_logits: torch.Tensor
    dc: torch.Tensor
    moments: dict
    audit: dict
    child_rotation: torch.Tensor | None = None


@contextmanager
def _preserve_pose(model):
    names = ('timestep', 'face_center', 'face_orien_mat', 'face_scaling', 'face_orien_quat')
    old = {name: getattr(model, name) for name in names if hasattr(model, name)}
    try:
        yield
    finally:
        for name, value in old.items():
            setattr(model, name, value)


def _model_scene(model):
    return SimpleNamespace(get_xyz=model.get_xyz.detach(), get_scaling=model.get_scaling.detach(),
                           get_rotation=model.get_rotation.detach(), get_opacity=model.get_opacity.detach(),
                           get_features=model.get_features.detach(), active_sh_degree=model.active_sh_degree,
                           max_sh_degree=model.max_sh_degree)


def _draw(render, constraints, i, scene):
    camera = constraints.cameras[i]
    rgb = render(camera, scene, constraints.pipe, constraints.black)
    alpha = render(camera, scene, constraints.pipe, constraints.black,
                   override_color=torch.ones_like(scene.get_xyz))['render'][0]
    return rgb['render'], alpha, rgb['radii']


def _target_rgb(view):
    return view.rgb.float().div(255) if view.rgb.dtype == torch.uint8 else view.rgb


def _target_alpha(view):
    return view.mask.reshape(view.inside.shape)


def _boxes(xyz, radii, bounds, matrix, *, pad=0):
    xy, depth = _projection(xyz, *matrix)
    xy = xy - .5
    radius = radii.float() + pad
    x0, y0, x1, y1 = bounds
    x, y = xy.unbind(-1)
    left, right = (x-radius).floor().long().clamp(x0, x1), (x+radius).ceil().long().add(1).clamp(x0, x1)
    top, bottom = (y-radius).floor().long().clamp(y0, y1), (y+radius).ceil().long().add(1).clamp(y0, y1)
    visible = (radii > 0) & (depth > 0) & (right > left) & (bottom > top)
    return torch.stack((left, top, right, bottom), 1), visible


def _box_sums(image, boxes, origin):
    x0, y0 = origin
    left, top, right, bottom = (boxes-boxes.new_tensor([x0, y0, x0, y0])).unbind(1)
    integral = F.pad(image.cumsum(0).cumsum(1), (1, 0, 1, 0))
    return integral[bottom, right]-integral[top, right]-integral[bottom, left]+integral[top, left]


@torch.no_grad()
def measure_size_parent_priority(model, constraints, ids, weights, screen_limit_px):
    """Score measured raster size and reliable source residual at each supported footprint."""
    from gaussian_renderer import render

    if screen_limit_px <= 0 or weights.shape != (len(constraints.views), len(ids), 3):
        raise ValueError('Invalid native parent ranking support or screen size.')
    radii_rows, residual_rows = [], []
    with _preserve_pose(model):
        for i, view in enumerate(constraints.views):
            model.select_mesh_by_timestep(constraints.slots[i])
            scene = _model_scene(model)
            rgb, alpha, radii = _draw(render, constraints, i, scene)
            parent_radii = radii[ids]
            x0, y0, x1, y1 = view.box
            crop = (slice(y0, y1), slice(x0, x1))
            boxes, valid = _boxes(scene.get_xyz[ids], parent_radii, view.box, constraints.matrices[i])
            reliable = view.inside[crop].float()
            area = _box_sums(reliable, boxes, (x0, y0))
            source_rgb = _target_rgb(view)
            source_alpha = _target_alpha(view)[crop]
            color_error = (rgb[:, crop[0], crop[1]]/alpha[crop].clamp_min(1e-4)-source_rgb).abs().mean(0)
            opacity_error = (alpha[crop]-source_alpha).abs()
            residual = _box_sums((color_error+.5*opacity_error)*reliable, boxes, (x0, y0))
            residual_rows.append((residual.clamp_min(0)/area.clamp_min(1)).masked_fill(~valid, 0))
            radii_rows.append(parent_radii)
    radii, view_residuals = torch.stack(radii_rows), torch.stack(residual_rows)
    mass = weights[..., 0].clamp_min(0)
    residuals = (view_residuals*mass).sum(0)/mass.sum(0).clamp_min(1e-6)
    peak_radii = radii.amax(0)
    scores = (peak_radii.float()/screen_limit_px) * mass.sum(0).log1p() * (residuals+.02)
    return SizeParentPriority(scores, peak_radii, residuals, radii, view_residuals)


def _view_cycle(bins, cycle):
    """Interleave source bins and rotate which views lead successive full sweeps."""
    groups = [[i for i, value in enumerate(bins) if value == group] for group in sorted(set(bins))]
    shift = cycle % len(groups)
    groups = groups[shift:] + groups[:shift]
    for index, views in enumerate(groups):
        shift = cycle % len(views)
        groups[index] = views[shift:] + views[:shift]
    return [group[j] for j in range(max(map(len, groups))) for group in groups if j < len(group)]


@dataclass
class _FitView:
    index: int
    fixed: SimpleNamespace
    face_center: torch.Tensor
    face_orient: torch.Tensor
    face_scale: torch.Tensor
    rotation: torch.Tensor
    rest: torch.Tensor
    crop: tuple
    target_rgb: torch.Tensor
    target_alpha: torch.Tensor
    inside: torch.Tensor
    outside: torch.Tensor
    source_weights: tuple
    foreground_equivalent_pixels: float
    parent_radii: torch.Tensor
    parent_rgb: torch.Tensor
    parent_alpha: torch.Tensor
    native_parent_alpha: torch.Tensor
    native_parent_error: torch.Tensor
    parent_error: torch.Tensor
    tile_crop: tuple
    tile_area: torch.Tensor
    regions: dict
    local_regions: list
    foreground_budget: float
    background_budget: float
    frame_inside_count: torch.Tensor
    representative: tuple | None


def _protected_regions(constraints, index):
    view = constraints.views[index]
    protected = getattr(constraints, 'protected_regions', None)
    regions = {} if protected is None else dict(protected[index])
    if 'face' not in regions and view.face is not None:
        x0, y0, _, _ = view.box
        regions['face'] = view.face+view.face.new_tensor([[y0], [x0]])
    return regions


def _local_regions(regions, view, box):
    vx, vy, vr, vb = view.box
    left, top, right, bottom = box
    local = []
    for y, x in regions.values():
        native = (x >= vx) & (x < vr) & (y >= vy) & (y < vb)
        y, x = y[native], x[native]
        trusted = view.inside[y, x]
        y, x = y[trusted], x[trusted]
        cropped = (x >= left) & (x < right) & (y >= top) & (y < bottom)
        if bool(cropped.any()):
            local.append((y[cropped]-top, x[cropped]-left, len(y)))
    return local


@torch.no_grad()
def _fit_view(render, model, constraints, i, parents, owners, keep, children, weights, options, child_rotation):
    view = constraints.views[i]
    model.select_mesh_by_timestep(constraints.slots[i])
    parent = _model_scene(model)
    rgb, alpha, radii = _draw(render, constraints, i, parent)
    parent_radii = radii[parents]
    height, width = view.inside.shape
    canvas = (0, 0, width, height)
    # A moved child can extend beyond the parent's old 3-sigma box.
    padded_radii = (1.3*parent_radii).ceil().int().clamp_min(1)
    boxes, valid = _boxes(parent.get_xyz[parents], padded_radii, canvas, constraints.matrices[i], pad=8)
    vx, vy, vr, vb = view.box
    tile = options['coverage_tile_px']
    left, top, right, bottom = boxes[0].tolist() if bool(valid[0]) else canvas
    representative = (left, top, right, bottom) if bool(valid[0]) and int(parent_radii[0]) > 0 else None
    x0, y0 = max(0, vx+((left-vx)//tile)*tile), max(0, vy+((top-vy)//tile)*tile)
    x1, y1 = min(width, vx+math.ceil((right-vx)/tile)*tile), min(height, vy+math.ceil((bottom-vy)/tile)*tile)
    crop = (slice(y0, y1), slice(x0, x1))
    inside, outside = view.inside[crop], view.outside[crop]
    edge = ~(inside | outside)
    target_alpha = _target_alpha(view)[crop]
    source = _target_rgb(view)
    target_rgb = source.new_zeros(3, y1-y0, x1-x0)
    left, top, right, bottom = max(x0, vx), max(y0, vy), min(x1, vr), min(y1, vb)
    tile_crop, tile_area = (), None
    if left < right and top < bottom:
        tile_crop = (slice(top-y0, bottom-y0), slice(left-x0, right-x0))
        target_rgb[:, tile_crop[0], tile_crop[1]] = source[:, top-vy:bottom-vy, left-vx:right-vx]
        tile_area = _tile_sum(view.inside[top:bottom, left:right].float(), tile).clamp_min(1)
    foreground, background = weights[i, 0, :2].tolist()
    # An occluded parent provides no positive photometric observation.
    # Independent exterior and boundary supervision remains active.
    observed = inside if foreground > 0 else torch.zeros_like(inside)
    source_weights = tuple(mask.float()/mask.count_nonzero().clamp_min(1) for mask in (observed, outside, edge))
    fixed = SimpleNamespace(**vars(parent))
    for key in ('get_xyz', 'get_scaling', 'get_rotation', 'get_opacity', 'get_features'):
        setattr(fixed, key, getattr(parent, key)[keep])
    parent_rgb, parent_alpha = rgb[:, crop[0], crop[1]].contiguous(), alpha[crop].contiguous()
    parent_error = (parent_rgb/parent_alpha.clamp_min(1e-4)-target_rgb).abs().mean(0)
    slack, fraction = options['coverage_slack_px'], options['coverage_loss_fraction']
    regions = _protected_regions(constraints, i)
    rotation = (parent.get_rotation[parents].repeat(children, 1) if child_rotation is None else
                child_world_rotation(model, owners, _child_major(child_rotation)))
    return _FitView(i, fixed, model.face_center[owners], model.face_orien_mat[owners], model.face_scaling[owners],
                    rotation,
                    parent.get_features[parents, 1:].repeat(children, 1, 1), crop, target_rgb, target_alpha,
                    inside, outside, source_weights, foreground, parent_radii, parent_rgb, parent_alpha,
                    alpha.clone(), _rgb_error(rgb, alpha, view), parent_error, tile_crop, tile_area,
                    regions, _local_regions(regions, view, (x0, y0, x1, y1)), slack+fraction*foreground,
                    slack+fraction*background, view.inside[vy:vb, vx:vr].count_nonzero().clamp_min(1), representative)


def _child_major(value):
    return value.transpose(0, 1).reshape(-1, *value.shape[2:])


def _candidate(fit, params):
    centers, scales = _child_major(params['xyz']), _child_major(params['scaling']).exp()
    xyz = (fit.face_orient @ centers[..., None]).squeeze(-1)*fit.face_scale+fit.face_center
    features = torch.cat((_child_major(params['f_dc']), fit.rest), dim=1)
    fixed = fit.fixed
    return SimpleNamespace(get_xyz=torch.cat((fixed.get_xyz, xyz)),
                           get_scaling=torch.cat((fixed.get_scaling, scales*fit.face_scale)),
                           get_rotation=torch.cat((fixed.get_rotation, fit.rotation)),
                           get_opacity=torch.cat((fixed.get_opacity, _child_major(params['opacity']).sigmoid())),
                           get_features=torch.cat((fixed.get_features, features)),
                           active_sh_degree=fixed.active_sh_degree, max_sh_degree=fixed.max_sh_degree)


def _loss(fit, rgb, alpha):
    straight = rgb/alpha.clamp_min(1e-4)
    error = (straight-fit.target_rgb).abs().mean(0)
    premultiplied = (rgb-fit.target_rgb*fit.target_alpha).abs().mean(0)
    alpha_error = (alpha-fit.target_alpha).abs()
    foreground, exterior, edge = fit.source_weights
    return ((error*foreground).sum(), (alpha_error*(foreground+.5*exterior+.35*edge)).sum(),
            (premultiplied*foreground).sum(), error)


def _source_objective(loss, *, fit_alpha):
    color, alpha_error, premultiplied, _ = loss
    return .30*color + .70*premultiplied + alpha_error if fit_alpha else color


def _guard_loss(fit, alpha, color_error, options, target, hole_threshold):
    inside, outside = fit.inside, fit.outside
    old = fit.parent_alpha
    epsilon = 1e-6
    old_minus_new = (old-alpha).masked_fill(~inside, 0)
    lost = torch.where(old_minus_new > epsilon, old_minus_new, 0.)
    missing = torch.minimum(((target-alpha).relu()-(target-old).relu()).relu(), lost)
    rgb_delta = (color_error-fit.parent_error).masked_fill(~inside, 0)
    at_risk = inside & (old >= hole_threshold) & (lost > 0)
    holes = (hole_threshold-alpha).relu().masked_fill(~at_risk, 0)
    spill = (alpha-old).relu().masked_fill(~outside, 0)
    # Use the verifier's tolerance values and native tile origin. Relative
    # penalties prevent an RGB breach in one tile from averaging away.
    rgb_tolerance = options['coverage_rgb_mae_increase']+epsilon
    tile_tolerance = options['coverage_local_rgb_mae_increase']+epsilon
    loss = 6*holes.amax()
    loss += .08*(missing.sum()/fit.foreground_budget-1).relu()
    loss += .08*(spill.sum()/fit.background_budget-1).relu()
    loss += 3*(rgb_delta.sum()/fit.frame_inside_count-rgb_tolerance).relu()
    if fit.tile_crop:
        tile = options['coverage_tile_px']
        local_rgb = _tile_sum(rgb_delta[fit.tile_crop], tile)/fit.tile_area
        local_alpha = _tile_sum(missing[fit.tile_crop], tile)
        loss += 12*(local_rgb.amax()-tile_tolerance).relu()
        loss += .10*(local_alpha.amax()/options['coverage_local_deficit_px']-1).relu()
    for y, x, count in fit.local_regions:
        loss += 4*(rgb_delta[y, x].sum()/count-rgb_tolerance).relu()
        loss += .10*(missing[y, x].sum()/options['coverage_region_deficit_px']-1).relu()
    return loss


@torch.no_grad()
def _checkpoint(render, constraints, fits, params, options, cycle, updates, *, require_smaller_radius, fit_alpha):
    target, hole_threshold = constraints.options['foreground_target'], constraints.options['low_alpha']
    errors_by_bin, failures_by_view, severity = {}, {}, []
    child_peak = parent_peak = 0
    parent_support, metrics = False, []
    for fit in fits:
        index = fit.index
        view = constraints.views[index]
        rgb, alpha, radii = _draw(render, constraints, index, _candidate(fit, params))
        y, x = fit.crop
        loss = _loss(fit, rgb[:, y, x], alpha[y, x])
        objective = float(_source_objective(loss, fit_alpha=fit_alpha))
        errors_by_bin.setdefault(constraints.bins[index], []).append(objective)
        parent_support |= bool(fit.source_weights[0].any())
        k = params['opacity'].shape[1]
        child_peak = max(child_peak, int(radii[-k:].max()))
        parent_peak = max(parent_peak, int(fit.parent_radii[0]))
        evidence = _coverage_metrics(fit.native_parent_alpha, alpha, fit.native_parent_error,
                                     _rgb_error(rgb, alpha, view), view, fit.regions, target, hole_threshold,
                                     fit.foreground_budget, fit.background_budget, options)
        failures = evidence['failures']
        if failures:
            failures_by_view[constraints.rows[index]['frame']] = Counter(row['item'] for row in failures)
        for failure in failures:
            budget = failure.get('budget', 0)
            severity.append((failure.get('value', math.inf)-budget)/(budget if budget > 0 else 1))
        metrics.append(dict(rgb_mae=float(loss[0]), alpha_error=float(loss[1]), radii_px=[radii[-k:].tolist()]))
    rgb_alpha = sum(sum(values)/len(values) for values in errors_by_bin.values())/len(errors_by_bin)
    if require_smaller_radius and child_peak >= parent_peak:
        severity.append(1.+(child_peak-parent_peak)/max(parent_peak, 1))
    finite = math.isfinite(rgb_alpha) and all(math.isfinite(value) for value in severity)
    feasible = finite and parent_support and not severity
    score = (not feasible, max(severity, default=0.), sum(severity), len(failures_by_view), rgb_alpha)
    counters = Counter()
    for failures in failures_by_view.values():
        counters.update(failures)
    audit = dict(cycle=cycle, updates=updates, views_checked=len(fits), source_objective=rgb_alpha if finite else None,
                 guard_feasible=feasible, worst_normalized_guard_excess=score[1] if finite else None,
                 total_normalized_guard_excess=score[2] if finite else None, rejected_views=list(failures_by_view),
                 failures=dict(counters), parent_peak_radius_px=parent_peak, child_peak_radius_px=child_peak)
    return score, audit, metrics, finite, parent_support


@torch.no_grad()
def _observe(render, constraints, fit, params):
    rgb, alpha, radii = _draw(render, constraints, fit.index, _candidate(fit, params))
    y, x = fit.crop
    rgb, alpha = rgb[:, y, x].contiguous(), alpha[y, x].contiguous()
    rgb_error, alpha_error, _, _ = _loss(fit, rgb, alpha)
    n, k = params['opacity'].shape[:2]
    radii = radii[-n*k:].reshape(k, n).T
    return (rgb, alpha), dict(rgb_mae=float(rgb_error), alpha_error=float(alpha_error), radii_px=radii.tolist())


def _save_composite(path, names, stages):
    import numpy as np
    from PIL import Image, ImageDraw

    panels = []
    for rgb, alpha in stages:
        channels = [rgb, alpha.expand(3, -1, -1)]
        panels.append([(image.permute(1, 2, 0).detach().float().clamp(0, 1)*255).round().byte().cpu().numpy()
                       for image in channels])
    h, w = stages[0][1].shape
    column = max(w, 105)
    canvas = Image.new('RGB', (len(stages)*(column+3)+3, 2*h+39), '#161616')
    labels = ImageDraw.Draw(canvas)
    for i, (title, panels_at_stage) in enumerate(zip(names, panels, strict=True)):
        ox = 3 + i*(column+3)
        labels.text((ox, 3), title, fill='white')
        for j, panel in enumerate(panels_at_stage):
            canvas.paste(Image.fromarray(np.ascontiguousarray(panel)), (ox, 20+j*(h+3)))
    labels.text((3, 2*h+26), 'RGB above, alpha below. Native crop, no resizing.', fill='white')
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _source_crop(row, box):
    import numpy as np
    from PIL import Image

    with Image.open(row['image']) as image:
        pixels = np.array(image.convert('RGB').crop(box))
    return torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255)


def _save_representative(directory, fit, row, proposal, optimized):
    left, top, right, bottom = fit.representative
    y0, x0 = fit.crop[0].start, fit.crop[1].start
    y, x = slice(top-y0, bottom-y0), slice(left-x0, right-x0)
    raw = (_source_crop(row, fit.representative), fit.target_alpha[y, x])
    crops = [(rgb[:, y, x], alpha[y, x]) for rgb, alpha in
             ((fit.parent_rgb, fit.parent_alpha), proposal, optimized)]
    directory = Path(directory)
    path = directory/f'representative-view-{fit.index:02d}.png'
    _save_composite(path, ('Source', 'Parent', 'Proposal', 'Optimized'), [raw, *crops])
    return dict(frame=row['frame'], view_index=fit.index, path=path.name, box=list(fit.representative),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@torch.no_grad()
def save_size_decisions(model, constraints, representatives, diagnostic_dir):
    """Render the actual live scene after the event decision at the same native crop."""
    from gaussian_renderer import render

    if diagnostic_dir is None:
        return []
    decisions = []
    with _preserve_pose(model):
        for representative in representatives:
            i = representative['view_index']
            row, view = constraints.rows[i], constraints.views[i]
            box = representative['box']
            x0, y0, x1, y1 = box
            y, x = slice(y0, y1), slice(x0, x1)
            model.select_mesh_by_timestep(constraints.slots[i])
            rgb, alpha, _ = _draw(render, constraints, i, _model_scene(model))
            source = (_source_crop(row, box), _target_alpha(view)[y, x])
            path = Path(diagnostic_dir)/f'decision-view-{i:02d}.png'
            _save_composite(path, ('Source', 'Final scene'), (source, (rgb[:, y, x], alpha[y, x])))
            decisions.append(dict(frame=row['frame'], view_index=i, path=path.name, box=box,
                                  sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return decisions


@torch.no_grad()
def _project_params(params, optimizer, bounds):
    hits = {}
    for name, parameter in params.items():
        lower, upper = bounds[name]
        invalid = ((parameter < lower) | (parameter > upper))
        parameter.copy_(parameter.clamp(min=lower, max=upper))
        moment = optimizer.state[parameter].get('exp_avg')
        if moment is not None:
            outward = ((parameter <= lower) & (moment > 0)) | ((parameter >= upper) & (moment < 0))
            moment.masked_fill_(outward, 0)
        hits[name] = invalid.count_nonzero()
    return hits


def _adam_groups(model, params, parent_width):
    source = {group['name']: group for group in model.optimizer.param_groups}
    # Center LR is relative to the parent's local footprint, in triangle coordinates.
    # Adaptive moments remain in native parameter units for later migration.
    center_lr = float(parent_width[0]) * .016
    rates = dict(xyz=center_lr, scaling=.023, opacity=.075, f_dc=.016)
    return [dict(name=name, params=[param], lr=rates[name], betas=source[name]['betas'],
                 eps=source[name]['eps']) for name, param in params.items()]


def _adam_snapshot(optimizer, params):
    return {group['name']: dict(step=int(state['step']), betas=group['betas'],
                                exp_avg=state['exp_avg'].clone(), exp_avg_sq=state['exp_avg_sq'].clone())
            for group in optimizer.param_groups if (state := optimizer.state[params[group['name']]])}


@torch.no_grad()
def optimize_size_children(model, constraints, parents, centers, log_sigma, opacity_logits, dc, upper_log_sigma,
                           weights, *, max_cycles=8, guard_options, diagnostic_dir=None,
                           child_rotation=None, fit_opacity=True, require_smaller_radius=True,
                           max_normal_center=None):
    """Fit one parent's children across every source, selecting after complete guard sweeps."""
    from gaussian_renderer import render
    from utils.sh_utils import C0
    from .sh0_main import DCColorProjection

    start = perf_counter()
    if centers.ndim != 3 or len(parents) != 1 or centers.shape[0] != 1 or centers.shape[2] != 3:
        raise ValueError('Expected one independent parent with three local coordinates per child.')
    n, k, _ = centers.shape
    num_views = len(constraints.views)
    if (k < 2 or log_sigma.shape != (n, k, 3) or opacity_logits.shape != (n, k, 1)
            or dc.shape != (n, k, 1, 3) or weights.shape != (num_views, n, 3)
            or num_views != len(constraints.bins)):
        raise ValueError('Inconsistent Gaussian replacement parameters or source weights.')
    if not (1 <= max_cycles <= 8 and 2 <= num_views <= 24) or model.active_sh_degree != 0 or model.max_sh_degree != 0:
        raise ValueError('Size-child fitting requires bounded full source sweeps and the SH0 pipeline.')
    if any(guard_options[key] <= 0 for key in ('coverage_slack_px', 'coverage_local_deficit_px',
                                              'coverage_region_deficit_px', 'coverage_tile_px')):
        raise ValueError('Expected positive native guard budgets and tile size.')
    cap = upper_log_sigma[:, None] if upper_log_sigma.ndim == 2 else upper_log_sigma
    cap = torch.broadcast_to(cap, log_sigma.shape)
    if not all(bool(torch.isfinite(value).all()) for value in (centers, log_sigma, opacity_logits, dc, weights)):
        raise ValueError('Candidate parameters and source weights must be finite.')
    if child_rotation is not None and (child_rotation.shape != (n, k, 4)
            or not bool(torch.isfinite(child_rotation).all())
            or not bool((child_rotation.norm(dim=-1) > 1e-8).all())):
        raise ValueError('Each child requires a finite local wxyz rotation.')
    if max_normal_center is not None and (not math.isfinite(max_normal_center) or max_normal_center <= 0):
        raise ValueError('Expected a positive bound on child normal center offsets.')
    if bool(torch.isnan(cap).any()) or bool(torch.isneginf(cap).any()):
        raise ValueError('Invalid child scale caps.')

    parent_width = model._scaling[parents].detach().amax(1).exp()
    keep = torch.ones(len(model.binding), dtype=torch.bool, device=parents.device)
    keep[parents] = False
    owners = model.binding[parents].repeat(k)
    initial = dict(xyz=centers.detach(), scaling=log_sigma.detach(), opacity=opacity_logits.detach(), f_dc=dc.detach())
    params = {name: torch.nn.Parameter(value.clone(), requires_grad=name != 'opacity' or fit_opacity)
              for name, value in initial.items()}
    trainable = {name: parameter for name, parameter in params.items() if parameter.requires_grad}
    adam_groups = _adam_groups(model, trainable, parent_width)
    optimizer = torch.optim.Adam(adam_groups)
    dc_cap = DCColorProjection(model._features_dc, C0).bound
    # A child can move at most a quarter parent sigma on each local axis.
    displacement = .25*parent_width[:, None, None]
    lower_scale = torch.minimum(initial['scaling']-.55, cap)
    opacity_bounds = opacity_logits.new_tensor([.005, .995]).logit().tolist()
    lower_center, upper_center = initial['xyz']-displacement, initial['xyz']+displacement
    if max_normal_center is not None:
        lower_center[..., 1].clamp_(min=-max_normal_center)
        upper_center[..., 1].clamp_(max=max_normal_center)
    bounds = dict(xyz=(lower_center, upper_center),
                  scaling=(lower_scale, cap),
                  opacity=tuple(opacity_logits.new_tensor(value) for value in opacity_bounds),
                  f_dc=(dc.new_tensor(-dc_cap), dc.new_tensor(dc_cap)))
    if not fit_opacity:
        # Only the Gaussian geometry and SH0 color can move in this trial.
        del bounds['opacity']
    projections = _project_params(trainable, optimizer, bounds)
    proposal = {name: value.detach().clone() for name, value in params.items()}
    with _preserve_pose(model):
        fits = [_fit_view(render, model, constraints, i, parents, owners, keep, k, weights,
                          guard_options, child_rotation) for i in range(num_views)]
    proposal_score, proposal_check, proposal_metrics, finite, parent_support = _checkpoint(
        render, constraints, fits, proposal, guard_options, 0, 0,
        require_smaller_radius=require_smaller_radius, fit_alpha=fit_opacity)
    if not finite:
        raise RuntimeError('Initial size-child proposal produced a nonfinite native render.')
    initial_rates = {group['name']: group['lr'] for group in adam_groups}
    bins = Counter(constraints.bins)
    target, hole_threshold = constraints.options['foreground_target'], constraints.options['low_alpha']
    updates, stable = 0, 0
    cycles = []
    best = None
    invalid = None
    stop_reason = 'cycle_budget' if parent_support else 'no_supported_optimization_view'
    for cycle in range(max_cycles if parent_support else 0):
        order = _view_cycle(constraints.bins, cycle)
        for group in optimizer.param_groups:
            group['lr'] = initial_rates[group['name']]*.80**cycle
        for index in order:
            fit = fits[index]
            optimizer.zero_grad(set_to_none=True)
            with torch.enable_grad():
                rgb, alpha, _ = _draw(render, constraints, index, _candidate(fit, params))
                y, x = fit.crop
                alpha = alpha[y, x]
                losses = _loss(fit, rgb[:, y, x], alpha)
                guard = _guard_loss(fit, alpha, losses[-1], guard_options, target, hole_threshold)
                # A bin with more eligible frames does not get extra objective mass.
                weight = num_views/(len(bins)*bins[constraints.bins[index]])
                objective = weight*(_source_objective(losses, fit_alpha=fit_opacity)+guard)
                objective.backward()
            valid = all(value.grad is not None for value in trainable.values())
            if not valid or not bool(torch.stack([torch.isfinite(value.grad).all() for value in trainable.values()]
                                                 + [torch.isfinite(objective)]).all()):
                invalid = 'nonfinite_candidate_objective_or_gradient'
                break
            optimizer.step()
            for name, hit in _project_params(trainable, optimizer, bounds).items():
                projections[name] += hit
            updates += 1
        if invalid:
            break
        score, check, metrics, finite, _ = _checkpoint(render, constraints, fits, params, guard_options,
                                                       cycle+1, updates, require_smaller_radius=require_smaller_radius,
                                                       fit_alpha=fit_opacity)
        check['view_order'] = [constraints.rows[index]['frame'] for index in order]
        cycles.append(check)
        if not finite:
            invalid = 'nonfinite_full_cycle_checkpoint'
            break
        if best is None or score < best[0]:
            best = (score, check, metrics, {name: value.detach().clone() for name, value in params.items()},
                    _adam_snapshot(optimizer, trainable))
        previous = cycles[-2] if len(cycles) > 1 else proposal_check
        converged = check['guard_feasible'] and previous['guard_feasible'] and (
            abs(check['source_objective']-previous['source_objective']) <= 1e-5)
        stable = stable+1 if converged else 0
        if len(cycles) >= 4 and stable >= 2:
            stop_reason = 'two_stable_feasible_full_sweeps'
            break

    if best is None:
        selected, optimized, moments, optimized_metrics = proposal_check, proposal, {}, proposal_metrics
        status = invalid or stop_reason
    else:
        _, selected, optimized_metrics, optimized, moments = best
        status = invalid or ('fitted' if selected['guard_feasible'] else 'no_feasible_checkpoint')

    frames = []
    delta = {name: optimized[name]-proposal[name] for name in params}
    local_center_delta = delta['xyz'].norm(dim=-1)
    for fit in fits:
        row = constraints.rows[fit.index]
        parent_color, parent_alpha, _, _ = _loss(fit, fit.parent_rgb, fit.parent_alpha)
        center_delta_m = local_center_delta*fit.face_scale.reshape(k, n).T
        frames.append(dict(frame=row['frame'], view_index=fit.index, angular_bin=constraints.bins[fit.index],
                           foreground_equivalent_pixels=fit.foreground_equivalent_pixels,
                           foreground_supervision=bool(fit.source_weights[0].any()),
                           exterior_supervision=bool(fit.source_weights[1].any()),
                           boundary_supervision=bool(fit.source_weights[2].any()),
                           parent=dict(rgb_mae=float(parent_color), alpha_error=float(parent_alpha),
                                       radii_px=fit.parent_radii.tolist()),
                           proposal=proposal_metrics[fit.index], optimized=optimized_metrics[fit.index],
                           max_optimized_center_move_m=float(center_delta_m.max())))

    representatives = []
    if diagnostic_dir is not None:
        eligible = [fit for fit in fits if fit.representative is not None]
        for fit in sorted(eligible, key=lambda value: (-int(value.parent_radii[0]), value.index))[:2]:
            row = constraints.rows[fit.index]
            before, _ = _observe(render, constraints, fit, proposal)
            after, _ = _observe(render, constraints, fit, optimized)
            representatives.append(_save_representative(diagnostic_dir, fit, row, before, after))

    audit = dict(status=status, parent_rows=parents.tolist(), children_per_parent=k, optimizer_updates=updates,
                 optimizer_update_budget=max_cycles*num_views, optimizer_view_budget=num_views,
                 budget_cycles=max_cycles, cycles=cycles, stop_reason=invalid or stop_reason,
                 checkpoint_updates=selected['updates'], checkpoint_cycle=selected['cycle'],
                 checkpoint_feasible=selected['guard_feasible'] and selected['cycle'] > 0, proposal_check=proposal_check,
                 seconds=perf_counter()-start, views=frames, representatives=representatives,
                 targets=('raw native RGB, independent foreground, edge and exterior alpha, and native guard penalties'
                          if fit_opacity else 'raw native straight RGB and native regression guard penalties'),
                 center_max_local_delta=float(local_center_delta.max()),
                 max_abs_parameter_deltas={name: float(value.abs().max()) for name, value in delta.items()},
                 parameter_projection_hits={name: int(value) for name, value in projections.items()},
                 group_learning_rates=initial_rates, learning_rate_cycle_decay=.80,
                 local_opacity_optimized=fit_opacity, require_smaller_radius=require_smaller_radius,
                 convergence_source_objective_delta=1e-5, convergence_min_cycles=4)
    return SizeChildResult(optimized['xyz'], optimized['scaling'], optimized['opacity'], optimized['f_dc'], moments, audit,
                           child_rotation=child_rotation)
