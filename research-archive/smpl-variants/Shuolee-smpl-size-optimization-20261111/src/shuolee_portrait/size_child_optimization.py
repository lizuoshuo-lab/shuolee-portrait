"""Fit bounded child rows against native observations without modifying the live scene."""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from torch.nn import functional as F

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


def _optimization_views(weights, bins, limit):
    """Choose supported views from distinct angular bins before adding repeated bins."""
    quality = weights[..., 0].clamp_min(0).log1p().sum(1).tolist()
    order = sorted(range(len(bins)), key=lambda i: (-quality[i], i))
    chosen, used = [], set()
    for i in order:
        if quality[i] > 0 and bins[i] not in used:
            chosen.append(i)
            used.add(bins[i])
    chosen.extend(i for i in order if quality[i] > 0 and i not in chosen)
    return chosen[:limit]


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
    rgb_weight: torch.Tensor
    alpha_weight: torch.Tensor
    parent_radii: torch.Tensor
    parent_rgb: torch.Tensor
    parent_alpha: torch.Tensor
    representative: tuple | None


@torch.no_grad()
def _fit_view(render, model, constraints, i, parents, owners, keep, children):
    view = constraints.views[i]
    model.select_mesh_by_timestep(constraints.slots[i])
    parent = _model_scene(model)
    rgb, alpha, radii = _draw(render, constraints, i, parent)
    parent_radii = radii[parents]
    height, width = view.inside.shape
    boxes, valid = _boxes(parent.get_xyz[parents], parent_radii, (0, 0, width, height),
                          constraints.matrices[i], pad=8)
    boxes = boxes[valid].tolist()
    if not boxes:
        return None
    left, top, right, bottom = map(list, zip(*boxes))
    x0, y0, x1, y1 = min(left), min(top), max(right), max(bottom)
    crop = (slice(y0, y1), slice(x0, x1))
    union = torch.zeros_like(view.inside[crop])
    for left, top, right, bottom in boxes:
        union[top-y0:bottom-y0, left-x0:right-x0] = True
    inside = view.inside[crop] & union
    outside = view.outside[crop] & union
    if not bool(inside.any()):
        return None
    target_alpha = _target_alpha(view)[crop]
    source = _target_rgb(view)
    target_rgb = source.new_zeros(3, y1-y0, x1-x0)
    vx, vy, vr, vb = view.box
    left, top, right, bottom = max(x0, vx), max(y0, vy), min(x1, vr), min(y1, vb)
    target_rgb[:, top-y0:bottom-y0, left-x0:right-x0] = source[:, top-vy:bottom-vy, left-vx:right-vx]
    rgb_weight = inside.float()/inside.count_nonzero()
    alpha_weight = rgb_weight + .5*outside.float()/outside.count_nonzero().clamp_min(1)
    fixed = SimpleNamespace(**vars(parent))
    for key in ('get_xyz', 'get_scaling', 'get_rotation', 'get_opacity', 'get_features'):
        setattr(fixed, key, getattr(parent, key)[keep])
    # Record an unrescaled native crop around one measured offending footprint.
    representative = boxes[int(parent_radii[valid].argmax())]
    return _FitView(i, fixed, model.face_center[owners], model.face_orien_mat[owners], model.face_scaling[owners],
                    parent.get_rotation[parents].repeat(children, 1),
                    parent.get_features[parents, 1:].repeat(children, 1, 1),
                    crop, target_rgb, target_alpha, rgb_weight, alpha_weight, parent_radii,
                    rgb[:, crop[0], crop[1]].contiguous(), alpha[crop].contiguous(), representative)


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
    straight = rgb/alpha.clamp_min(1e-3)
    error = (straight-fit.target_rgb).abs().mean(0)
    # The premultiplied component still supplies a usable geometry gradient in holes.
    premultiplied = (rgb-fit.target_rgb*fit.target_alpha).abs().mean(0)
    alpha_error = (alpha-fit.target_alpha).abs()
    return (error*fit.rgb_weight).sum(), (alpha_error*fit.alpha_weight).sum(), (premultiplied*fit.rgb_weight).sum()


@torch.no_grad()
def _observe(render, constraints, fit, params):
    rgb, alpha, radii = _draw(render, constraints, fit.index, _candidate(fit, params))
    y, x = fit.crop
    rgb, alpha = rgb[:, y, x].contiguous(), alpha[y, x].contiguous()
    rgb_error, alpha_error, _ = _loss(fit, rgb, alpha)
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
    center_lr = float(parent_width.median()) * .028
    rates = dict(xyz=center_lr, scaling=.035, opacity=.15, f_dc=.045)
    return [dict(name=name, params=[param], lr=rates[name], betas=source[name]['betas'],
                 eps=source[name]['eps']) for name, param in params.items()]


@torch.no_grad()
def optimize_size_children(model, constraints, parents, centers, log_sigma, opacity_logits, dc, upper_log_sigma,
                           weights, *, max_steps=24, max_views=6, diagnostic_dir=None):
    """Run an isolated bounded Adam fit, returning parent-major rows and raw Adam moments."""
    from gaussian_renderer import render
    from utils.sh_utils import C0
    from .sh0_main import DCColorProjection

    start = perf_counter()
    if centers.ndim != 3 or centers.shape[0] != len(parents) or centers.shape[2] != 3:
        raise ValueError('Expected parent-major centers with three local coordinates.')
    n, k, _ = centers.shape
    if (not n or k < 2 or log_sigma.shape != (n, k, 3) or opacity_logits.shape != (n, k, 1)
            or dc.shape != (n, k, 1, 3) or weights.shape != (len(constraints.views), n, 3)):
        raise ValueError('Inconsistent Gaussian replacement parameters or source weights.')
    if not (0 < max_steps <= 24 and 0 < max_views <= 6) or model.active_sh_degree != 0 or model.max_sh_degree != 0:
        raise ValueError('Size-child fitting requires bounded updates, bounded views and the SH0 pipeline.')
    cap = upper_log_sigma[:, None] if upper_log_sigma.ndim == 2 else upper_log_sigma
    cap = torch.broadcast_to(cap, log_sigma.shape)
    if not all(bool(torch.isfinite(value).all()) for value in (centers, log_sigma, opacity_logits, dc, weights)):
        raise ValueError('Candidate parameters and source weights must be finite.')
    if bool(torch.isnan(cap).any()) or bool(torch.isneginf(cap).any()):
        raise ValueError('Invalid child scale caps.')

    parent_width = model._scaling[parents].detach().amax(1).exp()
    keep = torch.ones(len(model.binding), dtype=torch.bool, device=parents.device)
    keep[parents] = False
    owners = model.binding[parents].repeat(k)
    initial = dict(xyz=centers.detach(), scaling=log_sigma.detach(), opacity=opacity_logits.detach(), f_dc=dc.detach())
    params = {name: torch.nn.Parameter(value.clone()) for name, value in initial.items()}
    adam_groups = _adam_groups(model, params, parent_width)
    optimizer = torch.optim.Adam(adam_groups)
    dc_cap = DCColorProjection(model._features_dc, C0).bound
    # A child can move at most a quarter parent sigma on each local axis.
    displacement = .25*parent_width[:, None, None]
    lower_scale = torch.minimum(initial['scaling']-.55, cap)
    opacity_bounds = opacity_logits.new_tensor([.005, .995]).logit().tolist()
    bounds = dict(xyz=(initial['xyz']-displacement, initial['xyz']+displacement),
                  scaling=(lower_scale, cap),
                  opacity=tuple(opacity_logits.new_tensor(value) for value in opacity_bounds),
                  f_dc=(dc.new_tensor(-dc_cap), dc.new_tensor(dc_cap)))
    projections = _project_params(params, optimizer, bounds)
    proposal = {name: value.detach().clone() for name, value in params.items()}
    chosen = _optimization_views(weights, constraints.bins, max_views)
    with _preserve_pose(model):
        fits = [_fit_view(render, model, constraints, i, parents, owners, keep, k) for i in chosen]
    fits = [fit for fit in fits if fit is not None]
    updates = 0
    losses = []
    invalid = None
    for step in range(max_steps if fits else 0):
        fit = fits[step % len(fits)]
        optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad():
            rgb, alpha, _ = _draw(render, constraints, fit.index, _candidate(fit, params))
            y, x = fit.crop
            color, opacity_error, premultiplied = _loss(fit, rgb[:, y, x], alpha[y, x])
            # Favor raw photometry while giving independent coverage a direct gradient.
            objective = .30*color + .70*premultiplied + opacity_error
            objective.backward()
        valid = torch.stack([torch.isfinite(value.grad).all() for value in params.values()]).all()
        if not bool(valid & torch.isfinite(objective)):
            invalid = 'nonfinite_candidate_objective_or_gradient'
            break
        optimizer.step()
        for name, hit in _project_params(params, optimizer, bounds).items():
            projections[name] += hit
        losses.append(float(objective))
        updates += 1

    optimized = {name: value.detach() for name, value in params.items()}
    moments = {}
    for group in adam_groups:
        name = group['name']
        state = optimizer.state[params[name]]
        moments[name] = dict(step=updates, betas=group['betas'],
                             exp_avg=state.get('exp_avg', torch.zeros_like(params[name])).detach(),
                             exp_avg_sq=state.get('exp_avg_sq', torch.zeros_like(params[name])).detach())

    frames, representatives = [], []
    delta = {name: optimized[name]-proposal[name] for name in params}
    local_center_delta = delta['xyz'].norm(dim=-1)
    for fit in fits:
        row = constraints.rows[fit.index]
        before, before_metrics = _observe(render, constraints, fit, proposal)
        after, after_metrics = _observe(render, constraints, fit, optimized)
        parent_color, parent_alpha, _ = _loss(fit, fit.parent_rgb, fit.parent_alpha)
        center_delta_m = local_center_delta*fit.face_scale.reshape(k, n).T
        frames.append(dict(frame=row['frame'], view_index=fit.index, angular_bin=constraints.bins[fit.index],
                           parent=dict(rgb_mae=float(parent_color), alpha_error=float(parent_alpha),
                                       radii_px=fit.parent_radii.tolist()),
                           proposal=before_metrics, optimized=after_metrics,
                           max_optimized_center_move_m=float(center_delta_m.max())))
        if diagnostic_dir is not None and len(representatives) < 2:
            representatives.append(_save_representative(diagnostic_dir, fit, row, before, after))

    audit = dict(status=invalid or ('fitted' if updates else 'no_supported_optimization_view'),
                 parent_rows=parents.tolist(), children_per_parent=k, optimizer_updates=updates,
                 optimizer_update_budget=max_steps, optimizer_view_budget=max_views,
                 seconds=perf_counter()-start, views=frames, representatives=representatives,
                 targets='raw native RGB and independent source alpha inside measured parent footprint',
                 center_max_local_delta=float(local_center_delta.max()),
                 max_abs_parameter_deltas={name: float(value.abs().max()) for name, value in delta.items()},
                 parameter_projection_hits={name: int(value) for name, value in projections.items()},
                 group_learning_rates={group['name']: group['lr'] for group in adam_groups},
                 first_update_objective=losses[0] if losses else None, last_update_objective=losses[-1] if losses else None)
    return SizeChildResult(optimized['xyz'], optimized['scaling'], optimized['opacity'], optimized['f_dc'], moments, audit)
