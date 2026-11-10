"""Refine bound shape with balanced mask error and projected Gaussian support."""

from collections import Counter
from dataclasses import dataclass
import time
from types import SimpleNamespace

import numpy as np

from . import runtime as rt
from .coverage_refinement import _append, _clip, _reliable_regions, _surface_sites, coverage_config
from .surface_refinement import _face_pixels, _geometry_loss, _save


def shape_config():
    return dict(steps=1800, reseed_step=600, reset_steps=[], boundary_px=2, exterior_boundary_px=3,
                reliable_foreground=.99, reliable_background=.01, low_alpha=.65, required_source_bins=2,
                max_seed_fraction=.02, seed_opacity=.30, seed_tangent_area_factor=.5,
                site_barycentric=coverage_config()['site_barycentric'], foreground_target=.98,
                foreground_weight=.40, exterior_weight=.12, footprint_weight=.025,
                face_rgb_anchor=.40, face_alpha_anchor=.30, projection_near_px=24,
                footprint_sigma=1.5, mesh_occlusion_slack_m=.025, prune_bg_bins=3,
                normal_allowance_m=.022, tangent_allowance_m=.016, normal_weight=.018, tangent_weight=.010,
                position_lr=.00115, position_final_lr=.0002, scaling_lr=.0011, rotation_lr=.0004,
                feature_lr=.00085, opacity_lr=.014, geometry=dict(scale_weight=.04, axial_weight=.022,
                    major_sigma_over_extent=.008, axial_ratio=4., offset_limit=0., offset_weight=0.))


@dataclass
class View:
    rgb: object
    mask: object
    box: tuple
    inside: object
    outside: object
    distance: object
    mesh_depth: object
    triangle: object
    face: object
    face_rgb: object
    face_alpha: object


def _balanced_alpha(alpha, inside, outside, options):
    area = inside.count_nonzero().clamp_min(1)
    missing = (options['foreground_target']-alpha[inside]).relu()
    fn = (missing.square() + .20*missing).sum()/area
    fp = alpha[outside].square().sum()/area
    return options['foreground_weight']*fn + options['exterior_weight']*fp


def _tangent_distance(points, triangle):
    edge = triangle.roll(-1, dims=1)-triangle
    displacement = points[:, None, :]-triangle
    fraction = ((edge*displacement).sum(-1)/edge.square().sum(-1).clamp_min(1e-10)).clamp(0, 1)
    distance = (displacement-fraction[..., None]*edge).norm(dim=-1).min(dim=1).values
    cross = edge[..., 0]*displacement[..., 1]-edge[..., 1]*displacement[..., 0]
    inside = (cross >= 0).all(dim=1) | (cross <= 0).all(dim=1)
    return distance.masked_fill(inside, 0)


def _projection(xyz, K, matrix):
    camera_xyz = xyz @ matrix[:3, :3].T + matrix[:3, 3]
    homogeneous = camera_xyz @ K.T
    return homogeneous[:, :2]/homogeneous[:, 2:3].clamp_min(.01), homogeneous[:, 2]


def _screen_visibility(xy, depth, view, options):
    height, width = view.mesh_depth.shape
    x, y = xy.detach().floor().long().unbind(-1)
    valid = (depth.detach() > .2) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y = x.clamp(0, width-1), y.clamp(0, height-1)
    visible = valid & (depth.detach() <= view.mesh_depth[y, x] + options['mesh_occlusion_slack_m'])
    return visible, x, y


def _support(model, views, slots, bins, matrices, options):
    import torch

    votes = torch.zeros((2, len(model.binding)), device=model.binding.device, dtype=torch.int16)
    for group in sorted(set(bins)):
        supported = torch.zeros_like(votes, dtype=torch.bool)
        for i, view in enumerate(views):
            if bins[i] != group:
                continue
            model.select_mesh_by_timestep(slots[i])
            xy, depth = _projection(model.get_xyz, *matrices[i])
            visible, x, y = _screen_visibility(xy, depth, view, options)
            supported[0] |= visible & view.inside[y, x]
            supported[1] |= visible & view.outside[y, x]
        votes += supported
    return votes.unbind(0)


def _footprints(model, visible, view, K, matrix, options):
    import torch
    import torch.nn.functional as functional
    from utils.general_utils import build_rotation

    xy, depth = _projection(model.get_xyz, K, matrix)
    supported, x, y = _screen_visibility(xy, depth, view, options)
    indices = torch.where(visible & supported & (view.distance[y, x] > -options['projection_near_px']))[0]
    if not len(indices):
        return xy.new_zeros(())
    xy, depth = xy[indices], depth[indices]
    jacobian = (K[:2]-xy[..., None]*K[2])/depth[:, None, None]
    axes = build_rotation(model.get_rotation[indices])*model.get_scaling[indices, None, :]
    projected = jacobian @ matrix[:3, :3] @ axes
    u, v = projected.unbind(1)
    sx = (u.square().sum(1)+.3).sqrt()
    syx = (u*v).sum(1)/sx
    sy = (v.square().sum(1)+.3-syx.square()).clamp_min(1e-6).sqrt()
    axis_a = torch.stack((sx, syx), 1)
    axis_b = torch.stack((torch.zeros_like(sy), sy), 1)
    offsets = torch.stack((axis_a, -axis_a, axis_b, -axis_b), 1)
    samples = xy[:, None]+options['footprint_sigma']*offsets
    height, width = view.distance.shape
    grid = 2*samples/xy.new_tensor([width, height])-1
    # Only the reliable exterior contributes; detach opacity so the footprint
    # term steers centers and covariance even when reducing opacity is easier.
    field = (view.distance.float()-options['exterior_boundary_px']).relu()*view.outside
    outside = functional.grid_sample(field[None, None], grid[None], padding_mode='border',
                                     align_corners=False)[0, 0]
    return ((outside/8).square().mean(1)*model.get_opacity[indices, 0].detach()).mean()


def _shape_loss(model, visible, view, bad, extent, options):
    owners = model.binding[visible]
    local = model._xyz[visible]
    scale = model.face_scaling[owners, 0]
    normal = local[:, 1].abs()*scale
    tangent = _tangent_distance(local[:, [0, 2]], view.triangle[owners])*scale
    # A metric slack allows genuine clothing thickness. Only unsupported
    # centers receive the stronger pull toward their actual bound triangle.
    weight = 1+2*bad[visible]
    normal = ((normal-options['normal_allowance_m']-.20*scale)/.020).relu()
    tangent = ((tangent-options['tangent_allowance_m']-.40*scale)/.025).relu()
    return (_geometry_loss(local, model.get_scaling[visible], extent, options['geometry'])
            + (weight*(options['normal_weight']*normal.square()+options['tangent_weight']*tangent.square())).mean())


def _prepare_views(model, rows, slots, cameras, fit, pipe, black, options, *, native_targets=None, capture_face_anchor=True):
    import nvdiffrast.torch as dr
    from scipy.ndimage import distance_transform_edt
    import torch
    from gaussian_renderer import render
    from .training import _native_foreground_target

    ctx = dr.RasterizeCudaContext(device=model.binding.device)
    faces = model.faces.int().contiguous()
    views, matrices = [], []
    for i, (row, slot) in enumerate(zip(rows, slots, strict=True)):
        rgb, mask, box = _native_foreground_target(row) if native_targets is None else native_targets[i]
        face = _face_pixels(fit, row, slot, mask, box)
        hard = (mask[0] > options['reliable_foreground']).cpu().numpy()
        distance = distance_transform_edt(~hard)-distance_transform_edt(hard)
        distance = torch.from_numpy(np.clip(distance, -64, 64).astype(np.float16)).cuda()
        mask = mask.cuda()
        inside, outside = _reliable_regions(mask, options)
        if not inside.any() or not outside.any():
            raise ValueError(f'Missing reliable foreground or background: {row["frame"]}')
        model.select_mesh_by_timestep(slot)
        width, height = row['wh']
        vertices = model.vertices_sequence[slot]
        K, matrix = vertices.new_tensor(row['K']), vertices.new_tensor(row['w2c'])
        projected = (vertices @ matrix[:3, :3].T + matrix[:3, 3]) @ K.T
        raster, _ = dr.rasterize(ctx, _clip(projected, width, height)[None].contiguous(),
                                 faces, (height, width), grad_db=False)
        raster = raster[0].flip(0)
        depth = .0200002/(1.00002-raster[..., 2]).clamp_min(1e-6)
        depth = depth.masked_fill(raster[..., 3] == 0, float('inf')).half()
        triangle = vertices[model.faces]-model.face_center[:, None, :]
        triangle = (triangle @ model.face_orien_mat)/model.face_scaling[:, None, :]
        base_rgb = base_alpha = None
        if face is not None:
            face = torch.as_tensor(face, device=model.binding.device)
            if capture_face_anchor:
                x0, y0, _, _ = box
                y, x = face[0]+y0, face[1]+x0
                base_rgb = render(cameras[i], model, pipe, black)['render'][:, y, x].clone()
                base_alpha = render(cameras[i], model, pipe, black, override_color=torch.ones_like(model._xyz))['render'][0, y, x].clone()
        matrices.append((K, matrix))
        views.append(View(rgb.cuda(), mask, box, inside, outside, distance, depth, triangle[..., [0, 2]],
                          face, base_rgb, base_alpha))
    return views, matrices


def _reseed(model, rows, cameras, views, slots, bins, matrices, pipe, black, options, reference, *,
            preserve_optimizer=False, confirm_prune=None):
    import torch

    foreground, background = _support(model, views, slots, bins, matrices, options)
    model.select_mesh_by_timestep(reference)
    face_scale = model.face_scaling[model.binding, 0]
    normal = model._xyz[:, 1].abs()*face_scale
    triangles = views[slots.index(reference)].triangle[model.binding]
    tangent = _tangent_distance(model._xyz[:, [0, 2]], triangles)*face_scale
    isolated = (foreground == 0) & (background >= options['prune_bg_bins'])
    off_surface = (normal > options['normal_allowance_m']+.20*face_scale) | (tangent > .028+.40*face_scale)
    before = len(model.binding)
    prune = isolated & off_surface
    model.prune_points(prune if confirm_prune is None else confirm_prune(prune))
    pruned = before-len(model.binding)
    targets = [(v.rgb, v.mask, v.box, v.inside, v.outside, v.face) for v in views]
    owners, bary, color, evidence = _surface_sites(model, rows, cameras, targets, slots, bins, pipe, black, options)
    if not preserve_optimizer:
        model.optimizer = None
    _append(model, owners, bary, color, options, reference, preserve_optimizer=preserve_optimizer)
    if int(model.binding_counter.min()) < 1 or len(model.binding) != before-pruned+len(owners):
        raise RuntimeError('Shape refinement changed the required triangle support.')
    return dict(evidence, pruned=pruned, isolated_off_surface_candidates=int((isolated & off_surface).sum()))


def refine(root, baseline_snapshot, model, effective):
    """Polish geometry and cover verified gaps with the fitted meshes unchanged."""
    import torch
    from gaussian_renderer import render
    from .stable_selection import source_bin_training_order
    from .training import camera, foreground_loss, named_pose_layout, render_pose, verify_official_fit

    options = shape_config()
    if model.optimizer is not None or model.active_sh_degree != 0:
        raise ValueError('Shape refinement requires a frozen bound SH0 snapshot.')
    verify_official_fit(root)
    selection = rt.read(root/'fit/official-fit-result.json')['selection']
    rows = rt.read(root/'data/training-inputs.json')['records']
    with np.load(root/'fit/fitted.npz', allow_pickle=False) as archive:
        fit = dict(archive)
    layout = named_pose_layout(fit, rows)
    slots, reference = layout['pose_indices'], layout['reference_pose_index']
    cameras = [camera(row, i, pose_index=slot)[0] for i, (row, slot) in enumerate(zip(rows, slots, strict=True))]
    bins, order = source_bin_training_order(rows, selection, seed=effective['seed'])
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    black = torch.zeros(3, device='cuda')
    posed, topology = model.vertices_sequence.clone(), model.faces.clone()
    model.binding = model.binding.long()
    initial = len(model.binding)
    started = time.monotonic()
    out = root/'gaussian-run'
    with torch.no_grad():
        views, matrices = _prepare_views(model, rows, slots, cameras, fit, pipe, black, options)
        foreground, background = _support(model, views, slots, bins, matrices, options)
    bad = (background >= 2) & (background > 2*foreground)
    fields = ('_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation')
    for name in fields:
        getattr(model, name).requires_grad_(True)
    model.spatial_lr_scale = 1.
    args = dict(effective['optimization'], position_lr_init=options['position_lr'],
                position_lr_final=options['position_final_lr'], position_lr_max_steps=options['steps'],
                **{name: options[name] for name in ('scaling_lr', 'rotation_lr', 'feature_lr', 'opacity_lr')})
    optimizer_args = SimpleNamespace(**args)
    model.max_radii2D = torch.zeros(len(model.binding), device='cuda')
    model.training_setup(optimizer_args)
    extent = float(np.linalg.norm(np.ptp(fit['vertices_world'][reference], axis=0)))
    draws, evidence = Counter(), None
    for step in range(1, options['steps']+1):
        model.update_learning_rate(step)
        i = next(order)
        draws[str(bins[i])] += 1
        view = views[i]
        pkg = render_pose(render, cameras[i], model, pipe, black, slots[i])
        rgb = pkg['render']
        alpha = render(cameras[i], model, pipe, black, override_color=torch.ones_like(model._xyz))['render'][0]
        x0, y0, x1, y1 = view.box
        crop = (..., slice(y0, y1), slice(x0, x1))
        photo, _, _ = foreground_loss(rgb[crop]/alpha[crop].clamp_min(effective['straight_rgb_alpha_floor']),
                                      view.rgb, view.mask[crop])
        visible = pkg['visibility_filter']
        loss = photo + _balanced_alpha(alpha, view.inside, view.outside, options)
        loss += _shape_loss(model, visible, view, bad, extent, options)
        loss += options['footprint_weight']*_footprints(model, visible, view, *matrices[i], options)
        if view.face is not None:
            y, x = view.face[0]+y0, view.face[1]+x0
            loss += options['face_rgb_anchor']*(rgb[:, y, x]-view.face_rgb).abs().mean()
            loss += options['face_alpha_anchor']*(alpha[y, x]-view.face_alpha).abs().mean()
        loss.backward()
        with torch.no_grad():
            if step == 1 or step % 100 == 0:
                if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all()
                        for p in (getattr(model, name) for name in fields if name != '_features_rest')):
                    raise RuntimeError('Nonfinite shape refinement loss or gradient.')
                rt.save(out/'shape-refinement-progress.json', dict(step=step, steps=options['steps'],
                    points=len(model.binding), seconds=time.monotonic()-started, loss=float(loss)))
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)
            if step == options['reseed_step']:
                evidence = _reseed(model, rows, cameras, views, slots, bins, matrices, pipe, black, options, reference)
                rt.save(out/'shape-coverage-support.json', evidence)
                foreground, background = _support(model, views, slots, bins, matrices, options)
                bad = (background >= 2) & (background > 2*foreground)
                model.max_radii2D = torch.zeros(len(model.binding), device='cuda')
                model.training_setup(optimizer_args)
    model.optimizer = None
    for name in fields:
        getattr(model, name).requires_grad_(False)
    if not torch.equal(model.vertices_sequence, posed) or not torch.equal(model.faces, topology):
        raise RuntimeError('Shape refinement changed the frozen mesh or poses.')
    snapshot = out/f'{baseline_snapshot.name}-shape-{options["steps"]:06d}'
    replay = _save(root, baseline_snapshot, snapshot, model, reference, cameras[slots.index(reference)], pipe, black)
    return replay, dict(state='completed', steps=options['steps'], snapshot=snapshot.name,
        baseline_snapshot=baseline_snapshot.name, initial_points=initial, points=len(model.binding),
        frozen_mesh_verified=True, source_bin_draws=dict(draws), options=options,
        coverage_support=evidence, face_anchor_views=sum(v.face is not None for v in views), seconds=time.monotonic()-started)
