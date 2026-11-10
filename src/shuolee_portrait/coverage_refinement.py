"""Seed bound Gaussians at sites undercovered in distinct source bins."""

from collections import Counter
import time
from types import SimpleNamespace

import numpy as np

from . import runtime as rt
from .head_appearance import face_composite_loss
from .surface_refinement import _face_pixels, _geometry_loss, _save


def coverage_config():
    return dict(steps=1500, reset_steps=[], boundary_px=2, exterior_boundary_px=3,
                reliable_foreground=.99, reliable_background=.01, low_alpha=.65, required_source_bins=2,
                max_seed_fraction=.035, seed_opacity=.35, seed_tangent_area_factor=.55,
                site_barycentric=[[1/3, 1/3, 1/3], [.6, .2, .2], [.2, .6, .2], [.2, .2, .6]],
                foreground_alpha_target=.98, foreground_alpha_weight=.30, exterior_alpha_weight=.055,
                face_composite_weight=.08, face_alpha_weight=.12, position_lr=.0014, position_final_lr=.00025,
                scaling_lr=.0012, rotation_lr=.0004, feature_lr=.0010, opacity_lr=.018,
                geometry=dict(scale_weight=.035, axial_weight=.02, offset_weight=.02,
                              major_sigma_over_extent=.008, axial_ratio=4., offset_limit=.28))


def _reliable_regions(mask, options):
    from torch.nn.functional import max_pool2d

    inner, outer = options['boundary_px'], options['exterior_boundary_px']
    foreground = max_pool2d((mask <= options['reliable_foreground']).float(), 2*inner+1, 1, inner) == 0
    exterior = max_pool2d((mask >= options['reliable_background']).float(), 2*outer+1, 1, outer) == 0
    return foreground[0], exterior[0]


def _region_alpha(alpha, inside, outside, options):
    foreground = (options['foreground_alpha_target'] - alpha[inside]).relu().square().mean()
    exterior = alpha[outside].square().mean()
    return options['foreground_alpha_weight']*foreground + options['exterior_alpha_weight']*exterior


def _clip(projected, width, height):
    import torch

    x, y, z = projected.unbind(-1)
    return torch.stack((2*x/width-z, z-2*y/height, 1.00002*z-.0200002, z), dim=-1)


def _select_sites(votes, veto, deficits, counts, options, budget):
    import torch

    eligible = (votes >= options['required_source_bins']) & ~veto
    score = (deficits / counts.clamp_min(1)).masked_fill(~eligible, -1)
    per_face, best = score.max(dim=1)
    candidates = torch.where(per_face > 0)[0]
    count = min(len(candidates), budget)
    owners = candidates[per_face[candidates].topk(count).indices]
    return owners, best[owners]


def _surface_sites(model, rows, cameras, targets, slots, bins, pipe, black, options):
    import nvdiffrast.torch as dr
    import torch
    from gaussian_renderer import render

    device = model.binding.device
    raster_context = dr.RasterizeCudaContext(device=device)
    faces = model.faces.int().contiguous()
    bary = torch.as_tensor(options['site_barycentric'], device=device, dtype=model._xyz.dtype)
    shape = (len(faces), len(bary))
    owners = torch.arange(len(faces), device=device)[:, None]
    votes = torch.zeros(shape, device=device, dtype=torch.int16)
    veto = torch.zeros(shape, device=device, dtype=torch.bool)
    counts = torch.zeros_like(votes)
    deficits = torch.zeros(shape, device=device)
    colors = torch.zeros((*shape, 3), device=device)
    for group in sorted(set(bins)):
        supported = torch.zeros_like(veto)
        for i, group_id in enumerate(bins):
            if group_id != group:
                continue
            row, slot = rows[i], slots[i]
            source, _, box, inside, outside, _ = targets[i]
            model.select_mesh_by_timestep(slot)
            width, height = row['wh']
            vertices = model.vertices_sequence[slot]
            matrix = vertices.new_tensor(row['w2c'])
            projected = (vertices @ matrix[:3, :3].T + matrix[:3, 3]) @ vertices.new_tensor(row['K']).T
            raster, _ = dr.rasterize(raster_context, _clip(projected, width, height)[None].contiguous(),
                                     faces, (height, width), grad_db=False)
            # Test the same barycentric locations in every pose against the nearest mesh triangle.
            samples = torch.einsum('sk,fkc->fsc', bary, projected[model.faces])
            depth = samples[..., 2]
            xy = (samples[..., :2] / depth[..., None].clamp_min(.01)).floor().long()
            x, y = xy.unbind(-1)
            valid = (depth > .2) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
            x, y = x.clamp(0, width-1), y.clamp(0, height-1)
            visible = valid & (raster[0, height-1-y, x, 3].long() == owners+1)
            veto |= visible & outside[y, x]
            alpha = render(cameras[i], model, pipe, black, override_color=torch.ones_like(model._xyz))['render'][0]
            opacity = alpha[y, x]
            low = visible & inside[y, x] & (opacity < options['low_alpha'])
            supported |= low
            counts += low
            deficits += (1-opacity)*low
            x0, y0, x1, y1 = box
            rgb = source[:, (y-y0).clamp(0, y1-y0-1), (x-x0).clamp(0, x1-x0-1)].permute(1, 2, 0)
            colors += rgb*low[..., None]
        votes += supported
    budget = int(len(model.binding)*options['max_seed_fraction'])
    chosen, site = _select_sites(votes, veto, deficits, counts, options, budget)
    color = colors[chosen, site]/counts[chosen, site].clamp_min(1)[:, None]
    facts = dict(sites=int(votes.numel()), source_bins=len(set(bins)), seed_budget=budget,
                 sites_with_required_bins=int((votes >= options['required_source_bins']).sum()),
                 foreground_background_conflicts=int(((votes >= options['required_source_bins']) & veto).sum()),
                 seeded_sites=len(chosen), seed_source_bin_counts=dict(Counter(map(str, votes[chosen, site].tolist()))))
    return chosen, bary[site], color, facts


def _append(model, owners, weights, color, options, reference):
    import torch
    from utils.sh_utils import RGB2SH

    model.select_mesh_by_timestep(reference)
    faces = model.vertices_sequence[reference][model.faces[owners]]
    world = (faces*weights[..., None]).sum(dim=1)
    shift = (world-model.face_center[owners])/model.face_scaling[owners]
    local = (model.face_orien_mat[owners].transpose(1, 2) @ shift[..., None]).squeeze(-1)
    area = .5*torch.linalg.cross(faces[:, 1]-faces[:, 0], faces[:, 2]-faces[:, 0], dim=1).norm(dim=1)
    sigma = (options['seed_tangent_area_factor']*area.sqrt()/model.face_scaling[owners, 0]).clamp(.035, .4)
    log_sigma = torch.stack((sigma, .35*sigma, sigma), dim=1).log()
    rotation = local.new_zeros((len(owners), 4))
    rotation[:, 0] = 1
    opacity = local.new_full((len(owners), 1), options['seed_opacity']).logit()
    rest = model._features_rest.new_zeros((len(owners), *model._features_rest.shape[1:]))
    added = dict(_xyz=local, _features_dc=RGB2SH(color)[:, None], _features_rest=rest,
                 _opacity=opacity, _scaling=log_sigma, _rotation=rotation)
    for name, values in added.items():
        setattr(model, name, torch.nn.Parameter(torch.cat((getattr(model, name).detach(), values), dim=0)))
    model.binding = torch.cat((model.binding.long(), owners))
    model.binding_counter.scatter_add_(0, owners, torch.ones_like(owners, dtype=torch.int32))
    if len(owners) and not torch.allclose(model.get_xyz[-len(owners):], world, atol=2e-6, rtol=0):
        raise RuntimeError('Coverage sites do not round-trip through the face-local transform.')
    return tuple(added)


def refine(root, baseline_snapshot, model, effective):
    """Refine a pinned terminal without editing its pose, camera or fitted mesh."""
    import torch
    from gaussian_renderer import render
    from .stable_selection import source_bin_training_order
    from .training import (_frozen_smpl_surface, _native_foreground_target, camera, foreground_loss,
                           named_pose_layout, render_pose, verify_official_fit)

    options = coverage_config()
    if model.optimizer is not None or model.active_sh_degree != 0:
        raise ValueError('Coverage refinement requires a frozen bound SH0 snapshot.')
    verify_official_fit(root)
    selection = rt.read(root/'fit/official-fit-result.json')['selection']
    rows = rt.read(root/'data/training-inputs.json')['records']
    with np.load(root/'fit/fitted.npz', allow_pickle=False) as archive:
        fit = dict(archive)
    layout = named_pose_layout(fit, rows)
    slots = layout['pose_indices']
    reference = layout['reference_pose_index']
    asset = rt.ASSETS/'smplx/SMPLX_NEUTRAL.npz'
    _, _, _, protected, _ = _frozen_smpl_surface(fit, asset)
    head = torch.as_tensor(protected[fit['faces']].all(axis=1), device=model.binding.device)
    frozen = head[model.binding.long()]
    initial = len(frozen)
    posed = model.vertices_sequence.clone()
    topology = model.faces.clone()
    cameras, targets = [], []
    for i, (row, slot) in enumerate(zip(rows, slots, strict=True)):
        cameras.append(camera(row, i, pose_index=slot)[0])
        rgb, mask, box = _native_foreground_target(row)
        face = _face_pixels(fit, row, slot, mask, box)
        mask = mask.cuda()
        inside, outside = _reliable_regions(mask, options)
        if not inside.any() or not outside.any():
            raise ValueError(f'Missing reliable foreground or background: {row["frame"]}')
        targets.append((rgb.cuda(), mask, box, inside, outside,
                        None if face is None else torch.as_tensor(face, device='cuda')))
    bins, order = source_bin_training_order(rows, selection, seed=effective['seed'])
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    black = torch.zeros(3, device='cuda')
    out = root/'gaussian-run'
    started = time.monotonic()
    with torch.no_grad():
        owners, bary, color, evidence = _surface_sites(model, rows, cameras, targets, slots, bins, pipe, black, options)
        fields = _append(model, owners, bary, color, options, reference)
        evidence['strict_head_seeds'] = int(head[owners].sum())
        rt.save(out/'coverage-support.json', evidence)
    frozen = torch.cat((frozen, torch.zeros(len(owners), device='cuda', dtype=torch.bool)))
    originals = {name: getattr(model, name)[frozen].detach().clone() for name in fields}
    start_xyz = model._xyz.detach().clone()
    args = dict(effective['optimization'], position_lr_init=options['position_lr'],
                position_lr_final=options['position_final_lr'], position_lr_max_steps=options['steps'],
                **{name: options[name] for name in ('scaling_lr', 'rotation_lr', 'feature_lr', 'opacity_lr')})
    model.spatial_lr_scale = 1.
    model.training_setup(SimpleNamespace(**args))
    extent = float(np.linalg.norm(np.ptp(fit['vertices_world'][reference], axis=0)))
    draws = Counter()
    for step in range(1, options['steps']+1):
        model.update_learning_rate(step)
        i = next(order)
        draws[str(bins[i])] += 1
        source, mask, box, inside, outside, face = targets[i]
        rgb_pkg = render_pose(render, cameras[i], model, pipe, black, slots[i])
        rgb = rgb_pkg['render']
        alpha = render(cameras[i], model, pipe, black, override_color=torch.ones_like(model._xyz))['render'][:1]
        x0, y0, x1, y1 = box
        region = (..., slice(y0, y1), slice(x0, x1))
        straight = rgb[region]/alpha[region].clamp_min(effective['straight_rgb_alpha_floor'])
        photo, _, _ = foreground_loss(straight, source, mask[region])
        visible = rgb_pkg['visibility_filter'] & ~frozen
        loss = photo + _region_alpha(alpha[0], inside, outside, options)
        loss += _geometry_loss((model._xyz-start_xyz)[visible], model.get_scaling[visible], extent, options['geometry'])
        if face is not None:
            loss += options['face_composite_weight']*face_composite_loss(rgb, alpha, source, box, face)
            deficit = (options['foreground_alpha_target']-alpha[0, face[0]+y0, face[1]+x0]).relu().mean()
            loss += options['face_alpha_weight']*deficit
        loss.backward()
        with torch.no_grad():
            gradients = [getattr(model, name).grad for name in fields if name != '_features_rest']
            for gradient in gradients:
                gradient[frozen] = 0
            if step == 1 or step % 100 == 0:
                if not torch.isfinite(loss) or any(not torch.isfinite(g).all() for g in gradients):
                    raise RuntimeError('Nonfinite coverage refinement loss or gradient.')
                rt.save(out/'coverage-refinement-progress.json', dict(step=step, steps=options['steps'],
                    points=len(model.binding), seconds=time.monotonic()-started, loss=float(loss)))
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)
    model.optimizer = None
    for name in fields:
        getattr(model, name).requires_grad_(False)
    if (not torch.equal(model.faces, topology) or not torch.equal(model.vertices_sequence, posed)
            or any(not torch.equal(getattr(model, name)[frozen].contiguous().view(torch.uint8),
                                   value.contiguous().view(torch.uint8)) for name, value in originals.items())):
        raise RuntimeError('Coverage refinement changed the frozen mesh or prior strict-head Gaussians.')
    snapshot = out/f'{baseline_snapshot.name}-coverage-{options["steps"]:06d}'
    replay = _save(root, baseline_snapshot, snapshot, model, reference, cameras[slots.index(reference)], pipe, black)
    return replay, dict(state='completed', steps=options['steps'], snapshot=snapshot.name,
        baseline_snapshot=baseline_snapshot.name, initial_points=initial, points=len(model.binding),
        frozen_mesh_verified=True, frozen_strict_head_points=int(frozen.sum()), source_bin_draws=dict(draws),
        options=options, coverage_support=evidence, seconds=time.monotonic()-started)
