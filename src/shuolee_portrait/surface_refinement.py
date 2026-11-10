"""Refine terminal bound SH0 Gaussians on the existing named fit."""

from collections import Counter
import shutil
import time
from types import SimpleNamespace

import numpy as np

from . import runtime as rt
from .head_appearance import face_composite_loss, face_interior_pixels


def surface_config():
    return dict(steps=1800, split_steps=[450, 900], reset_steps=[],
                position_lr=.0015, position_final_lr=.00025, scaling_lr=.0015,
                rotation_lr=.0005, feature_lr=.00125, opacity_lr=.02,
                face_composite_weight=.10, face_alpha_weight=.18, exterior_weight=.025,
                scale_weight=.06, axial_weight=.03, offset_weight=.02,
                major_sigma_over_extent=.008, axial_ratio=4., offset_limit=1.15,
                minimum_screen_gradient=.00005, split_fraction=.0125, min_opacity=.005,
                interior_alpha_target=.98, mask_boundary_px=2, exterior_radius_px=24)


def _face_pixels(fit, row, slot, mask, box):
    observed = fit['observed_native_keypoints'][slot, 67:118].copy()
    projected = fit['joints_camera'][slot, 67:118] @ np.asarray(row['K_integer_centers']).T
    valid = projected[:, 2] > 0
    projected[valid, :2] /= projected[valid, 2:3]
    valid &= np.linalg.norm(projected[:, :2] - observed[:, :2], axis=1) <= 6.
    observed[~valid, 2] = 0.
    return face_interior_pixels(observed, mask, box)


def _exterior_pixels(mask, box, options):
    from scipy.ndimage import distance_transform_edt

    width, height = mask.shape[2], mask.shape[1]
    x0, y0, x1, y1 = box
    radius = options['exterior_radius_px']
    left, top = max(0, x0-radius), max(0, y0-radius)
    region = mask[0, top:min(height, y1+radius), left:min(width, x1+radius)].numpy()
    distance = distance_transform_edt(region <= .99)
    y, x = np.nonzero((region < .01) & (distance > options['mask_boundary_px']) & (distance <= radius))
    return np.stack((y+top, x+left))


def _geometry_loss(local_xyz, world_sigma, extent, options):
    # A thin third axis can represent a surface. Only penalize a long first
    # axis relative to the second, without inflating the smaller axes.
    major, second = world_sigma.sort(dim=1, descending=True).values[:, :2].unbind(1)
    size = (major / (extent * options['major_sigma_over_extent'])).log().relu()
    axial = (major / (second.detach() * options['axial_ratio'])).log().relu()
    offset = (local_xyz.norm(dim=1) - options['offset_limit']).relu()
    return (options['scale_weight']*size.square().mean() + options['axial_weight']*axial.square().mean()
            + options['offset_weight']*offset.square().mean())


def _split_candidates(score, head, options):
    import torch

    chosen = torch.zeros_like(head)
    for region in (head, ~head):
        pool = torch.where(region & (score >= options['minimum_screen_gradient']))[0]
        budget = min(len(pool), int(int(region.sum()) * options['split_fraction']))
        if budget:
            chosen[pool[score[pool].topk(budget).indices]] = True
    return chosen


def _save(root, source, snapshot, model, reference, camera, pipe, black):
    import torch
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from .bound_replay import save_bound_snapshot
    from .training import image, reference_ply_comparison, render_pose

    snapshot.mkdir(exist_ok=False)
    mesh = snapshot/'reference-mesh.npz'
    shutil.copyfile(source/mesh.name, mesh)
    mesh_sha = rt.sha(mesh)
    if mesh_sha != rt.sha(source/mesh.name):
        raise RuntimeError('Surface refinement changed the saved mesh.')
    with torch.no_grad():
        world = snapshot/'world.ply'
        model.export_world_ply(world, reference)
        reloaded = GaussianModel(0)
        reloaded.load_ply(str(world))
        _, delta = reference_ply_comparison(render, camera, model, reloaded, pipe, black, reference)
        parity = dict(mean_abs=float(delta.mean()), max_abs=float(delta.max()))
        if not (parity['mean_abs'] < 2e-6 and parity['max_abs'] < .005):
            raise RuntimeError(f'Surface refinement PLY readback differs: {parity}')
        rt.save(snapshot/'native-ply-parity.json', dict(passed=True, checks=[parity], sha256=rt.sha(world),
            mesh_sha256=mesh_sha, reference_frame_id=rt.read(source/'native-ply-parity.json')['reference_frame_id'],
            reference_pose_index=reference))
        replay = save_bound_snapshot(root, snapshot, model)
        image(render_pose(render, camera, replay[0], pipe, black, reference)['render']).save(snapshot/'reference-render.png')
    return replay


def refine(root, baseline_snapshot, model, effective):
    """Optimize shared local Gaussians, leaving all pose and mesh state frozen."""
    import torch
    from gaussian_renderer import render
    from .stable_selection import source_bin_training_order
    from .training import (_frozen_smpl_surface, _native_foreground_target, _sh0_foreground_alpha_loss,
                           camera, named_pose_layout, render_pose, verify_official_fit)

    options = surface_config()
    if model.optimizer is not None or model.active_sh_degree != 0:
        raise ValueError('Surface refinement requires a frozen SH0 baseline and an empty optimizer.')
    verify_official_fit(root)
    selection = rt.read(root/'fit/official-fit-result.json')['selection']
    with np.load(root/'fit/fitted.npz', allow_pickle=False) as archive:
        fit = dict(archive)
    rows = rt.read(root/'data/training-inputs.json')['records']
    layout = named_pose_layout(fit, rows)
    reference = layout['reference_pose_index']
    ref_index = layout['pose_indices'].index(reference)
    asset = rt.ASSETS/'smplx/SMPLX_NEUTRAL.npz'
    _, _, _, protected, _ = _frozen_smpl_surface(fit, asset)
    head_faces = torch.as_tensor(protected[fit['faces']].all(axis=1), device=model.binding.device)
    cameras, targets = [], []
    for i, (row, slot) in enumerate(zip(rows, layout['pose_indices'], strict=True)):
        cameras.append(camera(row, i, pose_index=slot)[0])
        rgb, mask, box = _native_foreground_target(row)
        face = _face_pixels(fit, row, slot, mask, box)
        exterior = _exterior_pixels(mask, box, options)
        targets.append((rgb.cuda(), mask.cuda(), box, None if face is None else torch.as_tensor(face, device='cuda'),
                        torch.as_tensor(exterior, device='cuda'), float((mask > .99).float().mean())))
    if not any(face is not None for _, _, _, face, _, _ in targets):
        raise ValueError('No source-supported face interiors for surface refinement.')

    seed = effective['seed']
    torch.manual_seed(seed)
    args = dict(effective['optimization'], position_lr_init=options['position_lr'],
        position_lr_final=options['position_final_lr'], position_lr_max_steps=options['steps'],
        scaling_lr=options['scaling_lr'], rotation_lr=options['rotation_lr'],
        feature_lr=options['feature_lr'], opacity_lr=options['opacity_lr'])
    fields = ('_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation')
    for name in fields:
        getattr(model, name).requires_grad_(True)
    model.spatial_lr_scale = 1.
    model.max_radii2D = torch.zeros(len(model.binding), device='cuda')
    model.training_setup(SimpleNamespace(**args))
    if model.optimizer.state:
        raise RuntimeError('Surface refinement must start with an empty optimizer.')
    posed = model.vertices_sequence.clone()
    topology = model.faces.clone()
    extent = float(np.linalg.norm(np.ptp(fit['vertices_world'][reference], axis=0)))
    initial = len(model.binding)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    black = torch.zeros(3, device='cuda')
    bins, order = source_bin_training_order(rows, selection, seed=seed)
    draws, events = Counter(), []
    out = root/'gaussian-run'
    start = time.monotonic()
    for step in range(1, options['steps']+1):
        model.update_learning_rate(step)
        i = next(order)
        draws[bins[i]] += 1
        source, mask, box, face, exterior, area = targets[i]
        rgb_pkg = render_pose(render, cameras[i], model, pipe, black, layout['pose_indices'][i])
        alpha_pkg = render(cameras[i], model, pipe, black, override_color=torch.ones_like(model._xyz))
        rgb, alpha = rgb_pkg['render'], alpha_pkg['render'][:1]
        photo, _, _, coverage = _sh0_foreground_alpha_loss(rgb, alpha, source, mask, box,
            alpha_floor=effective['straight_rgb_alpha_floor'])
        visible = rgb_pkg['visibility_filter']
        loss = photo + effective['alpha_loss_weight']*coverage + _geometry_loss(
            model._xyz[visible], model.get_scaling[visible], extent, options)
        if face is not None:
            x0, y0, _, _ = box
            interior = alpha[0, face[0]+y0, face[1]+x0]
            loss = loss + options['face_composite_weight']*face_composite_loss(rgb, alpha, source, box, face)
            loss = loss + options['face_alpha_weight']*(options['interior_alpha_target']-interior).relu().mean()
        if exterior.numel():
            loss = loss + options['exterior_weight']*alpha[0, exterior[0], exterior[1]].mean()
        loss.backward()
        with torch.no_grad():
            if step <= options['split_steps'][-1]:
                # Either channel can show missing support. Keep its magnitude
                # without letting the other raster's gradient cancel it.
                rgb_screen = rgb_pkg['viewspace_points'].grad[visible, :2].norm(dim=1, keepdim=True)
                alpha_screen = alpha_pkg['viewspace_points'].grad[visible, :2].norm(dim=1, keepdim=True)
                model.xyz_gradient_accum[visible] += torch.maximum(rgb_screen, alpha_screen)*area
                model.denom[visible] += 1
            if step == 1 or step % 100 == 0:
                if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all()
                                                  for p in (getattr(model, name) for name in fields if name != '_features_rest')):
                    raise RuntimeError('Nonfinite surface refinement loss or gradient.')
                rt.save(out/'surface-refinement-progress.json', dict(step=step, steps=options['steps'], points=len(model.binding),
                    seconds=time.monotonic()-start, loss=float(loss)))
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)
            if step in options['split_steps']:
                score = (model.xyz_gradient_accum / model.denom.clamp_min(1)).flatten()
                chosen = _split_candidates(score, head_faces[model.binding], options)
                before = len(model.binding)
                # The existing local-coordinate split gives small and large
                # candidates distinct centers instead of coincident clones.
                model.densify_and_split(chosen.float()[:, None], .5, 0.)
                after_split = len(model.binding)
                model.prune_points(model.get_opacity.flatten() < options['min_opacity'])
                if int(model.binding_counter.min()) < 1:
                    raise RuntimeError('Surface refinement pruned a required triangle.')
                events.append(dict(step=step, selected=int(chosen.sum()), added=after_split-before,
                                   pruned=after_split-len(model.binding)))
    model.optimizer = None
    for name in fields:
        getattr(model, name).requires_grad_(False)
    if not torch.equal(posed, model.vertices_sequence) or not torch.equal(topology, model.faces):
        raise RuntimeError('Surface refinement modified a frozen pose or mesh.')
    if not all(torch.isfinite(getattr(model, name)).all() for name in fields):
        raise RuntimeError('Surface refinement produced nonfinite terminal Gaussian parameters.')
    snapshot = out/f'{baseline_snapshot.name}-surface-{options["steps"]:06d}'
    replay = _save(root, baseline_snapshot, snapshot, model, reference, cameras[ref_index], pipe, black)
    return replay, dict(state='completed', steps=options['steps'], snapshot=snapshot.name,
        baseline_snapshot=baseline_snapshot.name, initial_points=initial, points=len(model.binding),
        frozen_mesh_verified=True, options=options, split_events=events, source_bin_draws=dict(draws),
        face_support_pixels={row['frame']: 0 if target[3] is None else int(target[3].shape[1])
                             for row, target in zip(rows, targets, strict=True)}, seconds=time.monotonic()-start)
