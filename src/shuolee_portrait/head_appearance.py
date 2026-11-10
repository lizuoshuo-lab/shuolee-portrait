"""Frozen-body appearance tail for the terminal SH0 bound snapshot."""

from collections import Counter
import shutil
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from . import runtime as rt


STEPS = 600
WEIGHT = .05
APPEARANCE_FIELDS = ('f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity')


def snapshot_name(receipt):
    """Validate the checkpoint selected by a stock or head-polished SH0 run."""
    steps = receipt.get('steps')
    if type(steps) is not int or steps <= 0:
        raise ValueError('Invalid terminal SH0 step count.')
    name = f'step-{steps:06d}'
    appearance = receipt.get('head_appearance')
    if appearance is not None:
        if not isinstance(appearance, dict):
            raise ValueError('Invalid head appearance receipt.')
        if appearance.get('state') == 'completed' and appearance.get('steps') == STEPS:
            name = f'{name}-head-{STEPS:06d}'
            if appearance.get('snapshot') != name:
                raise ValueError('The head appearance snapshot differs from the registered terminal.')
        elif (appearance.get('state') not in ('skipped', 'failed')
              or appearance.get('snapshot', name) != name):
            raise ValueError('Invalid head appearance fallback snapshot.')
    surface = receipt.get('surface_refinement')
    if surface is not None:
        if (not isinstance(surface, dict) or surface.get('state') != 'completed'
                or type(surface.get('steps')) is not int or surface['steps'] <= 0
                or surface.get('baseline_snapshot') != name or surface.get('frozen_mesh_verified') is not True):
            raise ValueError('Invalid surface refinement receipt.')
        name = f'{name}-surface-{surface["steps"]:06d}'
        if surface.get('snapshot') != name:
            raise ValueError('The refined snapshot differs from the registered terminal.')
    coverage = receipt.get('coverage_refinement')
    if coverage is not None:
        if (surface is None or not isinstance(coverage, dict) or coverage.get('state') != 'completed'
                or type(coverage.get('steps')) is not int or coverage['steps'] <= 0
                or coverage.get('baseline_snapshot') != name or coverage.get('frozen_mesh_verified') is not True
                or coverage.get('initial_points') != surface.get('points')):
            raise ValueError('Invalid coverage refinement receipt.')
        name = f'{name}-coverage-{coverage["steps"]:06d}'
        if coverage.get('snapshot') != name:
            raise ValueError('The coverage snapshot differs from the registered terminal.')
    elif surface is not None and surface.get('points') != receipt.get('points'):
        raise ValueError('The refined snapshot differs from the registered terminal.')
    shape = receipt.get('shape_refinement')
    if shape is not None:
        if (coverage is None or not isinstance(shape, dict) or shape.get('state') != 'completed'
                or type(shape.get('steps')) is not int or shape['steps'] <= 0
                or shape.get('baseline_snapshot') != name or shape.get('frozen_mesh_verified') is not True
                or shape.get('initial_points') != coverage.get('points')):
            raise ValueError('Invalid shape refinement receipt.')
        name = f'{name}-shape-{shape["steps"]:06d}'
        if shape.get('snapshot') != name or shape.get('points') != receipt.get('points'):
            raise ValueError('The shape snapshot differs from the registered terminal.')
    elif coverage is not None and coverage.get('points') != receipt.get('points'):
        raise ValueError('The coverage snapshot differs from the registered terminal.')
    if receipt.get('final_snapshot') != name:
        raise ValueError('The selected SH0 snapshot does not match its terminal receipt.')
    return name


def editable_points(protected_vertices, faces, binding):
    return protected_vertices[faces].all(axis=1)[binding]


def face_interior_pixels(landmarks, target_alpha, box):
    from scipy.ndimage import binary_erosion
    from scipy.spatial import ConvexHull, QhullError

    h, w = target_alpha.shape[-2:]
    xy = landmarks[:, :2]
    supported = np.isfinite(landmarks).all(axis=1) & (landmarks[:, 2] >= .5)
    supported &= ((xy >= 0) & (xy < [w, h])).all(axis=1)
    if supported.sum() < 8:
        return None
    x0, y0, x1, y1 = box
    points = xy[supported] - [x0, y0]
    try:
        hull = ConvexHull(points)
    except QhullError:
        return None
    raster = Image.new('1', (x1-x0, y1-y0))
    ImageDraw.Draw(raster).polygon([tuple(point) for point in points[hull.vertices]], fill=1)
    foreground = target_alpha[0, y0:y1, x0:x1].detach().cpu().numpy() > .99
    rows, columns = np.nonzero(binary_erosion(np.asarray(raster) & foreground, iterations=2))
    return np.stack((rows, columns)) if len(rows) >= 32 else None


def face_composite_loss(rgb, accumulated_alpha, target_rgb, box, pixels):
    x0, y0, _, _ = box
    rows, columns = pixels
    black_error = rgb[:, y0+rows, x0+columns] - target_rgb[:, rows, columns]
    white_error = black_error + (1 - accumulated_alpha[:, y0+rows, x0+columns])
    return .5*(black_error.abs() + white_error.abs()).mean()


def _vertices(path):
    from plyfile import PlyData

    return PlyData.read(path, mmap=False)['vertex'].data


def _check_frozen(source, output, editable):
    stock, result = _vertices(source), _vertices(output)
    if (stock.dtype != result.dtype or len(stock) != len(editable) or len(stock) != len(result)
            or stock[~editable].tobytes() != result[~editable].tobytes()
            or any(stock[key].tobytes() != result[key].tobytes()
                   for key in stock.dtype.names if key not in APPEARANCE_FIELDS)):
        raise RuntimeError('Head appearance tail changed frozen Gaussian rows, geometry, or bindings.')


def _save_snapshot(root, stock, snapshot, model, editable, reference, camera):
    from plyfile import PlyData, PlyElement
    import torch
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from .bound_replay import save_bound_snapshot
    from .training import image, reference_ply_comparison, render_pose

    snapshot.mkdir(exist_ok=False)
    mesh = snapshot/'reference-mesh.npz'
    shutil.copyfile(stock/mesh.name, mesh)
    mesh_sha = rt.sha(mesh)
    if mesh_sha != rt.sha(stock/mesh.name):
        raise RuntimeError('Head appearance tail changed the frozen mesh bytes.')
    world_path = snapshot/'world.ply'
    world = _vertices(stock/world_path.name)
    dc = model._features_dc.detach().cpu().numpy()[editable, 0, :]
    opacity = model._opacity.detach().cpu().numpy()[editable, 0]
    for key, values in zip(APPEARANCE_FIELDS, (*dc.T, opacity), strict=True):
        world[key][editable] = values
    PlyData([PlyElement.describe(world, 'vertex')], text=False, byte_order='<').write(world_path)
    _check_frozen(stock/world_path.name, world_path, editable)

    reloaded = GaussianModel(0)
    reloaded.load_ply(str(world_path))
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    black = torch.zeros(3, device=model.binding.device)
    with torch.no_grad():
        _, delta = reference_ply_comparison(render, camera, model, reloaded, pipe, black, reference)
        parity = dict(mean_abs=float(delta.mean()), max_abs=float(delta.max()))
        if not (parity['mean_abs'] < 2e-6 and parity['max_abs'] < .005):
            raise RuntimeError(f'Head appearance native PLY does not match its bound model: {parity}')
        rt.save(snapshot/'native-ply-parity.json', dict(passed=True, checks=[parity], sha256=rt.sha(world_path),
                mesh_sha256=mesh_sha, reference_frame_id=rt.read(stock/'native-ply-parity.json')['reference_frame_id'],
                reference_pose_index=reference))
        replay = save_bound_snapshot(root, snapshot, model)
        _check_frozen(stock/'bound-local.ply', snapshot/'bound-local.ply', editable)
        rgb = render_pose(render, camera, replay[0], pipe, black, reference)['render']
        image(rgb).save(snapshot/'reference-render.png')
    return replay


def polish(root, stock, replay, fit, rows, layout, selection, protected, cameras, targets, effective):
    """Run a separate small Adam on strict-owned color and opacity only."""
    import torch
    from gaussian_renderer import render
    from .stable_selection import source_bin_training_order
    from .training import _sh0_foreground_alpha_loss, render_pose

    model, _ = replay
    editable = editable_points(protected, fit['faces'], model.binding.cpu().numpy())
    if not editable.any():
        return replay, dict(state='skipped', reason='no_editable_head_points', steps=0, snapshot=stock.name)
    observed = fit.get('observed_native_keypoints')
    pixels = [face_interior_pixels(observed[slot, 67:118], mask, box) if observed is not None else None
              for slot, (_, mask, box) in zip(layout['pose_indices'], targets, strict=True)]
    if all(region is None for region in pixels):
        return replay, dict(state='skipped', reason='no_usable_face_observations', steps=0, snapshot=stock.name)

    device = model.binding.device
    owned = torch.as_tensor(np.flatnonzero(editable), device=device)
    pixels = [torch.as_tensor(region, device=device) if region is not None else None for region in pixels]
    original = (model._features_dc, model._opacity)
    leaves = [torch.nn.Parameter(value.detach().index_select(0, owned)) for value in original]
    rates = [effective['optimization'][name] for name in ('feature_lr', 'opacity_lr')]
    optimizer = torch.optim.Adam([dict(params=[leaf], lr=lr) for leaf, lr in zip(leaves, rates, strict=True)],
                                 lr=0., eps=1e-15)
    if model.optimizer is not None or optimizer.state:
        raise RuntimeError('Head appearance requires a fresh, separate optimizer.')

    _, order = source_bin_training_order(rows, selection, seed=effective['seed'])
    draws = Counter()
    frozen_pose_bytes = model.vertices_sequence.contiguous().view(torch.uint8).clone()
    frozen_faces = model.faces.clone()
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    black = torch.zeros(3, device=device)
    white = torch.ones_like(model._xyz)
    ref = layout['reference_pose_index']
    reference_camera = cameras[layout['pose_indices'].index(ref)]
    snapshot = stock.with_name(f'{stock.name}-head-{STEPS:06d}')
    started = time.monotonic()
    try:
        for step in range(1, STEPS+1):
            i = next(order)
            draws[rows[i]['frame']] += 1
            model._features_dc, model._opacity = (base.index_copy(0, owned, leaf)
                                                  for base, leaf in zip(original, leaves, strict=True))
            rgb = render_pose(render, cameras[i], model, pipe, black, layout['pose_indices'][i])['render']
            alpha = render(cameras[i], model, pipe, black, override_color=white)['render'][:1]
            source, mask, box = targets[i]
            photo, _, _, coverage = _sh0_foreground_alpha_loss(rgb, alpha, source, mask, box,
                alpha_floor=effective['straight_rgb_alpha_floor'])
            if pixels[i] is not None:
                photo = photo + WEIGHT*face_composite_loss(rgb, alpha, source, box, pixels[i])
            loss = photo + effective['alpha_loss_weight']*coverage
            loss.backward()
            if step == 1 or step % 50 == 0:
                if not (torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in leaves)):
                    raise RuntimeError('Nonfinite head appearance loss or gradient.')
                rt.save(stock.parent/'head-appearance-progress.json', dict(step=step, steps=STEPS,
                    frame=rows[i]['frame'], seconds=time.monotonic()-started, loss=float(loss.detach())))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        model._features_dc, model._opacity = (base.index_copy(0, owned, leaf.detach())
                                              for base, leaf in zip(original, leaves, strict=True))
        if (len(optimizer.state) != len(leaves) or any(int(state['step']) != STEPS for state in optimizer.state.values())
                or any(not torch.isfinite(leaf).all() for leaf in leaves)
                or all(torch.equal(base[owned], leaf) for base, leaf in zip(original, leaves, strict=True))):
            raise RuntimeError('The head appearance optimizer did not produce finite owned updates.')
        if (not torch.equal(frozen_pose_bytes, model.vertices_sequence.contiguous().view(torch.uint8))
                or not torch.equal(frozen_faces, model.faces)):
            raise RuntimeError('The head appearance tail changed the frozen posed meshes.')
        result = _save_snapshot(root, stock, snapshot, model, editable, ref, reference_camera)
    finally:
        model._features_dc, model._opacity = original
        model.select_mesh_by_timestep(ref)
    return result, dict(state='completed', steps=STEPS, snapshot=snapshot.name, editable_points=int(editable.sum()),
        frozen_points=int((~editable).sum()), face_weight=WEIGHT, feature_dc_lr=rates[0], opacity_lr=rates[1],
        seconds=time.monotonic()-started, frame_draws=dict(draws), frozen_ply_bytes_verified=True,
        roi_pixels={row['frame']: 0 if region is None else region.shape[1] for row, region in zip(rows, pixels, strict=True)})
