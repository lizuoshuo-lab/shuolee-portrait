"""Fresh mesh-local Gaussian training. This is a local SMPL-X adapter."""
import argparse
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
from PIL import Image
from . import runtime as rt


def camera(row, index, height=None, *, pose_index=0):
    w, h = row['wh']
    if height is not None:
        w, h = round(w * height / h), height
    K = np.diag([w / row['wh'][0], h / row['wh'][1], 1.]) @ np.asarray(row['K'])
    E = np.asarray(row['w2c'])
    if (K.shape != (3, 3) or E.shape != (4, 4) or
            not np.isfinite(K).all() or not np.isfinite(E).all() or
            not np.array_equal(K[2], [0., 0., 1.]) or K[0, 0] <= 0 or K[1, 1] <= 0):
        raise ValueError('Invalid full corner-convention K or w2c: ' + row['frame'])
    if K[0, 1] != 0 or K[1, 0] != 0:
        raise ValueError('The existing rasterizer does not support camera skew; K must not be silently changed.')
    from scene.cameras import CameraProj2
    cam = CameraProj2(index, E[:3, :3].T, E[:3, 3], K[0, 0], K[1, 1], 'black',
                     w, None, h, row.get('image', ''), row['frame'], index,
                     cx=K[0, 2], cy=K[1, 2], timestep=pose_index)
    # Test the actual matrix sent to CUDA against pinhole pixel coordinates.
    cp = np.array([[0., 0., 2.], [.1, .2, 3.], [-.4, -.3, 4.]])
    world = (cp - E[:3, 3]) @ E[:3, :3]
    q = cp @ K.T
    expected = q[:, :2] / q[:, 2:] - .5
    clip = np.c_[world, np.ones(3)] @ cam.full_proj_transform.numpy()
    actual = ((clip[:, :2] / clip[:, 3:] + 1) * [w, h] - 1) / 2
    error = float(abs(expected - actual).max())
    if error > .001:
        raise RuntimeError(f'Projection disagreement {row["frame"]}: {error} px')
    return cam, dict(frame=row['frame'], K=K.tolist(), w2c=E.tolist(), wh=[w, h],
                     pose_index=pose_index, pixel_convention='pixel_corners', projection_error_px=error)


def named_pose_layout(fit, rows):
    """Bind the saved mesh sequence to named observations, without sorting.

    Legacy unnamed files remain usable only for their original identical-pose
    case. Named files retain the native source K/E and one final shared beta.
    This verifies interface identity, not fitting or reconstruction quality.
    """
    def need(value, message):
        if not value:
            raise ValueError(message)
    ids = [r['frame'] for r in rows]
    need(len(ids) > 0 and all(isinstance(v, str) and v for v in ids) and len(set(ids)) == len(ids),
         'Observation frame identities must be nonempty and unique.')
    vertices, faces = np.asarray(fit['vertices_world']), np.asarray(fit['faces'])
    need(vertices.ndim == 3 and len(vertices) > 0 and vertices.shape[1] >= 3 and vertices.shape[2] == 3
         and np.isfinite(vertices).all(), 'Invalid common-topology mesh sequence.')
    need(faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0 and faces.dtype.kind in 'iu'
         and np.all(faces >= 0) and np.all(faces < vertices.shape[1]), 'Invalid common faces.')
    if 'frame_ids' not in fit:
        need(np.array_equal(vertices, np.broadcast_to(vertices[0], vertices.shape)),
             'Nonstatic meshes require explicit frame_ids; positional ordering is not sufficient.')
        return dict(named=False, frame_ids=ids, pose_indices=[0]*len(ids), pose_by_frame=dict.fromkeys(ids, 0),
                    reference_frame_id=ids[0], reference_pose_index=0, initialization_frame_id=ids[0])
    mesh_ids = np.asarray(fit['frame_ids'])
    need(mesh_ids.shape == (len(vertices),) and mesh_ids.dtype.kind in 'US', 'Invalid mesh frame_ids.')
    mesh_ids = mesh_ids.astype(str).tolist()
    need(len(set(mesh_ids)) == len(mesh_ids) and set(mesh_ids) == set(ids),
         'Mesh and observation frame identities must match exactly once.')
    slots = {name: i for i, name in enumerate(mesh_ids)}
    reference = np.asarray(fit['reference_frame_id'])
    need(reference.shape == () and reference.dtype.kind in 'US' and str(reference.item()) in slots,
         'A named reference frame in this sequence is required.')
    beta = np.asarray(fit['shared_betas'])
    need(beta.shape == (1, 10) and np.isfinite(beta).all(), 'One final shared beta is required.')
    if 'betas' in fit:
        need(np.array_equal(beta, fit['betas']), 'Duplicate shared beta values disagree.')
    for key in ('source_frames', 'image_sha256', 'mask_sha256'):
        need(np.asarray(fit[key]).shape == (len(mesh_ids),), 'Invalid named source field: ' + key)
    for key, shape in (('K_integer_centers', (len(mesh_ids), 3, 3)), ('w2c', (len(mesh_ids), 4, 4))):
        a = np.asarray(fit[key])
        need(a.shape == shape and np.isfinite(a).all(), 'Invalid named camera field: ' + key)
    for row in rows:
        slot = slots[row['frame']]
        need(fit['source_frames'][slot] == row['source_frame_index'] and
             str(fit['image_sha256'][slot]) == row['source_sha256'] and
             str(fit['mask_sha256'][slot]) == row['mask_sha256'], 'Named image/mask/source-frame association differs.')
        expected_K = np.asarray(fit['K_integer_centers'][slot], dtype=np.float64).copy()
        expected_K[0, 2] += .5; expected_K[1, 2] += .5
        need(np.array_equal(expected_K, row['K']), 'Observation K must be the saved native K plus the explicit half pixel.')
        need(np.array_equal(fit['w2c'][slot], row['w2c']), 'Named pose and observation w2c differ.')
    reference = str(reference.item())
    return dict(named=True, frame_ids=mesh_ids, pose_indices=[slots[name] for name in ids], pose_by_frame=slots,
                reference_frame_id=reference, reference_pose_index=slots[reference], initialization_frame_id=mesh_ids[0])


def render_pose(render, cam, model, pipe, background, pose_index, **kwargs):
    """Select the bound mesh for every call; a world-PLY model is already posed."""
    if getattr(model, 'binding', None) is not None:
        model.select_mesh_by_timestep(pose_index)
    return render(cam, model, pipe, background, **kwargs)


def reference_ply_comparison(render, cam, model, reloaded, pipe, background, reference_pose):
    reference = render_pose(render, cam, model, pipe, background, reference_pose)['render']
    return reference, abs(reference-render(cam, reloaded, pipe, background)['render'])


def foreground_loss(im, gt, alpha):
    import torch
    F = torch.nn.functional
    valid = alpha > .99
    support = F.max_pool2d(F.pad((~valid).float()[None], (5, 5, 5, 5), value=1), 11, stride=1)[0] == 0
    if valid.sum() < 1000 or support.sum() < 1000:
        raise RuntimeError('Foreground too small for reliable training at the configured resolution.')
    a, b = torch.where(valid, im, 0), torch.where(valid, gt, 0)
    q = torch.arange(11, device=im.device, dtype=im.dtype) - 5
    kernel = torch.exp(-q.square() / (2 * 1.5 ** 2)); kernel /= kernel.sum()
    win = (kernel[:, None] * kernel[None, :])[None, None].expand(3, 1, 11, 11).contiguous()
    conv = lambda x: F.conv2d(x[None], win, padding=5, groups=3)[0]
    u, v = conv(a), conv(b)
    aa, bb, ab = conv(a*a)-u*u, conv(b*b)-v*v, conv(a*b)-u*v
    ss = ((2*u*v+.01**2)*(2*ab+.03**2))/((u*u+v*v+.01**2)*(aa+bb+.03**2))
    l1 = (im-gt).abs()[:, valid[0]].mean(); ssim = ss[:, support[0]].mean()
    return .8*l1+.2*(1-ssim), l1, ssim


def image(tensor):
    return Image.fromarray(np.rint(tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).astype('uint8'))


class Views:
    def __init__(self, rows, height, cache_size, *, pose_indices=None):
        self.rows = rows; self.cache = OrderedDict(); self.cache_size = cache_size
        if pose_indices is None:
            pose_indices = [0]*len(rows)
        if len(pose_indices) != len(rows):
            raise ValueError('One named pose slot per observation is required.')
        self.cameras, self.records = zip(*(camera(r, i, height, pose_index=pose_indices[i]) for i, r in enumerate(rows)))
        for r in rows:
            if rt.sha(r['image']) != r['source_sha256'] or rt.sha(r['mask']) != r['mask_sha256']:
                raise RuntimeError(f'Input changed: {r["frame"]}')

    def get(self, i):
        import torch
        if i not in self.cache:
            r = self.rows[i]; size = tuple(self.records[i]['wh'])
            with Image.open(r['image']) as im:
                rgb = torch.from_numpy(np.array(im.convert('RGB').resize(size, Image.Resampling.LANCZOS))).permute(2, 0, 1)
            with Image.open(r['mask']) as im:
                alpha = torch.from_numpy(np.array(im.convert('L').resize(size, Image.Resampling.BILINEAR)))[None]
            self.cache[i] = rgb, alpha
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(i); rgb, alpha = self.cache[i]
        return self.cameras[i], rgb.cuda().float()/255, alpha.cuda().float()/255


def train(root, cfg):
    from .admission import require_alignment
    require_alignment(root)
    return _train_static(root, cfg)


def verify_official_fit(root):
    """Read actual official frame receipts; quality approval is not an input.

    Descriptor paths are absolute or relative to the reconstruction root. The
    small aggregate is an index of completed runs, not a new fitting format.
    """
    root = Path(root)
    receipt = rt.read(root/'fit/official-fit-result.json')
    if (receipt.get('state') != 'complete' or
            receipt.get('fit_sha256') != rt.sha(root/'fit/fitted.npz') or
            receipt.get('training_inputs_sha256') != rt.sha(root/'data/training-inputs.json')):
        raise ValueError('Official completed-fit receipt does not bind these actual fit/source inputs.')
    with np.load(root/'fit/fitted.npz', allow_pickle=False) as archive:
        fit = {key: archive[key] for key in archive.files}
    rows = rt.read(root/'data/training-inputs.json')['records']
    layout = named_pose_layout(fit, rows)
    if not layout['named']:
        raise ValueError('The official entry requires the complete named pose output.')
    frames = receipt.get('official_frame_results', [])
    if (not isinstance(frames, list) or len(frames) != len(rows) or
            {item.get('frame_id') for item in frames} != set(layout['frame_ids'])):
        raise ValueError('One completed official receipt per named frame is required.')
    def checked(item):
        path = Path(item['path'])
        if not path.is_absolute():
            path = root/path
        if not path.is_file() or rt.sha(path) != item.get('sha256'):
            raise ValueError('Changed or missing official source: ' + str(path))
        return path
    from .official_sequence import checked_image, check_result
    rows_by_frame = {row['frame']: row for row in rows}
    for item in frames:
        result = rt.read(checked(item))
        row = rows_by_frame[item['frame_id']]
        check_result(result, row)
        checked_image(row)
        if (result.get('state') != 'complete' or result.get('frame_id') != item['frame_id'] or
                result.get('upstream_loss_monitor_optimizer_unchanged') is not True or
                result.get('five_stages') is not True or not result.get('official_commit')):
            raise ValueError('Official frame fitting is not complete: ' + item['frame_id'])
        if (result.get('source_frame_index') != row['source_frame_index'] or
                result.get('source_image_sha256') != row['source_sha256']):
            raise ValueError('Official frame result belongs to different source pixels: ' + item['frame_id'])
    sources = receipt.get('sources', [])
    if not isinstance(sources, list) or not sources:
        raise ValueError('Actual official adapter/export source descriptors are required.')
    for item in sources:
        checked(item)
    return dict(receipt_sha256=rt.sha(root/'fit/official-fit-result.json'),
                fit_sha256=receipt['fit_sha256'], training_inputs_sha256=receipt['training_inputs_sha256'],
                official_frame_results=frames, sources=sources, quality_accepted=False,
                formal_training_admitted=False)


def train_official(root, cfg):
    """Use the existing numerical loop on hash-bound official named fits."""
    return _train_static(Path(root), cfg, execution_kind='official_smplifyx')


def _train_static(root, cfg, *, execution_kind='formal_training'):
    official_receipt = None
    if execution_kind == 'official_smplifyx':
        official_receipt = verify_official_fit(root)
    else:
        gate = rt.read(root/'fit/geometry-gate.json')
        if not gate['coarse_initialization_gate_passed'] or gate['fit_sha256'] != rt.sha(root/'fit/fitted.npz'):
            raise RuntimeError('Fresh geometry gate missing, failed or no longer matches the fit.')
    import torch
    from scene.gaussian_model import GaussianModel
    from scene.smplx_gaussian_model import SMPLXGaussianModel
    from gaussian_renderer import render
    from arguments import OptimizationParams
    with np.load(root/'fit/fitted.npz', allow_pickle=False) as archive:
        fit = {key: archive[key] for key in archive.files}
    rows = rt.read(root/'data/training-inputs.json')['records']
    layout = named_pose_layout(fit, rows)
    out = root/'gaussian-run'; out.mkdir()
    seed = cfg['seed']; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
    views = Views(rows, cfg['training_height'], cfg['image_cache_frames'], pose_indices=layout['pose_indices'])
    rt.save(out/'training-cameras.json', dict(records=list(views.records), source_sha256=rt.sha(root/'data/training-inputs.json'), interpolated_cameras=0))
    model = SMPLXGaussianModel(3); model.load_fresh_meshes(root/'fit/fitted.npz')
    init = model.initialize_surface(cfg['initial_points'], seed)
    parser = argparse.ArgumentParser(); options = OptimizationParams(parser).extract(parser.parse_args([]))
    schedule_steps = cfg['iterations']; steps = schedule_steps
    options.iterations = schedule_steps; options.position_lr_max_steps = schedule_steps
    options.densify_until_iter = schedule_steps; model.training_setup(options)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg = torch.zeros(3, device='cuda')
    rt.save(out/'config.json', dict(**cfg, route='local_named_SMPLX_mesh_bound' if layout['named'] else 'local_static_SMPLX_mesh_bound',
                                  upstream_complete_reproduction=False, pose_layout=layout,
                                  official_fit_provenance=official_receipt,
                                  execution_kind=execution_kind, executed_step_budget=steps,
                                  formal_training_admitted=execution_kind=='formal_training',
                                  from_scratch=True, old_subject_checkpoint_loaded=False, initialization=init,
                                  fit_sha256=rt.sha(root/'fit/fitted.npz'), optimizer_initial_state_entries=len(model.optimizer.state)))
    cam, gt, alpha = views.get(0); pkg = render_pose(render, cam, model, pipe, bg, layout['pose_indices'][0])
    photo, _, _ = foreground_loss(pkg['render'], gt, alpha)
    other = gt.clone(); other[:, alpha[0] <= .99] = .37
    outside_effect = float(abs(photo-foreground_loss(pkg['render'], other, alpha)[0]))
    if outside_effect > 1e-7:
        raise RuntimeError('Background RGB leaked into foreground photometric loss.')
    aa = render_pose(render, cam, model, pipe, bg, layout['pose_indices'][0], override_color=torch.ones_like(model.get_xyz))['render'][:1]
    (photo+.1*(aa-alpha).abs().mean()).backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for g in model.optimizer.param_groups for p in g['params']):
        raise RuntimeError('Nonfinite or missing training gradients.')
    model.optimizer.zero_grad(set_to_none=True)
    rt.save(out/'preflight.json', dict(foreground_external_RGB_effect=outside_effect, finite_backward=True, optimizer_updates=0))
    del pkg, photo, aa, other
    if (root/'status.json').is_file():
        status = rt.read(root/'status.json')
        status.update(stage='training', training_started=True, updated=rt.now())
        rt.save(root/'status.json', status)
    counts = np.zeros(len(rows), np.int32); stack = []; rng = np.random.default_rng(seed); start = time.monotonic()
    extent = float(np.linalg.norm(np.ptp(fit['vertices_world'][0], axis=0)))
    selected = np.unique(np.linspace(0, len(rows)-1, 6).astype(int))
    checkpoints = sorted({v for v in [1000, 6000, 12000, steps] if v <= steps})
    with (out/'loss.jsonl').open('x') as log:
        for step in range(1, steps+1):
            model.update_learning_rate(step)
            if step % 1000 == 0: model.oneupSHdegree()
            if not stack: stack = rng.permutation(len(rows)).tolist()
            i = stack.pop(); counts[i] += 1; cam, gt, alpha = views.get(i)
            pkg = render_pose(render, cam, model, pipe, bg, layout['pose_indices'][i]); photo, l1, ssim = foreground_loss(pkg['render'], gt, alpha)
            aa = render_pose(render, cam, model, pipe, bg, layout['pose_indices'][i], override_color=torch.ones_like(model.get_xyz))['render'][:1]
            silhouette = (aa-alpha).abs().mean(); visible = pkg['visibility_filter']
            if not visible.any(): raise RuntimeError('No visible Gaussians in a registered camera.')
            regularizer = torch.relu(model._xyz[visible].norm(dim=-1)-options.threshold_xyz).mean()*options.lambda_xyz
            regularizer += torch.relu(model._scaling[visible].exp()-options.threshold_scale).norm(dim=-1).mean()*options.lambda_scale
            loss = photo+.1*silhouette+regularizer
            if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss at step {step}')
            loss.backward()
            with torch.no_grad():
                model.max_radii2D[visible] = torch.maximum(model.max_radii2D[visible], pkg['radii'][visible])
                if step < steps: model.add_densification_stats(pkg['viewspace_points'], visible)
                model.optimizer.step(); model.optimizer.zero_grad(set_to_none=True)
                if options.densify_from_iter < step < steps and step % options.densification_interval == 0:
                    model.densify_and_prune(options.densify_grad_threshold, .005, extent, None)
                    if len(model._xyz) >= 1200000 or int(model.binding_counter.min()) < 1:
                        raise RuntimeError('Point budget or surface-coverage constraint failed.')
            if step == 1 or step % 100 == 0 or step == steps:
                progress = dict(updated=rt.now(), stage='training', step=step, target_steps=steps, points=len(model.get_xyz),
                                loss=float(loss), foreground_l1=float(l1), foreground_ssim=float(ssim), alpha_mae=float(silhouette),
                                seconds=time.monotonic()-start, views_seen=int((counts>0).sum()), quality_accepted=False,
                                execution_kind=execution_kind)
                rt.save(root/'progress.json', progress); log.write(__import__('json').dumps(progress)+'\n'); log.flush()
                print(progress, flush=True)
            if step in checkpoints:
                with torch.no_grad():
                    snapshot = out/f'step-{step:06d}'; snapshot.mkdir()
                    reference_pose = layout['reference_pose_index']
                    model.export_world_ply(snapshot/'world.ply', reference_pose)
                    reloaded = GaussianModel(3); reloaded.load_ply(str(snapshot/'world.ply')); reloaded.active_sh_degree=model.active_sh_degree
                    checks = []
                    for j in selected:
                        cam, gt, alpha = views.get(int(j))
                        reference, delta = reference_ply_comparison(render, cam, model, reloaded, pipe, bg, reference_pose)
                        checks.append(dict(frame=rows[j]['frame'], pose_frame=layout['reference_frame_id'],
                                           mean_abs=float(delta.mean()), max_abs=float(delta.max())))
                        native = reference if layout['pose_indices'][j] == reference_pose else render_pose(
                            render, cam, model, pipe, bg, layout['pose_indices'][j])['render']
                        image(gt).save(snapshot/f'{rows[j]["frame"]}_input.png')
                        image(gt*alpha).save(snapshot/f'{rows[j]["frame"]}_masked_input.png')
                        image(native).save(snapshot/f'{rows[j]["frame"]}_render.png')
                    passed = all(r['mean_abs'] < 2e-6 and r['max_abs'] < .005 for r in checks)
                    rt.save(snapshot/'native-ply-parity.json', dict(passed=passed, checks=checks, sha256=rt.sha(snapshot/'world.ply'),
                        reference_frame_id=layout['reference_frame_id'], reference_pose_index=reference_pose,
                        observed_render_pose='Each *_render.png uses its matching named input pose; PLY parity uses only the named reference pose.'))
                    if not passed: raise RuntimeError('World PLY does not reproduce native training geometry.')
                    del reloaded
    rt.save(out/'result.json', dict(state='completed', from_scratch=True, steps=steps,
                                   execution_kind=execution_kind, schedule_steps=schedule_steps,
                                   formal_training_admitted=execution_kind=='formal_training',
                                   per_frame_updates=counts.tolist(), pose_layout=layout,
                                   reference_frame_id=layout['reference_frame_id'],
                                   final_snapshot=f'step-{steps:06d}', quality_accepted=False))
