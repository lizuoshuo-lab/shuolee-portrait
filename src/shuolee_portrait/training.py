"""Fresh static mesh-bound training. This is a local SMPL-X adapter."""
import argparse
from collections import OrderedDict
from types import SimpleNamespace
import time
import numpy as np
from PIL import Image
from . import runtime as rt


def camera(row, index, height=None):
    from scene.cameras import CameraProj2
    w, h = row['wh']
    if height is not None:
        w, h = round(w * height / h), height
    K = np.diag([w / row['wh'][0], h / row['wh'][1], 1.]) @ np.asarray(row['K'])
    E = np.asarray(row['w2c'])
    cam = CameraProj2(index, E[:3, :3].T, E[:3, 3], K[0, 0], K[1, 1], 'black',
                     w, None, h, row.get('image', ''), row['frame'], index,
                     cx=K[0, 2], cy=K[1, 2], timestep=0)
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
    return cam, dict(frame=row['frame'], K=K.tolist(), w2c=E.tolist(), wh=[w, h], projection_error_px=error)


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
    def __init__(self, rows, height, cache_size):
        self.rows = rows; self.cache = OrderedDict(); self.cache_size = cache_size
        self.cameras, self.records = zip(*(camera(r, i, height) for i, r in enumerate(rows)))
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
    import torch
    from scene.gaussian_model import GaussianModel
    from scene.smplx_gaussian_model import SMPLXGaussianModel
    from gaussian_renderer import render
    from arguments import OptimizationParams
    gate = rt.read(root/'fit/geometry-gate.json')
    if not gate['coarse_initialization_gate_passed'] or gate['fit_sha256'] != rt.sha(root/'fit/fitted.npz'):
        raise RuntimeError('Fresh geometry gate missing, failed or no longer matches the fit.')
    out = root/'gaussian-run'; out.mkdir()
    fit = dict(np.load(root/'fit/fitted.npz'))
    if np.max(abs(fit['vertices_world']-fit['vertices_world'][0])) != 0:
        raise RuntimeError('Only one shared static body is supported.')
    seed = cfg['seed']; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
    rows = rt.read(root/'data/training-inputs.json')['records']
    views = Views(rows, cfg['training_height'], cfg['image_cache_frames'])
    rt.save(out/'training-cameras.json', dict(records=list(views.records), source_sha256=rt.sha(root/'data/training-inputs.json'), interpolated_cameras=0))
    model = SMPLXGaussianModel(3); model.load_fresh_meshes(root/'fit/fitted.npz')
    init = model.initialize_surface(cfg['initial_points'], seed)
    parser = argparse.ArgumentParser(); options = OptimizationParams(parser).extract(parser.parse_args([]))
    steps = cfg['iterations']; options.iterations = steps; options.position_lr_max_steps = steps
    options.densify_until_iter = steps; model.training_setup(options)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg = torch.zeros(3, device='cuda')
    rt.save(out/'config.json', dict(**cfg, route='local_static_SMPLX_mesh_bound', upstream_complete_reproduction=False,
                                  from_scratch=True, old_subject_checkpoint_loaded=False, initialization=init,
                                  fit_sha256=rt.sha(root/'fit/fitted.npz'), optimizer_initial_state_entries=len(model.optimizer.state)))
    cam, gt, alpha = views.get(0); pkg = render(cam, model, pipe, bg)
    photo, _, _ = foreground_loss(pkg['render'], gt, alpha)
    other = gt.clone(); other[:, alpha[0] <= .99] = .37
    outside_effect = float(abs(photo-foreground_loss(pkg['render'], other, alpha)[0]))
    if outside_effect > 1e-7:
        raise RuntimeError('Background RGB leaked into foreground photometric loss.')
    aa = render(cam, model, pipe, bg, override_color=torch.ones_like(model.get_xyz))['render'][:1]
    (photo+.1*(aa-alpha).abs().mean()).backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for g in model.optimizer.param_groups for p in g['params']):
        raise RuntimeError('Nonfinite or missing training gradients.')
    model.optimizer.zero_grad(set_to_none=True)
    rt.save(out/'preflight.json', dict(foreground_external_RGB_effect=outside_effect, finite_backward=True, optimizer_updates=0))
    del pkg, photo, aa, other
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
            pkg = render(cam, model, pipe, bg); photo, l1, ssim = foreground_loss(pkg['render'], gt, alpha)
            aa = render(cam, model, pipe, bg, override_color=torch.ones_like(model.get_xyz))['render'][:1]
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
                                seconds=time.monotonic()-start, views_seen=int((counts>0).sum()), quality_accepted=False)
                rt.save(root/'progress.json', progress); log.write(__import__('json').dumps(progress)+'\n'); log.flush()
                print(progress, flush=True)
            if step in checkpoints:
                with torch.no_grad():
                    snapshot = out/f'step-{step:06d}'; snapshot.mkdir()
                    model.export_world_ply(snapshot/'world.ply', 0)
                    reloaded = GaussianModel(3); reloaded.load_ply(str(snapshot/'world.ply')); reloaded.active_sh_degree=model.active_sh_degree
                    checks = []
                    for j in selected:
                        cam, gt, alpha = views.get(int(j)); native = render(cam, model, pipe, bg)['render']
                        delta = abs(native-render(cam, reloaded, pipe, bg)['render'])
                        checks.append(dict(frame=rows[j]['frame'], mean_abs=float(delta.mean()), max_abs=float(delta.max())))
                        image(gt).save(snapshot/f'{rows[j]["frame"]}_input.png')
                        image(gt*alpha).save(snapshot/f'{rows[j]["frame"]}_masked_input.png')
                        image(native).save(snapshot/f'{rows[j]["frame"]}_render.png')
                    passed = all(r['mean_abs'] < 2e-6 and r['max_abs'] < .005 for r in checks)
                    rt.save(snapshot/'native-ply-parity.json', dict(passed=passed, checks=checks, sha256=rt.sha(snapshot/'world.ply')))
                    if not passed: raise RuntimeError('World PLY does not reproduce native training geometry.')
                    del reloaded
    rt.save(out/'result.json', dict(state='completed_quality_review_pending', from_scratch=True, steps=steps,
                                   per_frame_updates=counts.tolist(), final_snapshot=f'step-{steps:06d}', quality_accepted=False))
