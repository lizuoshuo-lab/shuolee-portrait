"""Foreground RGB and explicit alpha supervision for the MSV2 demo adapter."""
from collections import OrderedDict
from pathlib import Path
import json, math
import numpy as np

TARGETS = OrderedDict()
LAST = {}


def targets(camera):
    import torch
    F = torch.nn.functional
    key = id(camera)
    if key not in TARGETS:
        while len(TARGETS) >= 2:
            TARGETS.popitem(last=False)
        mask = torch.as_tensor(camera.mask.copy(), device='cuda', dtype=torch.float32)[None] / 255
        valid = mask > .99
        support = F.max_pool2d(F.pad((~valid).float()[None], (5, 5, 5, 5), value=1), 11, stride=1)[0] == 0
        if valid.sum() < 1000 or support.sum() < 1000:
            raise RuntimeError('Insufficient reliable foreground support')
        TARGETS[key] = (mask, valid, support)
    TARGETS.move_to_end(key)
    return TARGETS[key]


def foreground_photo(rgb, target, valid, support):
    import torch
    from fused_ssim import FusedSSIMMap
    a = torch.where(valid, rgb, 0)
    b = torch.where(valid, target, 0)
    ssim_map = FusedSSIMMap.apply(.01 ** 2, .03 ** 2, a[None].contiguous(), b[None].contiguous(), 'same', True)[0]
    l1 = (a - b).abs().sum() / (3 * valid.sum())
    ssim = (ssim_map * support).sum() / (3 * support.sum())
    return .8 * l1 + .2 * (1 - ssim), l1, ssim


def render_alpha(camera, model, pipe, screen_points=None, culling=None):
    import torch
    from diff_gaussian_rasterization_ms import GaussianRasterizationSettings, GaussianRasterizer
    if screen_points is None:
        screen_points = torch.zeros_like(model.get_xyz, requires_grad=True) + 0
    if culling is None:
        culling = torch.zeros(len(model.get_xyz), device='cuda', dtype=torch.bool)
    settings = GaussianRasterizationSettings(
        image_height=camera.height, image_width=camera.width,
        tanfovx=math.tan(camera.FoVx / 2), tanfovy=math.tan(camera.FoVy / 2),
        bg=torch.zeros(3, device='cuda'), scale_modifier=1.,
        viewmatrix=camera.world_view_transform, projmatrix=camera.full_proj_transform,
        sh_degree=model.active_sh_degree, campos=camera.camera_center, prefiltered=False, debug=pipe.debug)
    image, radii, _ = GaussianRasterizer(settings)(
        means3D=model.get_xyz, means2D=screen_points, opacities=model.get_opacity,
        culling=culling, colors_precomp=torch.ones_like(model.get_xyz),
        scales=model.get_scaling, rotations=model.get_rotation, flag_max_count=False)
    return image[0:1], radii


def rgba_loss(camera, model, pipe, package):
    mask, valid, support = targets(camera)
    alpha, _ = render_alpha(camera, model, pipe, package['viewspace_points'], model._culling[:, camera.uid])
    # The RGB rasterizer is rendered on zero ONLY to obtain premultiplied RGB.
    # No background RGB is a training target; alpha is supervised separately.
    straight = package['render'] / alpha.clamp_min(1e-4)
    photo, l1, ssim = foreground_photo(straight, camera.original_image, valid, support)
    error = (alpha - mask).abs()
    silhouette = .5 * ((error * mask).sum() / mask.sum().clamp_min(1) +
                       (error * (1 - mask)).sum() / (1 - mask).sum().clamp_min(1))
    loss = photo + .2 * silhouette
    LAST.clear()
    LAST.update(photo=photo.detach(), foreground_l1=l1.detach(), foreground_ssim=ssim.detach(), alpha_loss=silhouette.detach())
    return loss, l1


def gate(ex, model):
    """Same-run 1k checkpoint gate; the comparison model is never an initializer."""
    import torch
    from scene import GaussianModel
    from gaussian_renderer import render
    from types import SimpleNamespace
    baseline = Path('/opt/gs-slam-lab/msv2-demo-swinir4x-20261104/model/point_cloud/iteration_1000/point_cloud.ply')
    old = GaussianModel(0); old.load_ply(str(baseline)); old.active_sh_degree = 0
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    rows = ex.read(ex.RUN / 'native-inputs.json')['records']
    out = ex.RUN / 'gate1000'; out.mkdir(exist_ok=False)
    checks = []
    with torch.no_grad():
        for i in sorted(set(ex.SELECTED + [27])):
            cam = ex.Camera(rows[i], i)
            mask = torch.as_tensor(cam.mask.copy(), device='cuda', dtype=torch.float32) / 255
            gt = torch.as_tensor(cam.rgb.copy(), device='cuda', dtype=torch.float32).permute(2, 0, 1) / 255
            entry = dict(frame=rows[i]['frame'])
            for key, g in [('baseline', old), ('transparent', model)]:
                pred = render(cam, g, pipe, torch.zeros(3, device='cuda'))['render']
                alpha = render(cam, g, pipe, torch.zeros(3, device='cuda'), override_color=torch.ones_like(g.get_xyz))['render'][0]
                entry[key] = dict(outside_alpha_mean=float(alpha[mask < .01].mean()),
                                  outside_alpha_gt_01=float((alpha[mask < .01] > .1).float().mean()),
                                  inside_alpha_mean=float(alpha[mask > .99].mean()),
                                  foreground_l1=float((pred - gt).abs()[:, mask > .99].mean()))
                ex.save_image(out / (rows[i]['frame'] + '_' + key + '_gray.png'), pred + .5 * (1 - alpha[None]))
            checks.append(entry)
    means = {k: {m: float(np.mean([c[k][m] for c in checks])) for m in checks[0][k]} for k in ['baseline', 'transparent']}
    b, t = means['baseline'], means['transparent']
    criteria = dict(outside_opacity_reduced_or_already_below_one_percent=t['outside_alpha_mean'] <= max(.5 * b['outside_alpha_mean'], .01),
                    foreground_opacity_retained=t['inside_alpha_mean'] >= b['inside_alpha_mean'] - .03,
                    foreground_l1_within_20_percent=t['foreground_l1'] <= 1.2 * b['foreground_l1'])
    passed = all(criteria.values())
    ex.save(out / 'gate.json', dict(passed=passed, criteria=criteria, means=means, checks=checks,
                                   baseline_sha256=ex.sha(baseline), quality_accepted=False,
                                   purpose='Early pollution reduction and subject-protection gate, not final reconstruction acceptance'))
    print('TRANSPARENT_1000_GATE', json.dumps(dict(passed=passed, means=means, criteria=criteria)), flush=True)
    del old
    torch.cuda.empty_cache()
    if not passed:
        raise RuntimeError('Transparent 1000-step protection gate failed; inspect gate1000/gate.json before additional training')
