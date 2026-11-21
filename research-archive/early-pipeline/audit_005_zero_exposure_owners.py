"""Attribute original 005 native render pixels to zero-exposure mesh owners."""

import argparse
import fcntl
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

import audit_005_bound_surface_exposure as exposure
import audit_008_head_ownership as ownership
from review_bound_alpha_ab import pin, read, require, sha
from review_bound_scale_coverage import scale_pixels


OUT = exposure.LAB / 'shuolee-smpl-coverage-005-zero-exposure-owners'
PLAN = OUT / 'preregistration.json'
ARCHIVED = exposure.SNAPSHOT / 'posed-replay'
SUPPORT = exposure.OUTPUT / 'per-face-exposure.npz'
FRAMES = ('frame_000018', 'frame_000020', 'frame_000047')
FIXED = {
    exposure.PLAN: '0d75caca451b6952991088f88dae903f1be675b7a0e30398c153a8ded2187331',
    exposure.RESULT: '2d04c63f047a318d591f19e335be27c256e0643d29af9f0545b89c0bc4ee6377',
    SUPPORT: 'c36bf74d551fae1fd774004bc0ee52d163b9d5571eaf8807d4c148fa17a1a815',
    ARCHIVED / 'result.json': '661107d648f4b9ed874184b91f4ee7d7788fd895a0cbce7dd820fe50fce309bc',
    Path(ownership.__file__).resolve(): '836f27e4551d4d7342ff48b8e80893279149fe0c85571d2b7218c039dda838c4',
    Path(scale_pixels.__code__.co_filename).resolve(): '720982965c8a9d6cd3627481d4dff61129e473d5e3aa3a878e26e6a05f6c0518',
    ownership.RASTER_BINARY: ownership.FIXED_PINS[ownership.RASTER_BINARY],
    ownership.RASTER_ENV / '__init__.py': ownership.FIXED_PINS[ownership.RASTER_ENV / '__init__.py'],
}


def preflight():
    from shuolee_portrait.bound_replay import inspect_bound_ply

    ownership.verify(FIXED)
    original, rt, asset, rows, cameras, fit, _, facts, _, _, binding = exposure.prepare()
    saved, counts = read(exposure.RESULT), read(ARCHIVED / 'result.json')
    require(read(exposure.PLAN) == original
            and saved['preregistration_sha256'] == FIXED[exposure.PLAN]
            and saved['per_face_exposure_sha256'] == FIXED[SUPPORT]
            and saved['state'] == 'source_only_exposure_measured'
            and counts['retained_holdouts'] == original['excluded_heldouts']
            and counts['optimizer_updates'] == 0,
            'The completed source-only audit or bound replay changed.')
    ids, actual_binding = inspect_bound_ply(exposure.SNAPSHOT / 'bound-local.ply', len(fit['faces']))
    with np.load(SUPPORT, allow_pickle=False) as data:
        require(data['frames'].tolist() == [row['frame'] for row in rows]
                and np.array_equal(binding, actual_binding)
                and np.array_equal(binding, data['bound_gaussian_counts'])
                and np.array_equal(data['visible_source_bins'] == 0,
                                   ~np.any(data['native_mask_pixels'] > 0, axis=0)),
                'Saved triangle ownership or native exposure counts differ.')
        zero_faces = data['visible_source_bins'] == 0
    zero_points = zero_faces[ids]
    require(int(zero_faces.sum()) == 8815 and int(zero_points.sum()) == 16784 and len(ids) == 87792,
            'Unexpected original zero-exposure group.')
    ordered = sorted(rows, key=lambda row: row['source_frame_index'])
    earliest, latest = ordered[0], ordered[-1]
    require((earliest['frame'], facts['reference_frame_id'], latest['frame']) == FRAMES
            and set(FRAMES).isdisjoint(exposure.HELD), 'Original earliest/reference/latest selection changed.')
    pins = dict(original['input_sha256'])
    pins.update({str(path): digest for path, digest in FIXED.items()})
    pins[str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    views = []
    for frame in FRAMES:
        row = ownership.one(rows, frame)
        archived = ownership.one(counts['records'], frame)
        expected = ownership.one(cameras, frame)
        require(archived['camera'] == expected and archived['pose_index'] == expected['pose_index']
                and all(row[k] == archived[k] for k in ('source_frame_index', 'source_sha256', 'mask_sha256')),
                f'Archived original pose, image, camera or mask changed: {frame}')
        image = ARCHIVED / archived['image']
        pins[str(image)] = pin(image, archived['image_sha256'])
        with Image.open(image) as archived_rgb, Image.open(row['mask']) as source_mask:
            require(list(archived_rgb.size) == list(source_mask.size) == row['wh'], 'Native raster differs.')
            positive = np.asarray(source_mask.convert('L')) >= exposure.ALPHA_MIN
        yy, xx = np.nonzero(positive)
        require(len(xx) > 0, f'No reliable foreground: {frame}')
        w, h = row['wh']
        box = [max(0, int(xx.min()) - 12), max(0, int(yy.min()) - 12),
               min(w, int(xx.max()) + 13), min(h, int(yy.max()) + 13)]
        views.append(dict(frame=frame, row=row, expected=expected, archive=str(image), crop=box,
                          camera_index=next(i for i, record in enumerate(rows) if record['frame'] == frame)))
    ownership.verify(pins)
    plan = dict(schema='shuolee-005-zero-direct-exposure-attribution-v1',
                original_run=str(exposure.RUN), terminal_step=10000, point_count=len(ids),
                zero_exposure_faces=int(zero_faces.sum()), bound_group_points=int(zero_points.sum()),
                excluded_heldouts=list(exposure.HELD), views=views, input_sha256=pins,
                regions='Original native masks. Interior uint8≥253, exterior uint8=0, report remainder separately.',
                group='All original splats whose assigned terminal mesh triangle had zero direct '
                      'mask-interior raster support across the 23 training views.',
                contribution='One binary RGB override on black retains all original visibility and opacities. '
                             'It measures the group’s transmittance-weighted alpha contribution.',
                counterfactual='Only group opacity logits go to −80 in memory for two renders. '
                               'Restore the complete raw tensor exactly and replay stock again.',
                limitation='Terminal fitted mesh visibility is not material ownership. Measured contribution '
                           'can include legitimate hair or garment appearance beyond the fitted triangle.',
                max_render_calls=21, fit_calls=0, optimizer_updates=0, gpu_calls_at_registration=0)
    return plan, rt, asset, ids, zero_points


def render_frame(view, model, group, directory):
    import torch
    from gaussian_renderer import render
    from shuolee_portrait.training import camera, render_pose

    row, expected = view['row'], view['expected']
    pose = expected['pose_index']
    cam, actual = camera(row, view['camera_index'], pose_index=pose)
    require(actual == expected, 'Native render camera differs from the original training camera.')
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    black, white = torch.zeros(3, device='cuda'), torch.ones(3, device='cuda')
    splat_white = torch.ones_like(model.get_xyz)

    def draw(**kwargs):
        return render_pose(render, cam, model, pipe, black, pose, **kwargs)['render']

    (rgb, _, alpha), composition_error = scale_pixels(render, render_pose, cam, model, pipe, black, white, pose, 1.)
    stock_bytes = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    with Image.open(view['archive']) as archived:
        stock_error = int(abs(stock_bytes.astype(np.int16) - np.asarray(archived.convert('RGB'), dtype=np.int16)).max())
    require(stock_error <= 1, 'Stock render differs from archived native RGB by more than 1 byte.')
    indicator = group[:, None].expand_as(splat_white).to(splat_white.dtype).contiguous()
    group_rgb = draw(override_color=indicator)
    require(torch.isfinite(group_rgb).all().item() and (group_rgb - group_rgb[:1]).abs().max().item() < 1e-6,
            'The visibility-preserving binary pass failed.')
    contrib = group_rgb[0].cpu().numpy()
    require(float(contrib.min()) >= -1e-6 and float((contrib-alpha).max()) <= 2e-5,
            'Group contribution lies outside total original accumulated alpha.')
    raw = model._opacity.detach().clone()
    try:
        model._opacity[group] = -80.
        off_rgb = draw().permute(1, 2, 0).cpu().numpy()
        off_alpha = draw(override_color=splat_white)[0].cpu().numpy()
    finally:
        model._opacity.copy_(raw)
    require(torch.equal(model._opacity, raw), 'The full opacity tensor did not restore bitwise.')
    restored = draw().permute(1, 2, 0).cpu().numpy()
    require(np.array_equal(restored, rgb) and np.isfinite(off_rgb).all() and np.isfinite(off_alpha).all()
            and float(off_alpha.min()) >= -1e-5 and float(off_alpha.max()) <= 1.00001,
            'Restored RGB or suppression alpha is invalid.')
    with Image.open(row['image']) as source, Image.open(row['mask']) as silhouette:
        image = source.convert('RGB')
        truth = np.asarray(image, dtype=np.float32) / 255
        mask = np.asarray(silhouette.convert('L'))
    interior = mask >= exposure.ALPHA_MIN
    exterior = mask == 0

    def measure(region, color_error):
        if not region.any():
            return dict(pixels=0)
        a, b, c = alpha[region], off_alpha[region], contrib[region]
        values = dict(pixels=int(region.sum()), stock_alpha_mass=float(a.sum(dtype=np.float64)),
                      group_alpha_mass=float(c.sum(dtype=np.float64)),
                      group_fraction_of_stock_alpha=float(c.sum(dtype=np.float64) / max(a.sum(dtype=np.float64), 1e-10)),
                      suppressed_alpha_mass=float(b.sum(dtype=np.float64)),
                      stock_pixels_alpha_ge_half=int((a >= .5).sum()),
                      suppressed_pixels_alpha_ge_half=int((b >= .5).sum()),
                      suppressed_alpha_increase_max=float((b-a).max()))
        if color_error:
            values.update(stock_foreground_rgb_mae=float(abs(rgb[region]-truth[region]).mean()),
                          suppressed_foreground_rgb_mae=float(abs(off_rgb[region]-truth[region]).mean()))
        return values

    metrics = dict(mask_interior=measure(interior, True), mask_exterior=measure(exterior, False),
                   mask_transition=measure(~(interior | exterior), False))
    box = tuple(view['crop'])
    group_image = Image.fromarray(np.rint(np.clip(contrib, 0, 1)*255).astype(np.uint8)).convert('RGB')
    off_image = Image.fromarray(np.rint(np.clip(off_rgb, 0, 1)*255).astype(np.uint8))
    panels = [item.crop(box) for item in (image, Image.fromarray(stock_bytes), group_image, off_image)]
    w, h = panels[0].size
    panel = Image.new('RGB', (4*w, h+22), '#181818')
    annotation = ImageDraw.Draw(panel)
    for i, (label, item) in enumerate(zip(('Original source', 'Full stock', 'Group alpha', 'Group suppressed'), panels)):
        annotation.text((i*w+4, 4), label, fill='white')
        panel.paste(item, (i*w, 22))
    target = directory / (row['frame'] + '-zero-owner-attribution.png')
    panel.save(target)
    return dict(frame=row['frame'], camera=actual, source_sha256=row['source_sha256'], mask_sha256=row['mask_sha256'],
                archive_rgb_max_byte=stock_error, alpha_composition_error=composition_error,
                restored_rgb_exact=True, restored_raw_opacity_exact=True, metrics=metrics,
                panel=target.name, panel_sha256=sha(target))


def replay(plan, rt, asset, binding, zero_points):
    require(read(PLAN) == plan and not (OUT / 'render').exists(), 'Preserve the preregistered original experiment.')
    with exposure.LOCK.open('a+b') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import torch
        from shuolee_portrait.bound_replay import load_bound_snapshot
        from shuolee_portrait.cli import gpu_lock

        require(torch.cuda.device_count() == 1, 'Expose exactly one GPU.')
        rt.configure(asset.parent.parent)
        import diff_gaussian_rasterization as rasterizer
        from diff_gaussian_rasterization import _C
        require(Path(rasterizer.__file__).resolve() == (ownership.RASTER_ENV / '__init__.py').resolve()
                and Path(_C.__file__).resolve() == ownership.RASTER_BINARY.resolve(),
                'Unexpected rasterizer runtime.')
        with gpu_lock(), torch.no_grad(), TemporaryDirectory(prefix='.render-', dir=OUT) as scratch:
            model, parity = load_bound_snapshot(exposure.RUN, exposure.SNAPSHOT, asset_path=asset)
            require(model.optimizer is None and np.array_equal(model.binding.cpu().numpy(), binding),
                    'Wrong replay binding or optimizer state.')
            group = torch.from_numpy(zero_points).cuda()
            staging = Path(scratch) / 'render'
            staging.mkdir()
            records = [render_frame(view, model, group, staging) for view in plan['views']]
            require(model.optimizer is None, 'Created an optimizer during read-only replay.')
            ownership.verify(plan['input_sha256'])
            ownership.save(staging / 'result.json', dict(state='complete_pending_visual_review', records=records,
                preregistration_sha256=sha(PLAN), replay_parity=parity, renders=21, fit_calls=0,
                optimizer_updates=0, model_changed=False, quality_accepted=False))
            staging.rename(OUT / 'render')
    print(OUT / 'render/result.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'prepare', 'render'))
    mode = parser.parse_args().mode
    if mode != 'render':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    plan, *state = preflight()
    if mode == 'render':
        replay(plan, *state)
    elif mode == 'prepare':
        OUT.mkdir()
        ownership.save(PLAN, plan)
        print(json.dumps(dict(preregistration=str(PLAN), sha256=sha(PLAN), gpu_calls=0)))
    else:
        if PLAN.exists():
            require(read(PLAN) == plan, 'The registered experiment changed.')
        print(json.dumps(dict(frames=FRAMES, group_points=plan['bound_group_points'], gpu_calls=0)))


if __name__ == '__main__':
    main()
