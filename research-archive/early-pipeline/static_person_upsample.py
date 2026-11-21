"""A paired input-resolution control using the archived sofa static trainer."""
from pathlib import Path
import argparse
import copy
import datetime
import hashlib
import json
import os
import shutil
import sys
import time
import traceback

import numpy as np
from PIL import Image

ROOT = Path('/opt/gs-slam-lab/static-person-upsampling-20261102/gray')
SOURCE = Path('/opt/gs-slam-lab/smpl-x-alignment-20261028/grayT-7b1c0fd30822c3de-r2')
PROJECT = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
ARCHIVE = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/clothed-surface-119/source')
ASSETS = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
DELIVERY = Path('/mnt/d/00/工作日志/2026-11-02/灰衣人体上采样静止重建')
ARMS = {'original': 1, 'upsampled2x': 2}
SUBJECT = 'gray'


def select_subject(subject):
    global ROOT, SOURCE, DELIVERY, SUBJECT
    SUBJECT = subject
    ROOT = Path('/opt/gs-slam-lab/static-person-upsampling-20261102')/subject
    if subject == 'sofa':
        SOURCE = Path('/opt/gs-slam-lab/shuolee-clothed-surface-119/sofa')
    elif subject == 'demo':
        SOURCE = Path('/opt/gs-slam-lab/static-person-upsampling-20261102/demo-static-input')
    else:
        SOURCE = Path('/opt/gs-slam-lab/smpl-x-alignment-20261028/grayT-7b1c0fd30822c3de-r2')
    DELIVERY = Path('/mnt/d/00/工作日志/2026-11-02/人体上采样静止重建')/subject


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def save(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(p)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def prepare():
    ROOT.mkdir(parents=True, exist_ok=False)
    rows = read(SOURCE/'data/training-inputs.json')['records']
    boxes = []
    for row in rows:
        assert sha(row['image']) == row['source_sha256']
        assert sha(row['mask']) == row['mask_sha256']
        a = np.asarray(Image.open(row['mask']).convert('L')) > 127
        y, x = np.nonzero(a)
        boxes.append([int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)])
    b = np.asarray(boxes)
    wh = rows[0]['wh']
    assert all(r['wh'] == wh for r in rows)
    box = [max(0, int(b[:, 0].min())-32), max(0, int(b[:, 1].min())-32),
           min(wh[0], int(b[:, 2].max())+32), min(wh[1], int(b[:, 3].max())+32)]
    w, h = box[2]-box[0], box[3]-box[1]
    src = ROOT/'source'
    shutil.copytree(PROJECT/'src', src/'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (src/'third_party').symlink_to(PROJECT/'third_party', target_is_directory=True)
    for name in ['clothed_training.py', 'clothed_surface.py', 'training.py', 'gaussian_model.py', 'static_world_gaussian_model.py']:
        folder = 'avatar_core/scene' if name in ['gaussian_model.py', 'static_world_gaussian_model.py'] else 'shuolee_portrait'
        shutil.copy2(ARCHIVE/name, src/'src'/folder/name)
    pinned = {str(p.relative_to(src)): sha(p) for p in sorted((src/'src').rglob('*')) if p.is_file()}
    save(ROOT/'source-manifest.json', pinned)
    shutil.copy2(__file__, ROOT/'experiment.py')
    plan = dict(created=now(), subject=SUBJECT, source_run=str(SOURCE),
                source_video=read(SOURCE/'run.json')['video'], source_video_sha256=read(SOURCE/'run.json')['video_sha256'],
                representation='shared_static_world_clothed_surface', frames=len(rows), crop_xyxy=box,
                crop_wh=[w, h], input_source='Existing original decoded training pixels, before enhancement',
                intervention='2x Lanczos RGB upsampling and bilinear mask sampling; renderer resolution and full K scale together.',
                no_learned_superresolution=True, new_observed_detail_created=False,
                controls='Same complete-person crop, all frame IDs, extrinsics, shared static mesh, initial 200k world points, seed6060, original sofa losses and 18000-step schedule. Both Gaussian models and optimizers start fresh.',
                cached_subject_fit_reused=True, cached_initial_surface_reused=True,
                source_models_modified=False, camera_calibration_changed=False, production_deployment_changed=False,
                initial_points=200000, iterations=18000, seed=6060, max_points=650000,
                memory_cap_gib=6, evaluation='Both results rendered at the original input camera/resolution, compared with unenhanced observations. Native-video face crops shown separately; training views are not held-out views.',
                limitations=['Interpolation does not recover missing observations.',
                             'Pixel-sized SSIM windows, mask edge sampling and raster/densification effects change with resolution.',
                             'This input-resolution test does not repair or validate original cameras, staticity or SMPL-X alignment.'],
                quality_accepted=False, script_sha256=sha(__file__), source_manifest_sha256=sha(ROOT/'source-manifest.json'))
    save(ROOT/'plan.json', plan)
    errors = []
    for arm, scale in ARMS.items():
        out = ROOT/arm
        for folder in ['data/images', 'data/masks', 'fit', 'clothing-init']:
            (out/folder).mkdir(parents=True)
        cfg = read(SOURCE/'config.json')
        cfg.update(training_height=h*scale, image_cache_frames=len(rows), iterations=18000,
                   seed=6060, gaussian_representation='static_clothed_world')
        save(out/'config.json', cfg)
        for rel in ['fit/fitted.npz', 'clothing-init/points.npz', 'clothing-init/motion.npz', 'clothing-init/audit.json']:
            shutil.copy2(SOURCE/rel, out/rel)
            assert sha(SOURCE/rel) == sha(out/rel)
        adapted = []
        for row in rows:
            rgb = Image.open(row['image']).convert('RGB').crop(box)
            mask = Image.open(row['mask']).convert('L').crop(box)
            assert np.count_nonzero(np.asarray(mask)>127) == np.count_nonzero(np.asarray(Image.open(row['mask']))>127)
            if scale != 1:
                rgb = rgb.resize((w*scale, h*scale), Image.Resampling.LANCZOS)
                mask = mask.resize((w*scale, h*scale), Image.Resampling.BILINEAR)
            ip, mp = out/'data/images'/row['name'], out/'data/masks'/row['name']
            rgb.save(ip); mask.save(mp)
            r = copy.deepcopy(row)
            C = np.array([[scale, 0, -box[0]*scale], [0, scale, -box[1]*scale], [0, 0, 1.]], dtype=float)
            K = C @ np.asarray(row['K'])
            Ki = K.copy(); Ki[:2, 2] -= .5
            r.update(image=str(ip), mask=str(mp), wh=[w*scale, h*scale],
                     source_sha256=sha(ip), mask_sha256=sha(mp), K=K.tolist(), K_integer_centers=Ki.tolist(),
                     parent_image_sha256=row['source_sha256'], parent_mask_sha256=row['mask_sha256'],
                     crop_xyxy=box, upsampling_scale=scale)
            for key in ['face_box_xyxy', 'initial_box_xyxy']:
                if key in r:
                    xy = np.asarray(row[key]).reshape(2, 2)
                    r[key] = ((xy-np.array(box[:2])+.5)*scale-.5).reshape(-1).tolist()
            cp = np.array([[.1, .2, 2.], [-.4, -.2, 4.], [.3, .1, 3.]])
            q = cp @ np.asarray(row['K']).T
            expected = (q[:, :2]/q[:, 2:]-np.array(box[:2]))*scale-.5
            actual = cp @ K.T; actual = actual[:, :2]/actual[:, 2:]-.5
            error = float(np.max(np.abs(actual-expected))); errors.append(error)
            assert error < 1e-9
            adapted.append(r)
        save(out/'data/training-inputs.json', dict(records=adapted, coordinate_contract='pixel_corners',
            source=str(SOURCE/'data/training-inputs.json'), transform='crop then scale in corner coordinates',
            fresh_gaussians=True, cached_fit_and_surface=True, quality_accepted=False))
    save(ROOT/'input-verification.json', dict(created=now(), frames=len(rows), arms=2,
        crop_contains_all_original_foreground=True, world_points_and_fit_hashes_unchanged=True,
        maximum_projection_transform_error_px=max(errors), source_manifest_sha256=sha(ROOT/'source-manifest.json')))
    print(json.dumps(plan, ensure_ascii=False), flush=True)


def runtime():
    sys.path.insert(0, str(ROOT/'source/src'))
    from shuolee_portrait import runtime as rt
    rt.configure(ASSETS, ROOT)
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    free, total = torch.cuda.mem_get_info()
    cap = 6 * 1024**3
    if free < cap + 512*1024**2:
        raise RuntimeError(f'Insufficient free GPU memory for isolated control: {free/1024**3:.2f} GiB free; no existing process will be interrupted.')
    torch.cuda.set_per_process_memory_fraction(cap/total)
    os.nice(5)
    return rt, torch


def preflight():
    rt, torch = runtime()
    from types import SimpleNamespace
    from shuolee_portrait.training import Views, foreground_loss
    from shuolee_portrait.clothed_surface import make_model
    from gaussian_renderer import render
    rows = read(ROOT/'upsampled2x/data/training-inputs.json')['records']
    views = Views(rows, rows[0]['wh'][1], 1)
    model = make_model(ROOT/'upsampled2x')
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg = torch.zeros(3, device='cuda')
    checks = []
    for idx in [0, len(rows)//2, len(rows)-1]:
        torch.cuda.reset_peak_memory_stats(); start = time.monotonic()
        cam, gt, alpha = views.get(idx)
        pkg = render(cam, model, pipe, bg)
        photo, l1, ssim = foreground_loss(pkg['render'], gt, alpha)
        aa = render(cam, model, pipe, bg, override_color=torch.ones_like(model.get_xyz))['render'][:1]
        loss = photo + .1*(aa-alpha).abs().mean()
        loss.backward(); torch.cuda.synchronize()
        assert torch.isfinite(loss) and torch.isfinite(model._xyz.grad).all()
        checks.append(dict(frame=rows[idx]['frame'], wh=rows[idx]['wh'], loss=float(loss),
            nonzero_xyz_gradients=int((model._xyz.grad.norm(dim=1)>0).sum()),
            peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3, seconds=time.monotonic()-start))
        for name in ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity']:
            getattr(model, name).grad = None
        del pkg, photo, l1, ssim, aa, loss, cam, gt, alpha
    save(ROOT/'preflight.json', dict(created=now(), passed=True, optimizer_updates=0, checks=checks,
        max_projection_error_px=max(x['projection_error_px'] for x in views.records), other_processes_interrupted=False))
    print(json.dumps(read(ROOT/'preflight.json')), flush=True)


def train(arm):
    assert read(ROOT/'preflight.json')['passed']
    rt, torch = runtime()
    from shuolee_portrait.clothed_training import train_clothed
    out = ROOT/arm
    cfg = read(out/'config.json')
    state = dict(created=now(), pid=os.getpid(), arm=arm, state='training', quality_accepted=False)
    save(out/'execution.json', state)
    start = time.monotonic()
    try:
        train_clothed(out, cfg)
        state.update(state='completed_pending_visual_comparison', seconds=time.monotonic()-start,
                     peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3, finished=now())
        save(out/'execution.json', state)
    except BaseException as exc:
        state.update(state='failed', error=repr(exc), finished=now())
        save(out/'execution.json', state)
        raise


def evaluate():
    rt, torch = runtime()
    from types import SimpleNamespace
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from shuolee_portrait.training import Views, image, foreground_loss
    DELIVERY.mkdir(parents=True, exist_ok=True)
    rows = read(ROOT/'original/data/training-inputs.json')['records']
    views = Views(rows, rows[0]['wh'][1], 1)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg = torch.zeros(3, device='cuda')
    selected = set(np.linspace(0, len(rows)-1, 6).astype(int).tolist())
    metrics = {arm: [] for arm in ARMS}
    for arm in ARMS:
        source = ROOT/arm/'gaussian-run/step-018000/world.ply'
        model = GaussianModel(3); model.load_ply(str(source)); model.active_sh_degree = 3
        target = DELIVERY/(arm+'.ply'); shutil.copy2(source, target)
        assert sha(source) == sha(target)
        with torch.no_grad():
            for i, row in enumerate(rows):
                cam, gt, alpha = views.get(i)
                pred = render(cam, model, pipe, bg)['render']
                photo, l1, ss = foreground_loss(pred, gt, alpha)
                aa = render(cam, model, pipe, bg, override_color=torch.ones_like(model.get_xyz))['render'][:1]
                face = np.asarray(row['face_box_xyxy']).reshape(2, 2)
                x0, y0 = np.maximum(0, np.floor(face[0]).astype(int)); x1, y1 = np.ceil(face[1]).astype(int)
                x1, y1 = min(x1, gt.shape[2]), min(y1, gt.shape[1])
                valid = alpha[0]>.99
                err = (pred-gt).square().mean(0)[valid].mean()
                metrics[arm].append(dict(frame=row['frame'], foreground_l1=float(l1), foreground_ssim=float(ss),
                    foreground_psnr=float(-10*torch.log10(err.clamp_min(1e-12))),
                    alpha_mae=float((aa-alpha).abs().mean()),
                    face_l1=float((pred[:,y0:y1,x0:x1]-gt[:,y0:y1,x0:x1]).abs().mean()),
                    mask_iou=float(((aa[0]>.5)&(alpha[0]>.5)).sum()/((aa[0]>.5)|(alpha[0]>.5)).sum())))
                if i in selected:
                    folder = DELIVERY/row['frame']; folder.mkdir(exist_ok=True)
                    image(pred).save(folder/(arm+'_body.png'))
                    if arm == 'original': image(gt).save(folder/'input_body.png')
                    # Same native-video camera resolution, not an upscaled screenshot of a render.
                    native_scale = 2704/1600
                    cam2, _ = __import__('shuolee_portrait.training', fromlist=['camera']).camera(
                        row, i, round(row['wh'][1]*native_scale))
                    native = render(cam2, model, pipe, bg)['render']
                    image(native).save(folder/(arm+'_native_render.png'))
        del model
        torch.cuda.empty_cache()
    summary = {}
    for arm, records in metrics.items():
        summary[arm] = {k: float(np.mean([r[k] for r in records])) for k in records[0] if k != 'frame'}
    save(DELIVERY/'comparison.json', dict(updated=now(), plan=read(ROOT/'plan.json'), frames=len(rows),
        metrics=metrics, mean_metrics=summary, render_evaluation='Same original, unenhanced crop and camera for both models',
        optimizer_updates_per_arm=18000, novel_view_ground_truth=False, quality_accepted=False))
    print(json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['prepare','preflight','train','evaluate'])
    p.add_argument('--subject', choices=['gray','demo','sofa'], default='gray')
    p.add_argument('--arm', choices=list(ARMS)); args = p.parse_args()
    select_subject(args.subject)
    if args.action == 'train' and args.arm is None: p.error('--arm is required for training')
    if args.action == 'prepare': prepare()
    elif args.action == 'preflight': preflight()
    elif args.action == 'train': train(args.arm)
    else: evaluate()


if __name__ == '__main__':
    main()
