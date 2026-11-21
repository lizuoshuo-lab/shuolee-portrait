"""Evaluate the actual 3x demo run at common enlarged and native resolutions."""
from pathlib import Path
import importlib.util
import json
import shutil
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('demo_3x_experiment', HERE / 'demo_static_upsample3x.py')
experiment = importlib.util.module_from_spec(spec); spec.loader.exec_module(experiment)
ex = experiment.ex
ROOT, RUN = experiment.ROOT, experiment.RUN
RECORDS = Path('/mnt/d/00/记录/2026-11-02_人体上采样静止重建')
OUT = RECORDS / 'demo_3x'
ARMS = ['original', 'upsampled2x', 'upsampled3x']
LABELS = ['原尺寸训练', '宽高2倍训练', '宽高3倍训练']


def mean(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'frame'}


def face_box(row, margin=True):
    box = np.asarray(row['face_box_xyxy']).reshape(2, 2)
    pad = np.maximum((box[1] - box[0]) * .3, 24) if margin else np.zeros(2)
    lo = np.maximum(np.floor(box[0] - pad).astype(int), 0)
    hi = np.minimum(np.ceil(box[1] + pad).astype(int), row['wh'])
    return [int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])]


def board(paths, labels, dest, thumbnail=None, frame=''):
    font = ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc', 20)
    ims = []
    for p in paths:
        with Image.open(p) as source:
            im = source.convert('RGB')
        if thumbnail:
            im.thumbnail(thumbnail, Image.Resampling.LANCZOS)
        ims.append(im)
    width = max(240, max(im.width for im in ims))
    height = max(im.height for im in ims) + 66
    result = Image.new('RGB', (width * len(ims), height), '#eeeeee')
    draw = ImageDraw.Draw(result)
    for i, (im, label) in enumerate(zip(ims, labels)):
        draw.text((i * width + 10, 6), label, font=font, fill='black')
        result.paste(im, (i * width + (width - im.width) // 2, 62))
    draw.text((10, 33), frame, font=font, fill='black')
    result.save(dest)


def main():
    experiment.verify_source()
    result = ex.read(RUN / 'gaussian-run/result.json')
    assert result['steps'] == 18000 and result['from_scratch'] and not result['old_checkpoint_loaded']
    rt, torch = ex.runtime()
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from shuolee_portrait.training import Views, camera, image, foreground_loss
    OUT.mkdir(exist_ok=False)
    rows = ex.read(RUN / 'data/training-inputs.json')['records']
    native_rows = ex.read(ROOT / 'original/data/training-inputs.json')['records']
    actual = ex.read(RUN / 'gaussian-run/training-cameras.json')['records']
    assert len(rows) == len(native_rows) == len(actual) == 59
    for r, n, c in zip(rows, native_rows, actual):
        assert r['wh'] == c['wh'] == [1428, 2880]
        assert list(Image.open(r['image']).size) == r['wh']
        assert c['K'] == r['K'] and c['w2c'] == n['w2c']
        assert np.allclose(np.asarray(c['K']), np.diag([3, 3, 1]) @ np.asarray(n['K']), atol=1e-12, rtol=0)
    selected = np.linspace(0, 58, 6).astype(int).tolist()
    views = Views(rows, 2880, 1)
    native_views = Views(native_rows, 960, 1)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg = torch.zeros(3, device='cuda')
    previous = ex.read(RECORDS / 'demo/comparison.json')
    enlarged_metrics = {}
    native_metrics = {a: previous['metrics'][a] for a in ['original', 'upsampled2x']}
    orbit = ex.read(RECORDS / 'demo/cameras.json')['novel_review_cameras']
    models = {}
    with torch.no_grad():
        for arm in ARMS:
            src = ROOT / arm / 'gaussian-run/step-018000/world.ply'
            parity = ex.read(src.parent / 'native-ply-parity.json')
            assert parity['passed'] and parity['sha256'] == ex.sha(src)
            model = GaussianModel(3); model.load_ply(str(src)); model.active_sh_degree = 3
            models[arm] = dict(source=str(src), sha256=ex.sha(src), points=len(model.get_xyz), steps=18000)
            records = []
            for i, row in enumerate(rows):
                cam, gt, mask = views.get(i)
                pred = render(cam, model, pipe, bg)['render']
                assert list(pred.shape[1:]) == [2880, 1428]
                _, l1, ssim = foreground_loss(pred, gt, mask)
                x0, y0, x1, y1 = face_box(row, False)
                selected_face = mask[0, y0:y1, x0:x1] > .99
                error = (pred - gt).abs().mean(0)[y0:y1, x0:x1][selected_face]
                assert error.numel() > 0 and torch.isfinite(error).all()
                records.append(dict(frame=row['frame'], foreground_l1=float(l1),
                    foreground_ssim=float(ssim), face_foreground_l1=float(error.mean())))
                if i in selected or arm == 'upsampled3x':
                    folder = OUT / row['frame']; folder.mkdir(exist_ok=True)
                    im = image(pred)
                    im.save(folder / (arm + '_3x.png'))
                    im.crop(tuple(face_box(row))).save(folder / (arm + '_3x_face.png'))
                    if arm == 'upsampled3x':
                        shutil.copy2(row['image'], folder / 'input_3x.png')
                        Image.open(row['image']).crop(tuple(face_box(row))).save(folder / 'input_3x_face.png')
                        assert ex.sha(folder / 'input_3x.png') == row['source_sha256']
                del pred, gt, mask, cam, error
                if i % 15 == 0 or i == 58:
                    print('EVALUATE_3X', arm, i + 1, '/59', flush=True)
            enlarged_metrics[arm] = records
            if arm == 'upsampled3x':
                records = []
                for i, row in enumerate(native_rows):
                    cam, gt, mask = native_views.get(i)
                    pred = render(cam, model, pipe, bg)['render']
                    alpha = render(cam, model, pipe, bg, override_color=torch.ones_like(model.get_xyz))['render'][:1]
                    _, l1, ssim = foreground_loss(pred, gt, mask)
                    fg = mask[0] > .99
                    mse = (pred - gt).square().mean(0)[fg].mean()
                    x0, y0, x1, y1 = face_box(row, False)
                    face_fg = fg[y0:y1, x0:x1]
                    ferr = (pred - gt).abs().mean(0)[y0:y1, x0:x1][face_fg]
                    records.append(dict(frame=row['frame'], foreground_l1=float(l1), foreground_ssim=float(ssim),
                        foreground_psnr=float(-10 * torch.log10(mse.clamp_min(1e-12))),
                        face_foreground_l1=float(ferr.mean()), face_foreground_pixels=int(ferr.numel()),
                        alpha_mae=float((alpha - mask).abs().mean()),
                        mask_iou=float(((alpha[0] > .5) & (mask[0] > .5)).sum() / ((alpha[0] > .5) | (mask[0] > .5)).sum())))
                    if i in selected:
                        folder = OUT / row['frame']
                        image(pred).save(folder / 'upsampled3x_native.png')
                        shutil.copy2(row['image'], folder / 'input_native.png')
                    if i % 20 == 0 or i == 58:
                        print('EVALUATE_NATIVE', i + 1, '/59', flush=True)
                native_metrics[arm] = records
                (OUT / 'orbit').mkdir()
                for i, row in enumerate(orbit):
                    cam, _ = camera(row, i)
                    image(render(cam, model, pipe, bg)['render']).save(OUT / 'orbit' / (row['frame'] + '_upsampled3x.png'))
                shutil.copy2(src, OUT / 'upsampled3x.ply')
                assert ex.sha(OUT / 'upsampled3x.ply') == models[arm]['sha256']
            del model
            torch.cuda.empty_cache()
    summary = dict(updated=ex.now(), subject='demo', frames=59, selected_frames=[rows[i]['frame'] for i in selected],
        training_wh=[1428, 2880], native_wh=[476, 960], iterations=18000,
        models=models, enlarged_metrics=enlarged_metrics, enlarged_mean={a: mean(r) for a, r in enlarged_metrics.items()},
        native_metrics=native_metrics, native_mean={a: mean(r) for a, r in native_metrics.items()},
        reused_native_baseline_metrics_source='../demo/comparison.json',
        enlarged_reference='Lanczos 3x input PNG; no new high-resolution observation',
        native_reference='Original 476x960 training observations',
        all_59_training_images_saved_at_exact_3x_size=True, novel_views=12, novel_view_ground_truth=False,
        quality_accepted=False, report_script_sha256=ex.sha(__file__))
    ex.save(OUT / 'comparison.json', summary)
    ex.save(OUT / 'cameras.json', dict(convention='OpenCV world-to-camera; K uses pixel corners',
        records=[{k: r[k] for k in ['frame','source_frame_index','time_seconds','wh','K','w2c']} for r in rows],
        novel_review_cameras=orbit))
    for name in ['plan.json', 'preflight.json', 'execution.json', 'source-adjustment.json', 'projection-precision-diagnostic.json']:
        shutil.copy2(RUN / name, OUT / name)
    for i in selected:
        row = rows[i]; folder = OUT / row['frame']
        board([folder / 'input_3x_face.png'] + [folder / (a + '_3x_face.png') for a in ARMS],
              ['3倍训练输入'] + LABELS, folder / 'face_comparison.png', frame=row['frame'])
        board([folder / 'input_3x.png'] + [folder / (a + '_3x.png') for a in ARMS],
              ['3倍训练输入'] + LABELS, folder / 'body_comparison.png', thumbnail=(320, 650), frame=row['frame'])
        board([folder / 'input_native.png', RECORDS / 'demo' / row['frame'] / 'original_native.png',
               RECORDS / 'demo' / row['frame'] / 'upsampled2x_native.png', folder / 'upsampled3x_native.png'],
              ['原尺寸输入'] + LABELS, folder / 'native_comparison.png', thumbnail=(320, 650), frame=row['frame'])
    for row in orbit:
        name = row['frame']
        board([RECORDS / 'demo/orbit' / (name + '_' + a + '.png') for a in ['original', 'upsampled2x']] +
              [OUT / 'orbit' / (name + '_upsampled3x.png')], LABELS,
              OUT / 'orbit' / (name + '_comparison.png'), thumbnail=(256, 384), frame=name)
    print(json.dumps({k: summary[k] for k in ['enlarged_mean', 'native_mean', 'models']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
