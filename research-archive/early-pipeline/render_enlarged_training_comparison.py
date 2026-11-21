"""Render both static models at the actual enlarged training input resolution."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import shutil
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DELIVERY = Path('/mnt/d/00/记录/2026-11-02_人体上采样静止重建')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main(subject, native_input=False):
    base = Path('/opt/gs-slam-lab/static-person-native-upsampling-20261102' if native_input else '/opt/gs-slam-lab/static-person-upsampling-20261102')
    root = base / subject
    out = DELIVERY / (subject + '_native' if native_input else subject)
    out.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('paired_static_enlarged', root / 'experiment.py')
    ex = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)
    ex.select_subject(subject)
    rt, torch = ex.runtime()
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from shuolee_portrait.training import Views, image, foreground_loss
    rows = read(root / 'upsampled2x/data/training-inputs.json')['records']
    original_rows = read(root / 'original/data/training-inputs.json')['records']
    chosen = np.linspace(0, len(rows) - 1, 6).astype(int).tolist()
    wh = rows[0]['wh']; original_wh = original_rows[0]['wh']
    assert wh == [v * 2 for v in original_wh]
    for arm in ['original', 'upsampled2x']:
        cfg = read(root / arm / 'gaussian-run/config.json')
        rr = read(root / arm / 'data/training-inputs.json')['records']
        cc = read(root / arm / 'gaussian-run/training-cameras.json')['records']
        assert len(rr) == len(cc) == len(rows)
        for row, cam in zip(rr, cc):
            assert row['wh'] == cam['wh'] == list(Image.open(row['image']).size)
            assert row['wh'][1] == cfg['training_height']
        assert read(root / arm / 'gaussian-run/result.json')['steps'] == 18000
    views = Views(rows, wh[1], 1)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg = torch.zeros(3, device='cuda')
    metrics = {}
    for arm in ['original', 'upsampled2x']:
        src = root / arm / 'gaussian-run/step-018000/world.ply'
        assert read(src.parent / 'native-ply-parity.json')['passed']
        model = GaussianModel(3); model.load_ply(str(src)); model.active_sh_degree = 3
        records = []
        with torch.no_grad():
            for i, row in enumerate(rows):
                cam, gt, mask = views.get(i)
                pred = render(cam, model, pipe, bg)['render']
                assert list(pred.shape[1:]) == list(gt.shape[1:]) == [wh[1], wh[0]]
                _, l1, ssim = foreground_loss(pred, gt, mask)
                box = np.asarray(row['face_box_xyxy']).reshape(2, 2)
                lo = np.maximum(np.floor(box[0]).astype(int), 0)
                hi = np.minimum(np.ceil(box[1]).astype(int), wh)
                fg = mask[0] > .99
                face_fg = fg[lo[1]:hi[1], lo[0]:hi[0]]
                err = (pred - gt).abs().mean(0)[lo[1]:hi[1], lo[0]:hi[0]][face_fg]
                assert err.numel() > 0
                records.append(dict(frame=row['frame'], foreground_l1=float(l1), foreground_ssim=float(ssim), face_foreground_l1=float(err.mean())))
                if i in chosen:
                    f = out / row['frame']; f.mkdir(exist_ok=True)
                    im = image(pred); im.save(f / (arm + '_enlarged.png'))
                    margin = np.maximum((box[1] - box[0]) * .3, 24)
                    fb = np.r_[np.floor(box[0] - margin), np.ceil(box[1] + margin)].astype(int)
                    fb[:2] = np.maximum(fb[:2], 0); fb[2:] = np.minimum(fb[2:], wh)
                    im.crop(tuple(fb)).save(f / (arm + '_enlarged_face.png'))
                    if arm == 'original':
                        shutil.copy2(row['image'], f / 'input_enlarged.png')
                        assert ex.sha(f / 'input_enlarged.png') == row['source_sha256']
                        Image.open(row['image']).crop(tuple(fb)).save(f / 'input_enlarged_face.png')
                        save(f / 'enlarged-image-record.json', dict(frame=row['frame'], source_frame_index=row['source_frame_index'],
                            original_training_wh=original_wh, enlarged_training_wh=wh, enlarged_render_wh=wh,
                            training_input_sha256=row['source_sha256'], training_input_is_saved_png=True,
                            actual_training_camera_wh=wh, renderer_resizes_input=False, face_crop_xyxy=fb.tolist()))
                if i % 20 == 0: print(subject, arm, 'enlarged evaluation', i + 1, '/', len(rows), flush=True)
        metrics[arm] = records
        del model; torch.cuda.empty_cache()
    keys = ['foreground_l1', 'foreground_ssim', 'face_foreground_l1']
    means = {a: {k: float(np.mean([r[k] for r in metrics[a]])) for k in keys} for a in metrics}
    save(out / 'enlarged-evaluation.json', dict(subject=subject, frames=len(rows), original_training_wh=original_wh,
        enlarged_training_wh=wh, common_render_wh=wh, no_training_downsampling=True,
        evaluation_target='Actual 2x enlarged PNG used for training, not newly observed high-resolution ground truth.',
        metrics=metrics, mean_metrics=means, quality_accepted=False, steps_each=18000))
    font = ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc', 20)
    panels = []; faces = []; bodies = []
    for i in chosen:
        row = rows[i]; f = out / row['frame']
        ims = [Image.open(f / (a + '_enlarged_face.png')).convert('RGB') for a in ['input', 'original', 'upsampled2x']]
        w = max(250, max(im.width for im in ims)); h = max(im.height for im in ims) + 62
        face = Image.new('RGB', (w * 3, h), '#eeeeee'); d = ImageDraw.Draw(face)
        for j, (label, im) in enumerate(zip(['放大后的训练输入', '原尺寸训练结果', '放大后训练结果'], ims)):
            d.text((j * w + 10, 5), label, font=font, fill='black'); face.paste(im, (j * w + (w - im.width) // 2, 60))
        d.text((10, 31), row['frame'], font=font, fill='black')
        face.save(f / 'enlarged_face_comparison.png'); faces.append(face)
        body = Image.new('RGB', (1080, 760), '#eeeeee'); d = ImageDraw.Draw(body)
        for j, (label, a) in enumerate(zip(['放大后的训练输入', '原尺寸训练结果', '放大后训练结果'], ['input', 'original', 'upsampled2x'])):
            im = Image.open(f / (a + '_enlarged.png')).convert('RGB'); im.thumbnail((360, 700), Image.Resampling.LANCZOS)
            d.text((j * 360 + 10, 5), label, font=font, fill='black'); body.paste(im, (j * 360 + (360 - im.width) // 2, 55))
        body.save(f / 'enlarged_body_comparison.png'); bodies.append(body)
        frame = row['frame']
        panels.append(f'<section><h2>{frame} · {row["time_seconds"]:.3f} 秒</h2><p><a href="{frame}/input_enlarged.png">放大后训练输入 {wh[0]}×{wh[1]}</a> · <a href="{frame}/original_enlarged.png">原尺寸训练模型，同分辨率渲染</a> · <a href="{frame}/upsampled2x_enlarged.png">放大后训练模型，完整渲染</a> · <a href="{frame}/input_source_frame.png">原视频完整帧</a></p><p class="note">下图为页面预览；链接中的 PNG 保持 {wh[0]}×{wh[1]} 完整尺寸。</p><img src="{frame}/enlarged_body_comparison.png"><img src="{frame}/enlarged_face_comparison.png"></section>')
    def contact(images, path):
        board = Image.new('RGB', (max(im.width for im in images), sum(im.height for im in images)), '#eeeeee'); y = 0
        for im in images: board.paste(im, (0, y)); y += im.height
        board.save(path)
    contact(faces, out / 'enlarged_face_contact.png')
    contact(bodies[:3], out / 'enlarged_body_contact_0.png'); contact(bodies[3:], out / 'enlarged_body_contact_1.png')
    title = ('沙发' if subject == 'sofa' else 'demo') + '：先放大，再训练'
    cells = ''.join(f'<tr><td>{label}</td><td>{means["original"][key]:.5f}</td><td>{means["upsampled2x"][key]:.5f}</td></tr>' for key, label in [('foreground_l1', '人体误差 ↓'), ('face_foreground_l1', '脸部误差 ↓'), ('foreground_ssim', 'SSIM ↑')])
    desc = '原视频直接裁切，保留原始像素，再放大2倍。' if native_input else '使用原视频分辨率的人体裁切图，再放大2倍。'
    page = f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>body{{font:17px/1.65 system-ui,sans-serif;max-width:1150px;margin:28px auto;padding:0 20px;background:#f2f4f6;color:#1c2833}}section{{background:white;border-radius:10px;padding:20px;margin:24px 0}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse}}td,th{{padding:10px 16px;border:1px solid #ccd1d8}}a{{color:#1763ad}}.note{{color:#5d6873}}</style><body><a href="../查看对照.html">返回总览</a><h1>{title}</h1><p>{desc}</p><p><strong>{original_wh[0]}×{original_wh[1]} 人体图 → {wh[0]}×{wh[1]} 放大图 → 按 {wh[0]}×{wh[1]} 训练18,000步</strong></p><p>放大后宽高各为2倍，像素数为4倍。已核对保存的输入PNG、训练相机和训练快照，训练期间保持放大尺寸。使用Lanczos插值。</p><p>两组模型均在相同的放大分辨率渲染。以下测量对照的是实际放大训练图；它不提供新的真实高分辨率观测。</p><table><tr><th>全部{len(rows)}帧</th><th>原尺寸训练</th><th>放大后训练</th></tr>{cells}</table><p><a href="enlarged-evaluation.json">测量与尺寸记录</a> · <a href="查看对照.html">原视频分辨率对照与新视角</a></p>{"".join(panels)}</body></html>'
    (out / '放大后训练.html').write_text(page, encoding='utf-8')
    print(json.dumps({'subject': subject, 'wh': wh, 'mean_metrics': means}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--subject', required=True, choices=['demo', 'sofa'])
    parser.add_argument('--native-input', action='store_true')
    args = parser.parse_args(); main(args.subject, args.native_input)
