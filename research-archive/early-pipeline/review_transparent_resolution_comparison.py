"""Common-camera rendering, standard PLY exports and an inspectable three-way report."""
from pathlib import Path
from types import SimpleNamespace
from html import escape
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
import argparse
import ast
import copy
import importlib.util
import json
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/mnt/d/00/记录/2026-11-04_MSV2_原始_2倍_4倍对比')
RUNS = {
    'original': Path('/opt/gs-slam-lab/msv2-demo-transparent-native-20261104'),
    'interpolation2x': Path('/opt/gs-slam-lab/msv2-demo-transparent-interpolate2x-20261104'),
    'swinir4x': Path('/opt/gs-slam-lab/msv2-demo-transparent-full-20261104'),
}
LABELS = {'original': '原始尺寸', 'interpolation2x': 'Lanczos 插值2倍', 'swinir4x': 'SwinIR 超分4倍'}
FOUR = Path('/mnt/d/00/记录/2026-11-04_透明背景_MSV2/demo_18000')
OLD_CAMERAS = Path('/mnt/d/00/记录/人体上采样静止重建/demo_3x/cameras.json')
ZOOM_INPUTS = RUNS['interpolation2x'] / 'training-inputs.json'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def helper():
    return load('portable_review', Path(__file__).with_name('review_msv2_upsampled_demo.py'))


def export(label):
    h = helper(); ex = load('export_experiment', RUNS[label] / 'experiment.py')
    torch = ex.runtime()
    from scene import GaussianModel
    run = RUNS[label]; out = ROOT / label
    assert ex.read(run / 'execution.json')['state'] == 'training_completed'
    final = run / 'model/point_cloud/iteration_18000/point_cloud.ply'
    assert ex.read(final.parent / 'ply-parity.json')['passed']
    out.mkdir(parents=True, exist_ok=False)
    h.ex = ex; h.RUN = run; h.OUT = out
    h.PLY_STEM = f'demo_{label}_msv2_transparent_18000'
    h.BUNDLE = f'demo_{LABELS[label].replace(" ", "")}_MSV2_透明训练_18000步_点云与相机.zip'
    for f in ['plan.json', 'config.json', 'execution.json', 'input-contract.json',
              'preflight.json', 'rgba-probe.json', 'configuration-parity.json']:
        if (run / f).exists():
            shutil.copy2(run / f, out / f)
    shutil.copy2(final, out / (h.PLY_STEM + '_world.ply'))
    shutil.copy2(final.parent / 'ply-parity.json', out / 'native-ply-parity.json')
    rows = ex.read(run / 'training-inputs.json')['records']
    ex.save(out / 'cameras_world.json', h.camera_records(rows))
    ex.save(out / 'camera-input-mapping.json', dict(records=rows,
        convention='Pixel-corner K; integer-center principal point is cx-.5, cy-.5',
        world='Original scene-background SfM coordinates', static_model=True))
    model = GaussianModel(3); model.load_ply(str(final)); model.active_sh_degree = 3
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    h.export_upright(rows, model, pipe, torch.zeros(3, device='cuda'))
    ex.save(out / 'delivery.json', dict(bundle=h.BUNDLE, world_ply=h.PLY_STEM + '_world.ply',
        upright_ply=h.PLY_STEM + '_yup.ply', sha256=ex.sha(out / h.BUNDLE),
        bytes=(out / h.BUNDLE).stat().st_size, cameras=59, points=len(model.get_xyz),
        export_verified=True, geometry_quality_accepted=False))
    print('EXPORTED', label, len(model.get_xyz), flush=True)


def foreground_metric(ex):
    source = ex.BASE / 'source/src/shuolee_portrait/training.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'foreground_loss')
    ns = {}; exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), ns)
    return ns['foreground_loss'], ex.sha(source)


def evaluate():
    ex = load('comparison_experiment', next(iter(RUNS.values())) / 'experiment.py')
    torch = ex.runtime(); h = helper()
    from scene import GaussianModel
    from gaussian_renderer import render
    models = {}; frozen = {}; executions = {}
    for label, run in RUNS.items():
        execution = ex.read(run / 'execution.json')
        assert execution['state'] == 'training_completed' and execution['iterations'] == 18000
        path = run / 'model/point_cloud/iteration_18000/point_cloud.ply'
        assert ex.read(path.parent / 'ply-parity.json')['passed']
        frozen[label] = dict(path=str(path), sha256=ex.sha(path))
        m = GaussianModel(3); m.load_ply(str(path)); m.active_sh_degree = 3
        models[label] = m; executions[label] = execution
    photo, metric_sha = foreground_metric(ex)
    rows = ex.read(next(iter(RUNS.values())) / 'native-inputs.json')['records']
    zoom_rows = ex.read(ZOOM_INPUTS)['records']
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg = torch.zeros(3, device='cuda'); checks = []; compositing = []
    with torch.no_grad():
        for i, row in enumerate(rows):
            folder = ROOT / 'views' / row['frame']; folder.mkdir(parents=True, exist_ok=False)
            shutil.copy2(row['image'], folder / 'source.png')
            cam = ex.Camera(row, i)
            gt = torch.as_tensor(cam.rgb.copy(), dtype=torch.float32, device='cuda').permute(2, 0, 1) / 255
            mask = torch.as_tensor(cam.mask.copy(), dtype=torch.float32, device='cuda')[None] / 255
            yy, xx = torch.meshgrid(torch.arange(cam.height, device='cuda'), torch.arange(cam.width, device='cuda'), indexing='ij')
            checker = torch.where((xx // 24 + yy // 24) % 2 == 0, .85, .45)[None]
            entry = dict(frame=row['frame']); outside = mask[0] < .01; inside = mask[0] > .99
            box = h.face_box(row, False); x0, y0, x1, y1 = box
            face_mask = inside[y0:y1, x0:x1]
            bx0, by0, bx1, by1 = h.face_box(row)
            face_region = torch.zeros_like(inside)
            face_region[by0:by1, bx0:bx1] = True
            face_exterior = face_region & outside
            face_interior = face_region & inside
            for label, model in models.items():
                c = render(cam, model, pipe, bg)['render']
                alpha = render(cam, model, pipe, bg, override_color=torch.ones_like(model.get_xyz))['render'][0:1]
                _, l1, ssim = photo(c, gt, mask)
                straight = c / alpha.clamp_min(1e-4); a = alpha[0]
                dark = (a > .5) & (straight.clamp(0, 1).mean(0) < .15)
                face_error = (c - gt).abs().mean(0)[y0:y1, x0:x1][face_mask]
                entry[label] = dict(foreground_l1=float(l1), foreground_ssim=float(ssim),
                    face_foreground_l1=float(face_error.mean()), outside_alpha_mean=float(a[outside].mean()),
                    outside_alpha_gt_01=float((a[outside] > .1).float().mean()),
                    outside_dark_opaque_fraction=float(dark[outside].float().mean()),
                    inside_alpha_mean=float(a[inside].mean()), inside_alpha_gt_05=float((a[inside] > .5).float().mean()),
                    mask_iou=float(((a > .5) & (mask[0] > .5)).sum() / ((a > .5) | (mask[0] > .5)).sum()),
                    face_exterior_alpha_mean=float(a[face_exterior].mean()),
                    face_exterior_gt_01=float((a[face_exterior] > .1).float().mean()),
                    face_inside_alpha_mean=float(a[face_interior].mean()))
                for name, back in [('gray', .5), ('white', 1.), ('black', 0.), ('checker', checker)]:
                    ex.save_image(folder / f'{label}_{name}.png', c + (1 - alpha) * back)
                rgba = torch.cat([straight.clamp(0, 1), alpha.clamp(0, 1)], dim=0)
                Image.fromarray(np.rint(rgba.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'), 'RGBA').save(folder / f'{label}_transparent.png')
                if i in ex.SELECTED:
                    direct = render(cam, model, pipe, torch.ones(3, device='cuda'))['render']
                    err = (direct - (c + 1 - alpha)).abs()
                    compositing.append(dict(frame=row['frame'], method=label, max_abs=float(err.max()), mean_abs=float(err.mean())))
                del c, alpha, straight, rgba
            paths = [folder / 'source.png'] + [folder / f'{k}_gray.png' for k in models]
            labels = ['原视频人物'] + list(LABELS.values())
            h.board(paths, labels, folder / 'body_compare.png', size=(300,605), title=row['frame'])
            for back in ['white', 'checker', 'black']:
                h.board([folder / 'source.png'] + [folder / f'{k}_{back}.png' for k in models], labels,
                    folder / f'{back}_compare.png', size=(300,605), title=row['frame'])
            # Common 2x rendering for facial inspection; source crop is enlarged only for display.
            zoom = copy.deepcopy(zoom_rows[i]); zoomcam = ex.Camera(zoom, i)
            face = h.face_box(zoom)
            source_face = folder / 'source_face.png'
            with Image.open(zoom['image']) as im: im.crop(face).save(source_face)
            face_paths = [source_face]
            for label, model in models.items():
                c = render(zoomcam, model, pipe, bg)['render']
                a = render(zoomcam, model, pipe, bg, override_color=torch.ones_like(model.get_xyz))['render'][0:1]
                pixels = np.rint((c + .5 * (1 - a)).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                p = folder / f'{label}_face.png'; Image.fromarray(pixels).crop(face).save(p); face_paths.append(p)
            h.board(face_paths, ['原图插值显示'] + list(LABELS.values()), folder / 'face_compare.png', size=(260,310), title=row['frame'])
            checks.append(entry)
            if i % 10 == 0 or i == 58: print('COMMON_CAMERA_REVIEW', i + 1, '/59', flush=True)
            del cam, zoomcam, gt, mask
        orbits = ex.read(OLD_CAMERAS)['novel_review_cameras']
        orbit_out = ROOT / 'orbit'; orbit_out.mkdir(exist_ok=False)
        for i, row in enumerate(orbits):
            r = copy.deepcopy(row); dummy = orbit_out / '_canvas.png'
            Image.new('RGB', tuple(r['wh'])).save(dummy)
            r.update(image=str(dummy), mask=str(dummy)); cam = ex.Camera(r, i)
            for label, model in models.items():
                c = render(cam, model, pipe, bg)['render']
                a = render(cam, model, pipe, bg, override_color=torch.ones_like(model.get_xyz))['render'][0:1]
                ex.save_image(orbit_out / f'{r["frame"]}_{label}.png', c + .5 * (1 - a))
            dummy.unlink()
            h.board([orbit_out / f'{r["frame"]}_{label}.png' for label in models], list(LABELS.values()),
                orbit_out / (r['frame'] + '_compare.png'), size=(256,384), title=r['frame'])
    assert all(ex.sha(f['path']) == f['sha256'] for f in frozen.values())
    assert all(x['max_abs'] < .005 for x in compositing)
    means = {label: {key: float(np.mean([r[label][key] for r in checks])) for key in checks[0][label]} for label in models}
    ex.save(ROOT / 'comparison.json', dict(created=ex.now(), frames=59, common_reference_wh=[476,960],
        common_face_render_wh=[952,1920], checks=checks, means=means, executions=executions,
        models=frozen, models_unchanged=True, learned_states_reused=False, native_metric_source_sha256=metric_sha,
        training_views_not_heldout=True, novel_views=12, novel_views_have_no_ground_truth=True,
        four_x_is_learned_SR_and_two_x_is_interpolation=True, quality_accepted=False,
        dark_pixel_definition='Among mask<0.01 pixels, alpha>0.5 and mean straight RGB<0.15; a pixel fraction, not a Gaussian count.',
        compositor_passed=True, compositor_checks=compositing))
    print('COMPARISON_MEANS', json.dumps(means), flush=True)


def publish():
    ex = load('publish_experiment', RUNS['original'] / 'experiment.py')
    c = ex.read(ROOT / 'comparison.json')
    f = ex.read(ROOT / 'review-findings.json') if (ROOT / 'review-findings.json').exists() else {}
    font = ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc', 18)
    for kind in ['body', 'face', 'white', 'checker', 'black']:
        for begin in range(1, 60, 12):
            ids = list(range(begin, min(begin + 12, 60)))
            canvas = Image.new('RGB', (1600, ((len(ids) + 1) // 2) * 260), '#eee')
            d = ImageDraw.Draw(canvas)
            for j, i in enumerate(ids):
                n = f'frame_{i:06d}'
                with Image.open(ROOT / 'views' / n / (kind + '_compare.png')) as im:
                    im = im.convert('RGB'); im.thumbnail((790,225), Image.Resampling.LANCZOS)
                x = j % 2 * 800; y = j // 2 * 260
                d.text((x + 4, y), n, font=font, fill='black'); canvas.paste(im, (x, y + 28))
            canvas.save(ROOT / f'all_{kind}_{(begin - 1)//12 + 1:02d}.png')
    for first in [0, 180]:
        canvas = Image.new('RGB', (1536, 1332), '#eee')
        for j, a in enumerate(range(first, first + 180, 30)):
            with Image.open(ROOT / 'orbit' / f'orbit_{a:03d}_compare.png') as im:
                canvas.paste(im, ((j % 2) * 768, (j // 2) * 444))
        canvas.save(ROOT / f'all_orbit_{first:03d}.png')
    tr = []
    for key, label in [('foreground_l1', '原图人物L1 ↓'), ('foreground_ssim', '原图人物SSIM ↑'),
                       ('face_foreground_l1', '原图脸部L1 ↓'), ('outside_alpha_mean', '人物外平均Alpha ↓'),
                       ('outside_dark_opaque_fraction', '人物外黑色不透明像素比例 ↓'), ('mask_iou', '轮廓IoU ↑')]:
        tr.append('<tr><td>' + label + '</td>' + ''.join(f'<td>{c["means"][m][key]:.5f}</td>' for m in RUNS) + '</tr>')
    links = []
    for label in ['original', 'interpolation2x']:
        d = ex.read(ROOT / label / 'delivery.json')
        links.append(f'<p><a href="{label}/{d["bundle"]}">{LABELS[label]}：PLY＋59台相机</a> · <a href="{label}/{d["upright_ply"]}">独立PLY</a></p>')
    four_relative = '../2026-11-04_透明背景_MSV2/demo_18000/'
    links.append(f'<p><a href="{four_relative}demo_SwinIR4倍_MSV2_透明训练_18000步_点云与相机.zip">SwinIR4倍：PLY＋59台相机</a></p>')
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>demo · MSV2 原始 / 2倍 / 4倍</title>',
        '<style>body{max-width:1300px;margin:32px auto;padding:0 20px;background:#f5f5f2;color:#222;font:16px/1.7 system-ui}section{padding:24px;background:white;border-radius:12px;margin:20px 0}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}a{color:#176684}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(500px,1fr));gap:20px}</style>',
        '<h1>demo · MSV2 原始 / 2倍 / 4倍</h1><p>统一透明监督 · 59帧 · 同一相机外参 · 每版独立初始化 · 各18,000步</p>',
        '<section><h2>对比结论</h2>' + ''.join('<p>' + escape(p) + '</p>' for p in f.get('summary_paragraphs', ['训练、导出及同相机渲染已完成，完整视觉审阅进行中。'])) + '</section>',
        '<section><h2>三组输入</h2><table><tr><th>版本</th><th>训练分辨率</th><th>方法</th></tr><tr><td>原始</td><td>476×960</td><td>原始人物裁切</td></tr><tr><td>2倍</td><td>952×1920</td><td>Lanczos插值，遮罩双线性放大</td></tr><tr><td>4倍</td><td>1904×3840</td><td>SwinIR超分，遮罩双线性放大</td></tr></table><p>内参前两行随倍率缩放，外参保持一致。使用相同未训练200,000点、种子6060、透明损失与MSV2时序。4倍同时改变分辨率和预测纹理，不能将差异单独归因于倍率。</p><p>下方整人图与指标统一用476×960原图相机；脸部统一用952×1920相机重新渲染，原图仅为便于观察插值显示。未经原图支持的超分细节不当作真值。</p></section>',
        '<section><h2>点云和相机</h2>' + ''.join(links) + '<p>每包是通用3DGS PLY和59台对应相机；保留透明度，导出经过原生重读及+Y坐标、协方差、球谐旋转检查。</p></section>',
        '<section><h2>统一原图指标</h2><table><tr><th>指标</th><th>原始</th><th>插值2倍</th><th>超分4倍</th></tr>' + ''.join(tr) + '</table><p>统计全部59个训练视角，非留出评测。黑色污染是人物遮罩外的黑色不透明像素比例，并非高斯数量；手持物保留原遮罩。环绕视角没有实拍真值。</p><p><a href="comparison.json">完整逐帧指标</a> · <a href="comparison-plan.json">固定条件</a></p></section>']
    for i in [1, 12, 24, 28, 35, 47, 59]:
        n = f'frame_{i:06d}'
        parts.append(f'<section><h2>{n}</h2><a href="views/{n}/body_compare.png"><img src="views/{n}/body_compare.png"></a><a href="views/{n}/face_compare.png"><img src="views/{n}/face_compare.png"></a><p><a href="views/{n}/white_compare.png">白底</a> · <a href="views/{n}/checker_compare.png">棋盘底</a> · <a href="views/{n}/black_compare.png">黑底</a></p></section>')
    parts.append('<section><h2>全部59帧，逐张原图与渲染</h2><div class="grid">')
    for i in range(1, 60):
        n = f'frame_{i:06d}'
        parts.append(f'<div><a href="views/{n}/body_compare.png">{n} 整人对比</a> · <a href="views/{n}/face_compare.png">脸部</a> · <a href="views/{n}/source.png">原图</a><img loading="lazy" src="views/{n}/body_compare.png"><p>' + ' · '.join(f'<a href="views/{n}/{k}_transparent.png">{LABELS[k]}透明PNG</a>' for k in RUNS) + '</p></div>')
    parts.append('</div></section><section><h2>同相机12个环绕视角</h2><div class="grid">')
    for a in range(0, 360, 30):
        parts.append(f'<a href="orbit/orbit_{a:03d}_compare.png"><img loading="lazy" src="orbit/orbit_{a:03d}_compare.png"></a>')
    parts.append('</div></section></html>')
    html = ''.join(parts); (ROOT / '查看对照.html').write_text(html, encoding='utf-8')
    class Links(HTMLParser):
        def __init__(self): super().__init__(); self.links = []
        def handle_starttag(self, tag, attrs): self.links.extend(v for k, v in attrs if k in ('href', 'src'))
    p = Links(); p.feed(html)
    for link in p.links:
        u = urlsplit(link)
        if not u.scheme and u.path: assert (ROOT / unquote(u.path)).is_file(), link
    count = 0
    for path in ROOT.rglob('*.png'):
        with Image.open(path) as im: im.verify()
        count += 1
    ex.save(ROOT / 'archive-verification.json', dict(passed=True, PNGs=count, links=len(p.links)))
    print('REPORT_PUBLISHED', ROOT / '查看对照.html', count, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['export', 'evaluate', 'publish'])
    parser.add_argument('--method', choices=['original', 'interpolation2x'])
    args = parser.parse_args()
    if args.action == 'export': export(args.method)
    else: globals()[args.action]()
