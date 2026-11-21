"""Read-only diagnosis of first-frame clothing artifacts, with unchanged geometry.

Rendering interventions are diagnostic, not retrained models or a new test set.
The original checkpoints, fixed ROIs, and all four held-out views are preserved.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import gc
import hashlib
import json
import subprocess
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

LAB = Path(__file__).resolve().parents[2]
BASE = LAB/'outputs/dynamic-human-reconstruction-003'
OUT = LAB/'outputs/dynamic-human-first-frame-diagnosis-007'
CHECKPOINTS = [BASE/f'dynamic/initial-{n:05d}.pt' for n in [2000, 4000, 6000, 8000, 10000]]
SHIRT_ROI = [275, 200, 365, 310]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, data):
    p.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf8')


def quantiles(values):
    a = np.asarray(values)
    return dict(zip(['min', 'p01', 'p10', 'p50', 'p90', 'p99', 'max'],
                    np.quantile(a, [0, .01, .1, .5, .9, .99, 1]).tolist()))


def inspect_parameters():
    rows = []
    for path in CHECKPOINTS:
        p = torch.load(path, map_location='cpu', weights_only=True)
        assert all(torch.isfinite(v).all() for v in p.values())
        fg = p['seg_colors'][:, 0].numpy() > .5
        c = p['rgb_colors'].numpy()
        opacity = p['logit_opacities'].sigmoid().numpy()[:, 0]
        scale = p['log_scales'].exp().numpy()
        row = dict(checkpoint=path.name, sha256=sha(path), points=len(fg), foreground_points=int(fg.sum()))
        for name, mask in [('foreground', fg), ('background', ~fg)]:
            row[name] = dict(rgb_component_quantiles=quantiles(c[mask]),
                fraction_points_any_color_outside_01=float(np.any((c[mask] < 0) | (c[mask] > 1), axis=1).mean()),
                opacity_quantiles=quantiles(opacity[mask]),
                scale_max_quantiles_scene_units=quantiles(scale[mask].max(1)),
                anisotropy_quantiles=quantiles(scale[mask].max(1)/scale[mask].min(1)))
        row['training_camera_gain_quantiles'] = quantiles(p['cam_m'][:27].exp().numpy())
        row['training_camera_bias_quantiles'] = quantiles(p['cam_c'][:27].numpy())
        rows.append(row)
    report = dict(status='parameter_audit_complete_rendering_pending', checkpoints=rows,
        interpretation='Unbounded Gaussian colors can compensate each other. Their existence alone does not establish the cause of visible artifacts.')
    save(OUT/'parameter-audit.json', report)
    print(json.dumps(dict(parameter_audit='complete', final_foreground=rows[-1]['foreground']), indent=2), flush=True)


def wait_gpu():
    deadline = time.monotonic()+4*3600
    while True:
        free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()[0])
        if free >= 3200:
            print('GPU CAPACITY AVAILABLE', free, flush=True)
            return
        if time.monotonic() > deadline:
            raise TimeoutError('No safe GPU capacity for diagnostic renders; no existing jobs were interrupted')
        print('WAIT GPU FREE MiB', free, flush=True)
        time.sleep(30)


def make_panel(arrays, labels, path, footer):
    from evaluate_benchmark import png
    h, w = arrays[0].shape[:2]
    panel = Image.new('RGB', (w*len(arrays), h+58), 'white')
    d = ImageDraw.Draw(panel)
    f = ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc', 17)
    small = ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc', 13)
    for i, (a, label) in enumerate(zip(arrays, labels)):
        panel.paste(png(a), (i*w, 29))
        d.text((i*w+w/2, 3), label, font=f, anchor='mt', fill='#183747')
    d.text((6, h+35), footer, font=small, fill='#183747')
    panel.save(path)


def render_diagnostics(protocol):
    from adapter import DATA, load_official, dataset
    from evaluate_benchmark import metric, png
    wait_gpu()
    torch.cuda.set_per_process_memory_fraction(.12)
    tr, helpers = load_official()
    md = json.loads((DATA/'basketball/test_meta.json').read_text())
    assets = OUT/'assets'
    assets.mkdir(exist_ok=True)
    rows = []
    for path in CHECKPOINTS:
        assert sha(path) == next(x['sha256'] for x in protocol['checkpoints'] if x['path'] == str(path.relative_to(LAB)))
        p = {k:v.cuda() for k, v in torch.load(path, map_location='cpu', weights_only=True).items()}
        for view in dataset(0, md):
            cid = int(view['filename'].split('/')[0])
            gt = view['im'].permute(1, 2, 0).cpu().numpy()
            rv = helpers.params2rendervar(p)
            raw, _, alpha = tr.Renderer(view['cam'])(**rv)
            rgb_original = raw.permute(1, 2, 0).cpu().numpy()
            rv_clamp = {**rv, 'colors_precomp':p['rgb_colors'].clamp(0, 1)}
            clamped, _, alpha_clamp = tr.Renderer(view['cam'])(**rv_clamp)
            assert torch.equal(alpha, alpha_clamp), 'Color intervention must not alter geometry/opacity'
            label_render, _, alpha_seg = tr.Renderer(view['cam'])(**{**rv, 'colors_precomp':p['seg_colors']})
            assert torch.equal(alpha, alpha_seg)
            fg_contribution = label_render[0].cpu().numpy()
            total_alpha = alpha.cpu().numpy()
            np.testing.assert_allclose(label_render[0].cpu().numpy()+label_render[2].cpu().numpy(), total_alpha, atol=2e-5)
            raw_component_outside = np.any((rgb_original < 0) | (rgb_original > 1), axis=2)
            rgb_original = rgb_original.clip(0, 1)
            rgb_clamped = clamped.permute(1, 2, 0).cpu().numpy().clip(0, 1)
            roi = protocol['fixed_human_rois_xyxy'][str(cid)]
            x0, y0, x1, y1 = roi
            rect = (slice(y0, y1), slice(x0, x1))
            row = dict(checkpoint=path.name, camera=cid,
                original=dict(full_image=metric(rgb_original, gt), person_region=metric(rgb_original[rect], gt[rect])),
                color_clipped=dict(full_image=metric(rgb_clamped, gt), person_region=metric(rgb_clamped[rect], gt[rect])),
                raw_render_outside_01_fraction=float(raw_component_outside[rect].mean()),
                alpha_identical_across_interventions=True)
            if cid == 15:
                a,b,c,d = SHIRT_ROI
                sr = (slice(b,d), slice(a,c))
                row['source_defined_shirt_interior'] = dict(roi=SHIRT_ROI,
                    original=metric(rgb_original[sr], gt[sr]), color_clipped=metric(rgb_clamped[sr], gt[sr]),
                    total_alpha_quantiles=quantiles(total_alpha[sr]),
                    foreground_contribution_quantiles=quantiles(fg_contribution[sr]),
                    fraction_foreground_contribution_below_half=float((fg_contribution[sr]<.5).mean()),
                    fraction_foreground_contribution_below_09=float((fg_contribution[sr]<.9).mean()))
            rows.append(row)
            folder = OUT/'renders'/path.stem/str(cid)
            folder.mkdir(parents=True, exist_ok=True)
            png(gt).save(folder/'observation.png')
            png(rgb_original).save(folder/'original.png')
            png(rgb_clamped).save(folder/'color-clipped.png')
            png(np.repeat(fg_contribution[...,None],3,axis=2)).save(folder/'foreground-contribution.png')
            np.savez_compressed(folder/'coverage.npz', foreground_contribution=fg_contribution, total_alpha=total_alpha)
            make_panel([gt[rect], rgb_original[rect], rgb_clamped[rect], np.repeat(fg_contribution[rect][...,None],3,axis=2)],
                ['原始观测', '原始渲染', '仅限制高斯颜色范围', '前景不透明度贡献'],
                assets/f'{path.stem}-camera{cid}.png', f'相机 {cid} · 第一帧 · {path.stem} · 灰度越亮前景贡献越大；后处理诊断，非训练改善')
            print('FIRST FRAME DIAGNOSTIC', path.name, cid, flush=True)
        del p, rv, rv_clamp, raw, clamped, label_render, view
        gc.collect()
        torch.cuda.empty_cache()
    save(OUT/'render-diagnostics.json', dict(status='rendered_visual_review_pending',
        protocol_sha256=sha(OUT/'diagnostic-protocol.json'), implementation_sha256=sha(Path(__file__)),
        max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), per_view=rows,
        limitations=protocol['limitations']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    old = json.loads((BASE/'evaluation-protocol.json').read_text())
    spec = dict(role='post-hoc first-frame appearance diagnosis, not independent validation',
        checkpoints=[dict(path=str(p.relative_to(LAB)), sha256=sha(p)) for p in CHECKPOINTS],
        fixed_human_rois_xyxy=old['fixed_human_rois_xyxy'], source_defined_shirt_interior_camera15=SHIRT_ROI,
        shirt_roi_definition='Frame 0 source RGB, interior of grey shirt selected before diagnostic renders; avoids head/arms/background',
        interventions=['unchanged rendering', 'clamp each Gaussian RGB to [0,1] with identical geometry and opacity', 'render fixed binary foreground labels with original geometry and opacity'],
        limitations=['Known-camera post-hoc diagnosis, no test tuning claim', 'Clipping Gaussian colors is not retraining',
                     'Foreground contribution is the alpha-weighted binary label integral, not geometric ground truth',
                     'Indoor 640x360 basketball sequence, not street-scale completion'],
        implementation_sha256=sha(Path(__file__)))
    protocol_file = OUT/'diagnostic-protocol.json'
    if protocol_file.exists():
        protocol = json.loads(protocol_file.read_text())
        assert {k:v for k,v in protocol.items() if k != 'created_utc'} == spec
    else:
        protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), **spec)
        save(protocol_file, protocol)
    if not (OUT/'parameter-audit.json').exists():
        inspect_parameters()
    if args.render:
        assert not (OUT/'render-diagnostics.json').exists(), 'Preserve completed diagnostic'
        with torch.no_grad():
            render_diagnostics(protocol)


if __name__ == '__main__':
    main()
