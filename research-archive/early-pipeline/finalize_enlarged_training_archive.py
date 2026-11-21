"""Verify and publish the direct-video enlargement experiment records."""
from pathlib import Path
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
import hashlib
import json
import shutil

import numpy as np
from PIL import Image
from plyfile import PlyData

DELIVERY = Path('/mnt/d/00/记录/2026-11-02_人体上采样静止重建')
PREVIOUS = Path('/opt/gs-slam-lab/static-person-upsampling-20261102')
NATIVE = Path('/opt/gs-slam-lab/static-person-native-upsampling-20261102')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(value for key, value in attrs if key in ['href', 'src'] and value)


def main():
    report = dict(updated=datetime.now(timezone.utc).isoformat(), subjects={}, passed=False,
        method='Crop person at native video resolution, enlarge width/height 2x before training, train without shrinking.',
        learned_superresolution=False, quality_accepted=False, production_changed=False)
    cards = []; table_rows = []
    for subject, folder, base in [('demo', 'demo', PREVIOUS), ('sofa', 'sofa_native', NATIVE)]:
        root = base / subject; out = DELIVERY / folder
        enlarged = read(out / 'enlarged-evaluation.json')
        native = read(out / 'comparison.json')
        assert sha(Path(native['resolved_source_video'])) == native['plan']['source_video_sha256']
        assert (out / '放大后训练.html').exists()
        assert enlarged['steps_each'] == 18000 and enlarged['no_training_downsampling']
        orig_wh = enlarged['original_training_wh']; big_wh = enlarged['enlarged_training_wh']
        assert big_wh == [v * 2 for v in orig_wh]
        arm_checks = {}
        for arm, expected in [('original', orig_wh), ('upsampled2x', big_wh)]:
            rr = read(root / arm / 'data/training-inputs.json')['records']
            cc = read(root / arm / 'gaussian-run/training-cameras.json')['records']
            result = read(root / arm / 'gaussian-run/result.json')
            assert result['steps'] == 18000 and len(rr) == len(cc) == enlarged['frames']
            assert all(r['wh'] == c['wh'] == expected for r, c in zip(rr, cc))
            snapshots = list((root / arm / 'gaussian-run/step-018000').glob('*_render.png'))
            assert len(snapshots) == 6
            assert all(list(Image.open(p).size) == expected for p in snapshots)
            model = out / (arm + '.ply')
            assert sha(model) == native['models'][arm]['sha256']
            vertices = PlyData.read(str(model))['vertex'].data
            assert len(vertices) == native['models'][arm]['points'] and len(vertices.dtype.names) == 62
            assert all(np.isfinite(vertices[k]).all() for k in vertices.dtype.names)
            arm_checks[arm] = dict(input_wh=expected, actual_training_camera_wh=expected,
                final_snapshot_render_wh=expected, steps=18000, points=len(vertices), model_sha256=sha(model))
        images = {}
        for p in sorted(out.rglob('*.png')):
            with Image.open(p) as im:
                wh = list(im.size); im.verify()
            images[str(p.relative_to(out))] = dict(wh=wh, sha256=sha(p))
        for frame in [r['frame'] for r in read(out / 'image-manifest.json')['frames']]:
            p = out / frame / 'input_enlarged.png'
            assert images[str(p.relative_to(out))]['wh'] == big_wh
            assert images[str((out / frame / 'upsampled2x_enlarged.png').relative_to(out))]['wh'] == big_wh
        save(out / 'enlarged-image-manifest.json', dict(subject=subject, images=images, total=len(images)))
        if subject == 'sofa':
            shutil.copy2(root / 'independent-size-audit.json', out / 'training-size-audit.json')
            shutil.copy2(root / 'input-verification.json', out / 'native-input-verification.json')
        report['subjects'][subject] = dict(frames=enlarged['frames'], source_video_wh=native['original_video_wh'],
            original_training_wh=orig_wh, enlarged_training_wh=big_wh, actual_training=arm_checks,
            measured_at_enlarged_resolution=enlarged['mean_metrics'], measured_at_native_resolution=native['mean_metrics'],
            validated_images=len(images))
        title = '沙发' if subject == 'sofa' else 'demo'
        table_rows.append(f'<tr><td>{title}</td><td>{orig_wh[0]}×{orig_wh[1]}</td><td><strong>{big_wh[0]}×{big_wh[1]}</strong></td><td>18,000步，已完成</td></tr>')
        frame = read(out / 'image-manifest.json')['frames'][0]['frame']
        cards.append(f'<section><h2><a href="{folder}/放大后训练.html">{title}：查看放大后训练的结果</a></h2><p><a href="{folder}/放大后训练.html">放大输入与同尺寸渲染</a> · <a href="{folder}/查看对照.html">完整源帧、原分辨率对照与新视角</a> · <a href="{folder}/training-size-audit.json">实际训练尺寸记录</a></p><a href="{folder}/放大后训练.html"><img src="{folder}/{frame}/enlarged_body_comparison.png" alt="{title}放大后的输入和重建结果"></a><p><a href="{folder}/{frame}/input_enlarged.png">打开完整放大输入 {big_wh[0]}×{big_wh[1]}</a> · <a href="{folder}/{frame}/upsampled2x_enlarged.png">打开完整放大训练结果</a></p></section>')
    findings_path = DELIVERY / 'native-review-findings.json'
    findings = read(findings_path)
    assert findings['visual_review_complete']
    note = findings['conclusion_zh']
    style = 'body{font:17px/1.65 system-ui,sans-serif;max-width:1150px;margin:28px auto;padding:0 20px;background:#f2f4f6;color:#1c2833}section{padding:20px;margin:24px 0;background:white;border-radius:10px}img{max-width:100%;height:auto}table{border-collapse:collapse;background:white}td,th{padding:10px 18px;border:1px solid #ccd1d8}a{color:#1763ad}.note{color:#5d6873}'
    page = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>先放大，再训练 · demo与沙发</title><style>{style}</style></head><body><h1>人体图像先放大，再训练</h1><p><strong>原视频尺寸的人体裁切 → 宽高各放大2倍 → 使用放大图训练 → 按放大尺寸渲染</strong></p><p>使用Lanczos插值，放大后像素数为4倍。已验证训练输入、相机和快照，训练器保持放大尺寸。</p><table><tr><th>数据</th><th>原视频人体裁切</th><th>实际放大训练尺寸</th><th>放大组状态</th></tr>{"".join(table_rows)}</table><p>demo使用已核实的放大训练结果；沙发已重新从原视频直接裁切，完成原尺寸和放大尺寸两组训练。</p><section><h2>对照观察</h2><p>{note}</p><p class="note">观察覆盖每组6个拍摄视角与12个新视角；新视角没有实拍真值，像素误差不等同于几何正确。</p><a href="native-review-findings.json">观察和测量记录</a></section>{"".join(cards)}<p class="note">页面预览会适应窗口宽度。点击图片链接可打开完整尺寸PNG；原视频完整帧、脸部、人体和结果图均单独保存。</p><details><summary>上一轮沙发输入记录</summary><p>上一轮从旧流程1600×899帧裁切后再放大。本轮直接使用2704×1520原视频。</p><a href="sofa/查看对照.html">保留的旧对照</a></details><p><a href="native-training-verification.json">归档校验与实际尺寸</a></p></body></html>'
    (DELIVERY / '查看对照.html').write_text(page, encoding='utf-8')
    count = 0
    for p in DELIVERY.rglob('*.html'):
        parser = Links(); parser.feed(p.read_text(encoding='utf-8'))
        for ref in parser.links:
            u = urlsplit(ref)
            assert not u.scheme and not u.netloc
            target = (p.parent / unquote(u.path)).resolve()
            assert target.is_relative_to(DELIVERY.resolve())
            if target == DELIVERY / 'native-training-verification.json':
                continue
            assert target.exists(), str(target)
            count += 1
    report.update(passed=True, html_local_references_verified=count)
    save(DELIVERY / 'native-training-verification.json', report)
    print(json.dumps({'passed': True, 'subjects': report['subjects'], 'references': count}), flush=True)


if __name__ == '__main__':
    main()
