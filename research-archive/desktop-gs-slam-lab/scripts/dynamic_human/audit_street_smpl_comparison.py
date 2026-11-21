"""Compare actual saved images at matching update budgets, without GPU work."""
from pathlib import Path
import argparse
import hashlib
import json
import statistics
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from audit_street_joint_images import check_quantization

LAB = Path(__file__).resolve().parents[2]
BASE = LAB / 'outputs/dynamic-human-street-joint-009'
BODY = LAB / 'outputs/dynamic-human-street-smpl-joint-011'
SCENE = LAB / 'data/dynamic-human/argoverse2/processed/training/000'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', choices=['updates-00500', 'updates-03000', 'updates-10000', 'final'], default='final')
    args = parser.parse_args()
    final = args.label == 'final'
    steps = 30000 if final else int(args.label.split('-')[1])
    runs = [('deformable', BASE / ('resume-10000-expandable' if final else 'baseline-bounded')),
            ('articulated', BODY / 'articulated')]
    # Wait for the metric receipt, which is written after all images in that stage.
    metrics_paths = [r / 'evaluation' / args.label / 'metrics.json' for _, r in runs]
    if not all(p.exists() for p in metrics_paths):
        print(json.dumps(dict(status='not_ready_no_comparison_created', label=args.label,
                              missing=[str(p) for p in metrics_paths if not p.exists()])))
        return
    results = [json.loads(p.read_text()) for p in metrics_paths]
    expected_views = ({(f, c) for f in range(10, 151, 10) for c in range(7)} if final
                      else {(130, 4), (140, 4), (150, 6)})
    roi_rows = json.loads((BASE / 'baseline-bounded/person-rois.json').read_text())['regions']
    rois = {(r['frame'], r['camera'], r['instance']): r for r in roi_rows
            if (r['frame'], r['camera']) in expected_views}
    assert len(rois) == (435 if final else 12)
    target_cache, image_checks, region_checks, branch_scores = {}, [], [], {}
    for (name, run), result in zip(runs, results):
        assert result['optimizer_updates'] == steps
        assert {(v['frame'], v['camera']) for v in result['views']} == expected_views
        actual_rows = {(r['frame'], r['camera'], r['instance']): r for r in result['person_box_regions']}
        assert actual_rows.keys() == rois.keys()
        assert len(actual_rows) == len(result['person_box_regions'])
        for row in result['views']:
            key = row['frame'], row['camera']
            stem = f'frame{key[0]:03d}-camera{key[1]}'
            path = run / 'evaluation' / args.label / (stem + '-prediction.png')
            pred = np.asarray(Image.open(path).convert('RGB'))
            original = Image.open(SCENE / 'images' / f'{key[0]:03d}_{key[1]}.jpg').convert('RGB')
            gt = np.asarray(original.resize((pred.shape[1], pred.shape[0]), Image.Resampling.BILINEAR))
            saved = np.asarray(Image.open(run / 'ground-truth' / (stem + '.png')).convert('RGB'))
            assert np.array_equal(gt, saved), (name, key)
            if key in target_cache:
                assert np.array_equal(target_cache[key], gt), ('comparison targets differ', key)
            target_cache[key] = gt
            score, quantization_diff = check_quantization(gt, pred, row['psnr_db'])
            image_checks.append(dict(branch=name, frame=key[0], camera=key[1],
                sha256=sha(path), png_psnr_db=score, quantization_difference_db=quantization_diff))
            for roi_key, fixed in rois.items():
                if roi_key[:2] != key:
                    continue
                current = actual_rows[roi_key]
                assert current['box'] == fixed['box'] and current['image_size'] == fixed['image_size']
                x0, y0, x1, y1 = fixed['box']
                score, diff = check_quantization(gt[y0:y1, x0:x1], pred[y0:y1, x0:x1], current['psnr_db'])
                if roi_key[2] == 79:
                    crop = run / 'evaluation' / args.label / (stem + '-person79.png')
                    assert np.array_equal(pred[y0:y1, x0:x1], np.asarray(Image.open(crop).convert('RGB')))
                    saved_target_crop = run / 'ground-truth' / (stem + '-person79.png')
                    assert np.array_equal(gt[y0:y1, x0:x1], np.asarray(Image.open(saved_target_crop).convert('RGB')))
                region_checks.append(dict(branch=name, frame=key[0], camera=key[1], instance=roi_key[2],
                    psnr_db=current['psnr_db'], ssim=current['ssim'], png_psnr_db=score, quantization_difference_db=diff))
        branch_scores[name] = dict(full_image_mean_psnr_db=statistics.mean(r['psnr_db'] for r in result['views']),
            all_boxes_mean_psnr_db=statistics.mean(r['psnr_db'] for r in result['person_box_regions']),
            main_person_mean_psnr_db=statistics.mean(r['psnr_db'] for r in result['person_box_regions'] if r['instance'] == 79))
    out = BODY / ('paired-' + args.label + '-audit')
    out.mkdir(exist_ok=True)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 20)
    figures = []
    for f, c, identity in sorted(rois):
        if identity != 79:
            continue
        stem = f'frame{f:03d}-camera{c}'
        target = Image.open(runs[0][1] / 'ground-truth' / (stem + '-person79.png')).convert('RGB')
        panels = [('原始观测', target)]
        for label, (_, run) in zip(('可变形模型', '人体关节模型'), runs):
            panels.append((label, Image.open(run / 'evaluation' / args.label / (stem + '-person79.png')).convert('RGB')))
        assert all(p.size == target.size for _, p in panels)
        scale = min(3., 260 / target.width, 650 / target.height)
        size = tuple(round(x * scale) for x in target.size)
        canvas = Image.new('RGB', (840, size[1] + 90), 'white')
        draw = ImageDraw.Draw(canvas)
        for col, (label, panel) in enumerate(panels):
            draw.text((col * 280 + 12, 8), label, font=font, fill='#182d39')
            canvas.paste(panel.resize(size, Image.Resampling.NEAREST), (col * 280 + (280 - size[0]) // 2, 42))
        draw.text((12, size[1] + 52), f'{steps:,} 次更新 · 帧 {f} / 相机 {c} · 同区域、未锐化', font=font, fill='#182d39')
        target_path = out / (stem + '-person79-comparison.png')
        canvas.save(target_path)
        figures.append(dict(file=target_path.name, sha256=sha(target_path), frame=f, camera=c))
    report = dict(status='matched_saved_images_and_fixed_regions_verified_visual_review_pending',
        label=args.label, optimizer_updates_per_branch=steps, final_complete=final,
        source_metrics=[dict(path=str(p.relative_to(LAB)), sha256=sha(p)) for p in metrics_paths],
        views_per_branch=len(expected_views), person_boxes_per_branch=len(rois),
        branch_scores=branch_scores, images=image_checks, person_boxes=region_checks, figures=figures,
        limitations=['Known camera/trajectory/body-pose priors, including heldout times; not prediction.',
            'SMPL representation changes geometry, topology, regularizers and missing-pose fallback; not a single-factor ablation.',
            'Fixed person boxes include background/occluders, no silhouette GT.',
            'Same resized original targets and unchanged source face anonymization; no enhancement.',
            'PSNR/SSIM do not certify clothing clarity, alignment, occlusion or temporal consistency.'])
    (out / 'independent-audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('images', 'person_boxes', 'figures')}, indent=2))


if __name__ == '__main__':
    main()
