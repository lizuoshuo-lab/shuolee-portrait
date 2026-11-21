"""Independent PNG/original-image audit and real person-region comparisons."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LAB = Path(__file__).resolve().parents[2]
BASE = LAB / 'outputs/dynamic-human-street-joint-009'
SCENE = LAB / 'data/dynamic-human/argoverse2/processed/training/000'
PARENT = BASE / 'baseline-bounded'
RESUME = BASE / 'resume-10000-expandable'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def psnr(a, b):
    rmse = float(np.sqrt(np.mean((a.astype(np.float64) / 255 - b.astype(np.float64) / 255) ** 2)))
    return -20 * math.log10(max(rmse, 1e-6)), rmse


def check_quantization(a, b, expected):
    actual, rmse = psnr(a, b)
    # Prediction PNG rounds each channel within 0.5/255 of the float tensor;
    # original target is uint8/255. Minkowski gives a deterministic RMSE bound.
    epsilon = 0.5 / 255 + 2e-7
    lower = -20 * math.log10(max(rmse + epsilon, 1e-6))
    upper = -20 * math.log10(max(rmse - epsilon, 1e-6))
    assert lower - 1e-4 <= expected <= upper + 1e-4, (actual, expected, lower, upper)
    return actual, abs(actual - expected)


def main():
    final = (RESUME / 'completion.json').exists()
    OUT = BASE / ('final-image-audit' if final else 'checkpoint-image-audit')
    OUT.mkdir(exist_ok=True)
    branches = [(PARENT, f'updates-{step:05d}', step) for step in (500, 3000, 10000)]
    if final:
        branches.append((RESUME, 'final', 30000))
    rois = json.loads((PARENT / 'person-rois.json').read_text())['regions']
    assert len(rois) == 435
    roi_lookup = {(r['frame'], r['camera'], r['instance']): r for r in rois}
    rows, differences, images = [], [], []
    for run, label, steps in branches:
        directory = run / 'evaluation' / label
        result = json.loads((directory / 'metrics.json').read_text())
        assert result['optimizer_updates'] == steps
        if label == 'final':
            assert {(r['frame'], r['camera']) for r in result['views']} == {(f, c) for f in range(10, 151, 10) for c in range(7)}
            assert len(result['views']) == 105 and len(result['person_box_regions']) == 435
        for view in result['views']:
            f, c = view['frame'], view['camera']
            assert f % 10 == 0 and 0 < f < 156
            stem = f'frame{f:03d}-camera{c}'
            path = directory / f'{stem}-prediction.png'
            pred = np.array(Image.open(path).convert('RGB'))
            observed = Image.open(SCENE / 'images' / f'{f:03d}_{c}.jpg').convert('RGB')
            gt = np.array(observed.resize((pred.shape[1], pred.shape[0]), Image.Resampling.BILINEAR))
            saved_gt = np.array(Image.open(run / 'ground-truth' / f'{stem}.png'))
            assert np.array_equal(gt, saved_gt), ('saved ground truth differs from resized source', stem)
            image_psnr, diff = check_quantization(gt, pred, view['psnr_db'])
            differences.append(diff)
            images.append(dict(steps=steps, frame=f, camera=c, path=str(path.relative_to(BASE)),
                sha256=sha(path), source_sha256=sha(SCENE / 'images' / f'{f:03d}_{c}.jpg'), png_psnr_db=image_psnr))
            for row in result['person_box_regions']:
                if (row['frame'], row['camera']) != (f, c):
                    continue
                original_roi = roi_lookup[(f, c, row['instance'])]
                assert row['box'] == original_roi['box']
                assert row['image_size'] == original_roi['image_size'] == [pred.shape[1], pred.shape[0]]
                x0, y0, x1, y1 = row['box']
                value, diff = check_quantization(gt[y0:y1, x0:x1], pred[y0:y1, x0:x1], row['psnr_db'])
                differences.append(diff)
                rows.append(dict(steps=steps, frame=f, camera=c, identity=row['instance'], box=row['box'],
                    psnr_db=row['psnr_db'], ssim=row['ssim'], png_psnr_db=value))
                if row['instance'] == 79:
                    assert np.array_equal(pred[y0:y1, x0:x1], np.array(Image.open(directory / f'{stem}-person79.png')))
                    assert np.array_equal(gt[y0:y1, x0:x1], np.array(Image.open(run / 'ground-truth' / f'{stem}-person79.png')))
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 22)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 18)
    figures = []
    for f, c in ((130, 4), (140, 4), (150, 6)):
        stem = f'frame{f:03d}-camera{c}'
        target = Image.open(PARENT / 'ground-truth' / f'{stem}-person79.png').convert('RGB')
        column_w = max(260, target.width)
        scale = min(2., column_w / target.width)
        size = (round(target.width * scale), round(target.height * scale))
        panels = [('原始观测', target)]
        for run, label, steps in branches:
            panels.append((f'{steps:,} 次更新', Image.open(run / 'evaluation' / label / f'{stem}-person79.png').convert('RGB')))
        canvas = Image.new('RGB', (column_w * len(panels), size[1] + 108), 'white')
        draw = ImageDraw.Draw(canvas)
        for column, (title, panel) in enumerate(panels):
            x = column * column_w
            draw.text((x + 12, 10), title, font=font, fill='#182d39')
            # Same nearest-neighbor display enlargement on every panel, no enhancement.
            canvas.paste(panel.resize(size, Image.Resampling.NEAREST), (x + (column_w - size[0]) // 2, 47))
            if column:
                step = branches[column - 1][2]
                score = next(r for r in rows if (r['steps'], r['frame'], r['camera'], r['identity']) == (step, f, c, 79))
                draw.text((x + 12, size[1] + 57), f"PSNR {score['psnr_db']:.2f} dB", font=small, fill='#182d39')
        path = OUT / f'{stem}-person79-comparison.png'
        canvas.save(path)
        figures.append(dict(path=str(path.relative_to(BASE)), frame=f, camera=c, identity=79,
            display='equal nearest-neighbor enlargement; source anonymization preserved; no sharpening', sha256=sha(path)))
        focused = Image.new('RGB', (column_w * 2, size[1] + 70), 'white')
        focused_draw = ImageDraw.Draw(focused)
        for column, (title, panel) in enumerate((panels[0], panels[-1])):
            focused_draw.text((column * column_w + 12, 10), title, font=font, fill='#182d39')
            focused.paste(panel.resize(size, Image.Resampling.NEAREST),
                (column * column_w + (column_w - size[0]) // 2, 47))
        focused_path = OUT / f'{stem}-person79-observation-reconstruction.png'
        focused.save(focused_path)
        figures.append(dict(path=str(focused_path.relative_to(BASE)), frame=f, camera=c, identity=79,
            display='original and latest audited checkpoint; equal enlargement; no sharpening', sha256=sha(focused_path)))
    report = dict(status='saved_images_source_targets_and_predeclared_regions_verified_visual_review_pending',
        final_complete=final, saved_images_verified=len(images), person_regions_verified=len(rows),
        max_png_float_psnr_difference_db=max(differences), psnr_check='Float metric within independently derived 8-bit quantization bound',
        main_person_rows=[r for r in rows if r['identity'] == 79],
        figures=figures, images=images,
        boundaries=['Only every-tenth-time heldout views; known trajectories, not forecasting.',
            'Projected person boxes include background and occluders; not segmentation GT.',
            'Checkpoint progression is not a controlled method ablation.',
            'Visual quality is not certified by aggregate PSNR.'])
    (OUT / 'independent-audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('images', 'figures')}, indent=2))


if __name__ == '__main__':
    main()
