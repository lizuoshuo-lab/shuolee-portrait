"""Summarize the completed baseline and compose unenhanced real image pairs."""
from pathlib import Path
import json
import statistics
from PIL import Image, ImageDraw, ImageFont

LAB = Path(__file__).resolve().parents[2]
BASE = LAB / 'outputs/dynamic-human-street-joint-009'
RUN = BASE / 'resume-10000-expandable'
OUT = BASE / 'final-image-audit'


def main():
    result = json.loads((RUN / 'completion.json').read_text())
    audit = json.loads((OUT / 'independent-audit.json').read_text())
    assert result['optimizer_updates'] == 30000 and audit['final_complete']
    assert len(result['views']) == 105 and len(result['person_box_regions']) == 435
    main_person = [r for r in result['person_box_regions'] if r['instance'] == 79]
    assert len(main_person) == 13
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 21)
    sheets = []
    for start in range(0, len(main_person), 5):
        subset = main_person[start:start + 5]
        canvas = Image.new('RGB', (800, 320 * len(subset)), 'white')
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(subset):
            stem = f"frame{row['frame']:03d}-camera{row['camera']}"
            target = Image.open(RUN / 'ground-truth' / (stem + '-person79.png')).convert('RGB')
            pred = Image.open(RUN / 'evaluation/final' / (stem + '-person79.png')).convert('RGB')
            assert target.size == pred.size
            scale = min(4., 370 / target.width, 270 / target.height)
            size = tuple(round(x * scale) for x in target.size)
            top = index * 320
            for col, (label, panel) in enumerate((('原始观测', target), ('30,000 次更新', pred))):
                draw.text((400 * col + 10, top + 4), f"{row['frame']} / 相机 {row['camera']} · {label}", font=font, fill='#182d39')
                canvas.paste(panel.resize(size, Image.Resampling.NEAREST),
                             (400 * col + (400 - size[0]) // 2, top + 42))
        name = f'person79-all-heldout-{start // 5 + 1}.png'
        canvas.save(OUT / name)
        sheets.append(dict(file=name, views=[dict(frame=r['frame'], camera=r['camera']) for r in subset]))
    def average(rows):
        return dict(count=len(rows), mean_psnr_db=statistics.mean(r['psnr_db'] for r in rows),
                    mean_ssim=statistics.mean(r['ssim'] for r in rows))
    progress = json.loads((RUN / 'progress.json').read_text())
    summary = dict(status='baseline_complete_numeric_audit_passed_visual_judgment_separate',
        optimizer_updates=30000, full_images=average(result['views']),
        all_person_boxes=average(result['person_box_regions']), main_person=average(main_person),
        lowest_full_image_psnr=min(result['views'], key=lambda r: r['psnr_db']),
        lowest_main_person_psnr=min(main_person, key=lambda r: r['psnr_db']),
        gaussians=progress['gaussians'], peak_allocated_bytes=progress['peak_allocated_bytes'],
        contact_sheets=sheets, display='Equal nearest-neighbor scaling per pair; no sharpening or content enhancement.',
        boundaries=['Mean PSNR is a mean of image/box dB values, not pooled pixel MSE.',
                    'Person boxes contain background and occluders; no silhouette GT.',
                    'Every tenth time held out; calibrated poses and trajectories remain known.',
                    'The model uses deformable people without SMPL; not full OmniRe.'])
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
