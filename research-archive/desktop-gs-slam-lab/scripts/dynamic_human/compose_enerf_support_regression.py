"""Preserve the worst paired regression and its adjacent instants at native pixels."""
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont

LAB = Path(__file__).resolve().parents[2]
out = LAB / 'outputs/dynamic-human-enerf-support-030'
metrics = json.loads((out / 'evaluation/metrics.json').read_text())
stats = json.loads((out / 'paired-statistics.json').read_text())
worst = next(c for c in stats['comparisons'] if c['candidate'] == 'both')['worst_psnr_pair']
camera, center = worst['camera'], worst['frame']
font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 18)
records = []
for frame in range(center - 1, center + 2):
    row = next(r for r in metrics['views'] if r['camera'] == camera and r['common_frame'] == frame and r['model'] == 'frozen')
    images = [Image.open(out / 'evaluation' / f'camera{camera}-frame{frame:03d}-{g}.png').convert('RGB') for g in ['reference', 'frozen', 'both']]
    for person in row['people']:
        box = person['upper_body_bbox']
        width, height = box[2] - box[0], box[3] - box[1]
        panel = Image.new('RGB', (3 * width, height + 44), 'white')
        draw = ImageDraw.Draw(panel)
        for column, (im, label) in enumerate(zip(images, ['Reference', 'Fixed scale / opacity', 'Both adapt'])):
            panel.paste(im.crop(box), (column * width, 44))
            draw.text((column * width + 6, 12), label, font=font, fill='black')
        file = f'comparison/regression-camera{camera}-frame{frame:03d}-person{person["person"]}.png'
        panel.save(out / file)
        records.append(dict(file=file, camera=camera, frame=frame, person=person['person'], crop=box, resized=False))
(out / 'regression-composition.json').write_text(json.dumps(dict(selection='Worst foreground PSNR delta of both versus fresh frozen, plus adjacent instants; not selected for favorable appearance', worst=worst, panels=records, enhancement=False), indent=2) + '\n', encoding='utf8')
print(json.dumps(dict(worst=worst, panels=len(records))))
