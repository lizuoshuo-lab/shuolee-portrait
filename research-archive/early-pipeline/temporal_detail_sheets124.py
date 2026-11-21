"""Lossless native detail sheets for an existing review, without new fitting."""
import argparse
import json
from pathlib import Path
from PIL import Image


def local(path):
    path = str(path)
    if path.startswith('/mnt/'):
        path = path[5].upper() + ':/' + path[7:]
    return Path(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('review')
    args = parser.parse_args()
    out = Path(args.review)
    report = json.loads((out / 'result.json').read_text(encoding='utf-8'))
    manifest = {}
    for region in ('face', 'feet'):
        paths = []
        # Keep source pixels; the sheet adds blank padding only.
        for begin in range(0, len(report['records']), 4):
            rows = report['records'][begin:begin + 4]
            images = [Image.open(local(row['images'][region])).convert('RGB') for row in rows]
            page = Image.new('RGB', (max(i.width for i in images), sum(i.height for i in images)))
            y = 0
            for im in images:
                page.paste(im, (0, y))
                y += im.height
            path = out / f'{region}-{begin // 4 + 1:02d}.png'
            if path.exists():
                assert Image.open(path).convert('RGB').tobytes() == page.tobytes()
            else:
                page.save(path)
            paths.append(dict(path=str(path), frames=[r['frame'] for r in rows]))
        manifest[region] = paths
    (out / 'detail-sheets.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False))
