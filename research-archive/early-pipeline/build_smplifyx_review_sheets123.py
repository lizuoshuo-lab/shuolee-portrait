"""Arrange existing diagnostic crops at their original pixel sizes; never edit fits."""
import argparse
import hashlib
import io
import json
from pathlib import Path

from PIL import Image

ROOT = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/demo-vhap-initialization/fullbody-smplifyx/temporal-root-sequence/review')


def main(first, last):
    out = ROOT / 'native-sheets'
    out.mkdir(exist_ok=True)
    for start in range(first, last + 1, 5):
        stop = min(start + 4, last)
        rows, manifest = [], []
        for i in range(start, stop + 1):
            fid = f'frame_{i:06d}'
            paths = [ROOT / 'frames' / f'{fid}_{g}_raw_baseline_temporal.png'
                     for g in ('face', 'left_hand', 'right_hand', 'feet')]
            ims = [Image.open(p).convert('RGB') for p in paths]
            row = Image.new('RGB', (sum(im.width for im in ims) + 36,
                                   max(im.height for im in ims)), (32, 32, 32))
            x, slots = 0, []
            for p, im in zip(paths, ims):
                row.paste(im, (x, 0))
                assert row.crop((x, 0, x + im.width, im.height)).tobytes() == im.tobytes()
                slots.append(dict(file=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                  xywh=[x, 0, im.width, im.height]))
                x += im.width + 12
            rows.append(row)
            manifest.append(dict(frame=fid, slots=slots))
        sheet = Image.new('RGB', (max(r.width for r in rows),
                                 sum(r.height for r in rows) + 12 * (len(rows) - 1)), (32, 32, 32))
        y = 0
        for row, entry in zip(rows, manifest):
            sheet.paste(row, (0, y))
            entry['row_y'] = y
            y += row.height + 12
        path = out / f'frames_{start:03d}_{stop:03d}_face_hands_feet_native.png'
        data = io.BytesIO()
        sheet.save(data, format='PNG')
        if path.exists():
            assert path.read_bytes() == data.getvalue(), path
        else:
            path.write_bytes(data.getvalue())
        path.with_suffix('.json').write_text(json.dumps(dict(
            rows=manifest, no_resampling=True, no_text_added=True,
            each_group_columns=['raw', 'baseline', 'candidate']), indent=2), encoding='utf-8')
        print(path.name, sheet.size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('first', type=int)
    parser.add_argument('last', type=int)
    args = parser.parse_args()
    assert 1 <= args.first <= args.last <= 59
    main(args.first, args.last)
