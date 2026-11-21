"""Preserve native-pixel snapshot crops for the authorized resolution control."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from PIL import Image


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def join(images, vertical=False):
    size = (max(i.width for i in images), sum(i.height for i in images)) if vertical else (sum(i.width for i in images), max(i.height for i in images))
    out = Image.new('RGB', size, (245, 245, 245))
    offset = 0
    for im in images:
        out.paste(im, (0, offset) if vertical else (offset, 0))
        offset += im.height if vertical else im.width
    return out


def main(root, baseline, records):
    root, baseline, records = map(Path, (root, baseline, records))
    snapshot = root / 'gaussian-run/step-001000'
    request = read(root / 'reviews/gaussian-step-001000-request.json')
    for item in request['files']:
        assert sha(snapshot / item['name']) == item['sha256']
    out = root / 'reviews/native-crops'
    out.mkdir(exist_ok=False)
    records.mkdir(exist_ok=True)
    index = dict(snapshot_sha256=request['snapshot_sha256'], actual_visual_review=False,
                 panels=['original', 'native1520_step1000_matching_pose'], images=[],
                 no_resizing=True, no_image_text=True, comparison_to_old_render=False)
    contacts = {'面部': [], '双脚': []}
    for item in read(baseline / 'reviews/step-001000-images/index.json')['images']:
        frame, region = item['frame'], item['region']
        source, render = [Image.open(snapshot / f'{frame}_{suffix}.png').convert('RGB') for suffix in ('input', 'render')]
        assert source.size == render.size == (2704, 1520)
        w, h = item['training_wh']
        x0, y0, x1, y1 = item['crop_training_integer_xyxy']
        crop = (math.floor(x0 * 2704 / w), math.floor(y0 * 1520 / h), math.ceil(x1 * 2704 / w), math.ceil(y1 * 1520 / h))
        panel = join([source.crop(crop), render.crop(crop)])
        name = f'3DGS_原生1520_s001000_{frame}_{region}_左原图_右高斯.png'
        dest = records / name
        alias = out / f'{frame}_{region}.png'
        panel.save(dest)
        panel.save(alias)
        index['images'].append(dict(frame=frame, region=region, crop_native_xyxy=crop,
                                   alias=str(alias), record=str(dest), sha256=sha(alias)))
        if region in contacts:
            contacts[region].append(panel)
    for region, panels in contacts.items():
        join(panels, vertical=True).save(out / ('face-contact.png' if region == '面部' else 'feet-contact.png'))
    (out / 'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(output=str(out), images=len(index['images'])), ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--baseline', required=True)
    p.add_argument('--records', required=True)
    a = p.parse_args()
    main(a.root, a.baseline, a.records)
