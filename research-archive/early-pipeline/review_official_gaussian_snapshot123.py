"""Prepare CPU-only, unchanged-pixel checkpoint comparisons; never touch training."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def horizontal(images):
    out = Image.new('RGB', (sum(im.width for im in images), max(im.height for im in images)), (245, 245, 245))
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += im.width
    return out


def main(root, step):
    root = Path(root)
    baseline = read(root / 'reviews/step-001000-images/index.json')
    snap = root / f'gaussian-run/step-{step:06d}'
    parity = read(snap / 'native-ply-parity.json')
    assert parity['passed'] and parity['sha256'] == sha(snap / 'world.ply')
    assert parity['reference_frame_id'] == 'frame_000032'
    out = root / f'reviews/step-{step:06d}-images'
    out.mkdir(exist_ok=True)
    receipt = dict(updated=datetime.now(timezone.utc).isoformat(), step=step,
                   world_ply_sha256=parity['sha256'], images=[], actual_visual_review=False,
                   contract='Matching named pose/camera; exact step1000 integer crops. No extra resizing, warping, or text.')
    contacts = {'面部': [], '双脚': []}
    for item in baseline['images']:
        frame, region = item['frame'], item['region']
        source, render = snap / f'{frame}_input.png', snap / f'{frame}_render.png'
        assert sha(source) == item['source_input_sha256'], f'Changed input for {frame}'
        old_render = root / f'gaussian-run/step-001000/{frame}_render.png'
        images = [Image.open(p).convert('RGB') for p in (source, old_render, render)]
        assert all(list(im.size) == item['training_wh'] for im in images)
        crop = tuple(item['crop_training_integer_xyxy'])
        images = [im.crop(crop) for im in images]
        alias = out / Path(item['alias']).name
        target = Path(item['path'].replace('s001000', f's{step:06d}'))
        pair = horizontal([images[0], images[2]])
        pair.save(alias)
        pair.save(target)
        comparison = alias.with_stem(alias.stem + '_vs001000')
        horizontal(images).save(comparison)
        if region in contacts:
            contacts[region].append(Image.open(comparison).convert('RGB'))
        receipt['images'].append(dict(frame=frame, source_frame=item['source_frame'], region=region,
            path=str(target), alias=str(alias), comparison=str(comparison), sha256=sha(alias),
            source_input_sha256=sha(source), source_render_sha256=sha(render),
            crop_training_integer_xyxy=list(crop), training_wh=item['training_wh'],
            comparison_panels=['input', 'step001000_matching_pose', f'step{step:06d}_matching_pose'],
            extra_resize=False, image_text_drawn=False))
    for region, panels in contacts.items():
        contact = Image.new('RGB', (max(im.width for im in panels), sum(im.height for im in panels)), (245, 245, 245))
        y = 0
        for im in panels:
            contact.paste(im, (0, y))
            y += im.height
        contact.save(out / ('face-contact.png' if region == '面部' else 'feet-contact.png'))
    (out / 'index.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(step=step, output=str(out), image_count=len(receipt['images']), actual_visual_review=False), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--step', type=int, required=True)
    args = parser.parse_args()
    if args.step not in (6000, 12000, 18000):
        parser.error('Use an already completed scheduled checkpoint: 6000, 12000, or 18000.')
    main(args.root, args.step)
