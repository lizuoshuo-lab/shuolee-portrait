"""Read-only native mask intake; no detector, fitter or training invocation."""
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from PIL import Image

ROOT = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    for subject, record_id in [('whiteT', '002'), ('grayT', '003')]:
        source = ROOT / (subject + '-input')
        output = source / 'mask-review'
        if (output / 'result.json').exists():
            raise FileExistsError('Review the existing mask intake rather than overwrite it.')
        output.mkdir(exist_ok=True)
        records = Path('/mnt/d/00/记录') / record_id
        records.mkdir(exist_ok=True)
        selection = read(source / 'selection.json')
        rows = {r['frame']: r for r in read(source / 'data/inputs.json')['records']}
        chosen = sorted(selection['training_frames'] + selection['review_only_frames'])
        items, panels = [], []
        for name in chosen:
            row = rows[name]
            for key, expected in [('image', 'source_sha256'), ('mask', 'mask_sha256')]:
                assert sha(row[key]) == row[expected], (name, key)
            raw = Image.open(row['image']).convert('RGB')
            mask = Image.open(row['mask']).convert('L')
            assert raw.size == mask.size == tuple(row['wh'])
            box = mask.getbbox()
            assert box is not None
            box = (max(0, box[0]-40), max(0, box[1]-40),
                   min(raw.width, box[2]+40), min(raw.height, box[3]+40))
            region = raw.crop(box)
            alpha = mask.crop(box)
            cutout = Image.composite(region, Image.new('RGB', region.size), alpha)
            panel = Image.new('RGB', (region.width*3, region.height))
            for i, im in enumerate((region, alpha.convert('RGB'), cutout)):
                panel.paste(im, (i*region.width, 0))
            file = records / ('mask_124_' + name + '_原图_mask_mask内部人像.png')
            if file.exists():
                raise FileExistsError(file)
            panel.save(file)
            a = np.asarray(mask)
            items.append(dict(frame=name,role='training' if name in selection['training_frames'] else 'review_only',
                              source_sha256=row['source_sha256'],mask_sha256=row['mask_sha256'],
                              image=str(file),crop_integer=list(box),wh=list(raw.size),
                              nonzero_mask_pixels=int(np.count_nonzero(a)),
                              fractional_alpha_pixels=int(np.count_nonzero((a>0)&(a<255)))))
            panel.thumbnail((900, 520))
            panels.append(panel)
        pages = []
        for first in range(0, len(panels), 4):
            group = panels[first:first+4]
            page = Image.new('RGB', (max(p.width for p in group), sum(p.height for p in group)))
            y = 0
            for im in group:
                page.paste(im, (0,y)); y += im.height
            file = output / ('page-%02d.png' % (first//4+1))
            page.save(file); pages.append(str(file))
        result = dict(updated=datetime.now(timezone.utc).isoformat(),subject=subject,records=items,pages=pages,
                      native_mask_unchanged=True,training_roster_unchanged=True,
                      actual_visual_review=False,gaussian_admitted=False,fit_started=False,
                      source_hashes={str(source/'selection.json'):sha(source/'selection.json'),
                                     str(source/'data/inputs.json'):sha(source/'data/inputs.json')})
        (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print('NATIVE_MASK_REVIEW_READY', subject, len(items), flush=True)


if __name__ == '__main__':
    main()
