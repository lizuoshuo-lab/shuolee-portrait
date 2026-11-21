"""CPU readback of actual heldout RGB and alpha evidence, with exact source pins."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion

LAB=Path(r'\\wsl.localhost\Ubuntu-20.04\opt\gs-slam-lab')
WSL=Path(r'\\wsl.localhost\Ubuntu-20.04')


def read(p):
    return json.loads(p.read_text(encoding='utf8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def path_on_windows(path):
    parts=path.split('/')
    return Path(f'{parts[2].upper()}:/',*parts[3:]) if parts[1]=='mnt' else WSL.joinpath(*parts[1:])


def measure(root):
    results=read(root/'result.json')
    manifest=root/results['heldout_bound_replay']
    assert sha(manifest)==results['heldout_bound_replay_sha256']
    rows=read(manifest)['records']
    inputs={r['frame']:r for r in read(root/'data/inputs.json')['records']}
    records=[]
    for row in rows:
        source=inputs[row['frame']]
        assert source['source_sha256']==row['source_sha256'] and source['mask_sha256']==row['mask_sha256']
        rgb_path=path_on_windows(source['image'])
        mask_path=path_on_windows(source['mask'])
        assert sha(rgb_path)==row['source_sha256'] and sha(mask_path)==row['mask_sha256']
        box=tuple(row['crop_xyxy'])
        rgb=np.asarray(Image.open(rgb_path).convert('RGB').crop(box),dtype=np.float32)/255
        mask=np.asarray(Image.open(mask_path).convert('L').crop(box),dtype=np.float32)/255
        black=row['files']['black']; alpha=row['files']['alpha']
        black_path=manifest.parent/black['file']; alpha_path=manifest.parent/alpha['file']
        assert sha(black_path)==black['sha256'] and sha(alpha_path)==alpha['sha256']
        def load_saved(path, mode):
            im=Image.open(path).convert(mode)
            if im.size==tuple(row['wh']):
                im=im.crop(box)
            return np.asarray(im,dtype=np.float32)/255
        render=load_saved(black_path,'RGB')
        opacity=load_saved(alpha_path,'L')
        if rgb.shape!=render.shape or mask.shape!=opacity.shape:
            raise ValueError('Source and stored heldout raster dimensions differ.')
        inside=binary_erosion(mask>=.99,iterations=2)
        outside=~binary_dilation(mask>=.01,iterations=2)
        h=opacity[inside]<.95
        records.append(dict(frame=row['frame'],source_sha256=row['source_sha256'],mask_sha256=row['mask_sha256'],
            camera=row['camera'],crop_xyxy=box,foreground_pixels=int(inside.sum()),
            foreground_composite_l1=float(np.abs(render[inside]-rgb[inside]).mean()),
            foreground_alpha_mean=float(opacity[inside].mean()),
            foreground_below_0p95_fraction=float(h.mean()),
            outer_alpha_sum_px=float(opacity[outside].sum())))
    return dict(steps=results['steps'],record_sha256=sha(manifest),records=records)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('subject',choices=['016','006'])
    ap.add_argument('output',type=Path)
    a=ap.parse_args()
    current=LAB/f'shuolee-smpl-delivery-round21-{a.subject}'/'run'
    previous=LAB/f'shuolee-smpl-delivery-round20-{a.subject}'/'run'
    arms={label:measure(root) for label,root in (('R20',previous),('R21-10k',current/'duration-10000'),('R21-18k',current))}
    keys=['frame','source_sha256','mask_sha256','camera','crop_xyxy']
    for arm in ('R21-10k','R21-18k'):
        for x,y in zip(arms['R20']['records'],arms[arm]['records'],strict=True):
            assert all(x[k]==y[k] for k in keys)
    report=dict(subject=a.subject,method='Saved native pose-conditioned RGB black and alpha; independently hash-verified source, same pixels and camera. RGB compares premultiplied black RGB to original confident foreground; 8-bit stored renders.',
        caveat='Heldout pose conditioned on 2D observations; not an unseen geometric truth measurement.',arms=arms)
    with a.output.open('x',encoding='utf8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')
    metrics=['foreground_composite_l1','foreground_alpha_mean','foreground_below_0p95_fraction','outer_alpha_sum_px']
    for label,arm in arms.items():
        print(label, {key:round(float(np.mean([r[key] for r in arm['records']])),5) for key in metrics})


if __name__=='__main__':
    main()
