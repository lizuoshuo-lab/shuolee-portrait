"""Combine measured mask-fit and RGB errors on identical held cameras."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


ARMS = ('baseline','matte','matte-centers','matte-mesh','combined')


def run(root):
    read=lambda p:json.loads(p.read_bytes())
    review=root/'fixed-review'
    rendering=read(review/'fixed-detail.json')
    held=rendering['rgb_holdout']
    all_scores={}
    for arm in ARMS:
        report=read(root/arm/'surface-fit/surface-fit.json')
        scores={}
        for geometry in ('fixed_mesh','fixed_gauss','mesh','gauss'):
            parts={}
            for region in ('all','hair','profile','neck'):
                samples=[report['frames'][n]['observed_contour'].get(geometry,{}).get(region) for n in held]
                samples=[s for s in samples if s]
                if samples:
                    total=sum(s['count'] for s in samples)
                    parts[region]=dict(views=len(samples),pixels=total,
                        mean_px=float(sum(s['mean']*s['count'] for s in samples)/total),
                        worst_view_p95_px=max(s['p95'] for s in samples),
                        over_3_fraction=sum(s['over3'] for s in samples)/total)
            scores[geometry]=parts
        scores['fixed_head_rgb_mae_255']=float(np.mean([report['frames'][n]['clean_head_rgb_mae255']['fixed'] for n in held]))
        scores['max_flipped_triangles']=max(f['flipped_vs_initial_triangles'] for f in report['frames'].values())
        all_scores[arm]=scores
    (review/'contour-comparison.json').write_text(json.dumps(dict(fixed_pose=rendering['pose'],held=held,
        measure='Observed native person/background edge to nearest predicted outer contour. RGB holdout only.',
        arms=all_scores),indent=2,allow_nan=False))
    for name in [rendering['pose'],'00300.jpg','00850.jpg',*held]:
        key=Path(name).stem
        panels=[]
        for arm in ('baseline','matte-centers','matte-mesh','combined'):
            image=cv2.imread(str(root/arm/'surface-fit'/(key+'-fixed-outlines.png')))
            if image is None:
                continue
            src,observed,mesh,gauss=np.hsplit(image,4)
            if not panels:
                panels.append(('Reference / Mask',np.vstack((src,observed))))
            panels.append((arm,np.vstack((mesh,gauss))))
        if panels:
            labeled=[]
            for title,image in panels:
                im=cv2.copyMakeBorder(image,27,0,0,0,cv2.BORDER_CONSTANT,value=(27,27,27))
                cv2.putText(im,title,(5,19),cv2.FONT_HERSHEY_SIMPLEX,.48,(250,250,250),1,cv2.LINE_AA)
                labeled.append(im)
            if not cv2.imwrite(str(review/(key+'-mesh-and-gaussians.png')),np.hstack(labeled)):
                raise IOError(key)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    run(p.parse_args().root)
