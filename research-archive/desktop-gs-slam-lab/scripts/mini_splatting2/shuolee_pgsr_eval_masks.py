"""Repair missing development masks from source-only annotation, before scoring.

The original training-only masks intentionally omit every development frame.
This addendum retains the frozen evaluation and records the correction separately.
"""
import shutil
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from shuolee_pgsr_evaluate import OUT, SOURCE, dump, read, sha


def main():
    original=read(OUT/'protocol.json')
    folder=OUT/'development-masks'
    folder.mkdir(exist_ok=False)
    review=OUT/'review'
    assert not any(review.iterdir()), 'Do not revise masks after scoring or comparison review'
    score=Path(__file__).with_name('shuolee_pgsr_score.py')
    # Source-only coarse rectangle annotation of the same near walking bystander
    # excluded by the training masks. Other background people remain included.
    rectangles={'000000':[0.0,.55,.04,.76], '000008':[.06,.36,.28,.68]}
    protocol=dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        status='source_only_annotation_pending_overlay_review',
        original_protocol_sha256=sha(OUT/'protocol.json'),
        old_score_sha256=sha(score),
        reason='Scorer referenced training-only masks absent for all eight development views.',
        ordering='After native renders, before any metrics or visual comparison of those renders.',
        annotation_source='All eight source photographs inspected; model renders not inspected to define masks.',
        rectangles_normalized=rectangles,dilation_pixels=8,
        scope='Only near walking bystander in frames0/8 excluded. Other people or motion may remain. Do not call this perfectly static background.',
        annotation='Coarse source-image bounding rectangles, not a learned segmentation or ground-truth motion field.',
        unmasked_full_metrics_also_required=True,rows={})
    shutil.copy2(score,folder/'score-before-mask-fix.py')
    for camera in original['development_cameras']:
        n=camera['img_name'];path=SOURCE/'images'/(n+'.png')
        assert sha(path)==original['images_sha256'][n]
        im=np.asarray(Image.open(path).convert('RGB'));h,w=im.shape[:2]
        mask=np.zeros((h,w),np.uint8)
        if n in rectangles:
            a,b,d,e=[round(v*s) for v,s in zip(rectangles[n],[w,h,w,h])]
            mask[b:e,a:d]=255
            mask=cv2.dilate(mask,np.ones((17,17),np.uint8))
            overlay=im.copy();selected=mask>0
            overlay[selected]=(overlay[selected]*.35+np.array([255,0,0])*.65).astype('uint8')
            Image.fromarray(overlay).save(folder/(n+'-source-overlay.jpg'))
        a,b,d,e=original['regions'][n]['person']
        assert not mask[b:e,a:d].any(), 'The intended subject must not be excluded'
        target=folder/(n+'.png');Image.fromarray(mask).save(target)
        protocol['rows'][n]=dict(path=target.name,sha256=sha(target),excluded_fraction=float((mask>0).mean()),
            source_sha256=sha(path),subject_excluded_fraction=0)
    dump(folder/'protocol.json',protocol)
    print({n:r['excluded_fraction'] for n,r in protocol['rows'].items()},flush=True)


if __name__=='__main__':main()
