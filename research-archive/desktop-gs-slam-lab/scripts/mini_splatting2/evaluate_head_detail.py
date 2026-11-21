"""CPU-only head/shoulder appearance diagnostics against real frozen images.

Gradient agreement measures fidelity to the reference, not raw sharpness.
The region is the top third of a frozen person box, not a head segmentation.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import statistics

import cv2
import numpy as np
from PIL import Image

LAB=Path(__file__).resolve().parents[2];ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'


def metric(pred,gt):
    diff=pred-gt
    psnr=-10*np.log10(max(float(np.mean(diff**2)),1e-12))
    coefficients=np.array([.2126,.7152,.0722],dtype='float32')
    p=pred@coefficients;g=gt@coefficients
    pg=[cv2.Sobel(p,cv2.CV_32F,dx,dy,ksize=3,scale=1/8) for dx,dy in [(1,0),(0,1)]]
    gg=[cv2.Sobel(g,cv2.CV_32F,dx,dy,ksize=3,scale=1/8) for dx,dy in [(1,0),(0,1)]]
    error=np.sqrt((pg[0]-gg[0])**2+(pg[1]-gg[1])**2)
    ref=np.sqrt(gg[0]**2+gg[1]**2);predmag=np.sqrt(pg[0]**2+pg[1]**2)
    interior=np.s_[2:-2,2:-2]
    return dict(psnr=float(psnr),mae=float(np.mean(np.abs(diff))),
        normalized_gradient_error=float(error[interior].mean()/max(float(ref[interior].mean()),1e-8)),
        gradient_magnitude_ratio=float(predmag[interior].mean()/max(float(ref[interior].mean()),1e-8)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--weight',choices=['003','010'],required=True);a=p.parse_args()
    target=ROOT/f'head-detail-depth{a.weight}-001.json';assert not target.exists()
    assignment=next(n for n in ast.parse(Path(__file__).with_name('local_refinement.py').read_text()).body
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SUBJECT_BOX' for t in n.targets))
    boxes=ast.literal_eval(assignment.value);rows=[];fingerprints={}
    for seed in range(3):
        baseline=ROOT/f'native-motion-mip-depthmask-18000-s{seed}'
        guided=ROOT/f'native-depthguide{a.weight}-18000-s{seed}'
        assert json.loads((guided/'manifest.json').read_text())['status']=='complete'
        summary=json.loads((baseline/'evaluation-summary.json').read_text())
        for index,row in enumerate(summary['native_rows']):
            relative=Path('model/test/ours_18000');filename=f'{index:05d}.png'
            gt_path=baseline/relative/'gt'/filename
            assert hashlib.sha256(gt_path.read_bytes()).digest()==hashlib.sha256((guided/relative/'gt'/filename).read_bytes()).digest()
            gt=np.asarray(Image.open(gt_path).convert('RGB')).astype('float32')/255
            h,w=gt.shape[:2];b=boxes[int(row['frame'])]
            x0,y0,x1,y1=[round(v*s) for v,s in zip(b,[w,h,w,h])];y1=y0+round((y1-y0)/3)
            values={}
            for key,path in [('baseline',baseline),('guided',guided)]:
                source=path/relative/'renders'/filename
                fingerprints[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
                pred=np.asarray(Image.open(source).convert('RGB')).astype('float32')/255
                assert pred.shape==gt.shape
                values[key]=metric(pred[y0:y1,x0:x1],gt[y0:y1,x0:x1])
            rows.append(dict(seed=seed,frame=row['frame'],box=[x0,y0,x1,y1],**values,
                difference={key:values['guided'][key]-values['baseline'][key] for key in values['baseline']}))
    by_seed=[]
    for seed in range(3):
        selected=[r for r in rows if r['seed']==seed]
        by_seed.append(dict(seed=seed,**{group:{metric_name:statistics.mean(r[group][metric_name] for r in selected)
            for metric_name in selected[0][group]} for group in ['baseline','guided','difference']}))
    result=dict(status='complete',weight=a.weight,rows=rows,by_seed=by_seed,
        aggregate={group:{key:dict(mean=statistics.mean(s[group][key] for s in by_seed),sample_std=statistics.stdev(s[group][key] for s in by_seed))
            for key in by_seed[0][group]} for group in ['baseline','guided','difference']},
        protocol=dict(region='top third of pre-existing person bounding rectangle; includes neck/shoulder/background',
            color='encoded RGB PNG values; weighted grayscale for gradients, not radiometric linear luminance',
            gradient='Sobel3x3/8 vector difference normalized by reference gradient mean; lower is more faithful',
            magnitude_ratio='descriptive only; maximizing it can reward oversharpening',
            interpretation='same-video development evidence across3 matched seeds; no new views or GT depth',
            new_region_not_independent_test=True),render_sha256=fingerprints)
    target.write_text(json.dumps(result,indent=2));print(json.dumps(dict(by_seed=by_seed,aggregate=result['aggregate']),indent=2))


if __name__=='__main__':main()
