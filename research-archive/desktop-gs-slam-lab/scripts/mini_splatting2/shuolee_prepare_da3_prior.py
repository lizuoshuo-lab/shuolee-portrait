"""Build a bounded, training-only DA3 geometry prior after dense consistency checks."""
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
from shuolee_geometry_diagnosis import BASE, ROOT, OUT, sample, sha, stats, write_json


def main():
    dest=OUT/'da3-priors'
    dest.mkdir(exist_ok=False)
    cv2.setNumThreads(2)
    with np.load(OUT/'da3-full-predictions.npz') as cache:
        depths=cache['depth'];names=cache['frames'].tolist();intrinsics=cache['intrinsics']
    with np.load(OUT/'dense-consistency-full-masks.npz') as cache:
        assert cache['frames'].tolist()==names
        support=cache['da3'];original_weight=cache['common_weight']
    cams={c['img_name']:c for c in json.loads((BASE/'viewer/cameras.json').read_text())}
    boxes={r['frame']:r['box'] for r in json.loads((ROOT/'training-head-regions-003/manifest.json').read_text())['rows']}
    observations=json.loads((OUT/'sparse-observations.json').read_text())
    assert len(names)==51 and all(int(n)%8!=0 for n in names)
    protocol=dict(owner='shuolee',created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),source_sha256=sha(Path(__file__)),
        depth='DA3-BASE supplied with known training cameras, output aligned to the input camera scale; no additional affine fit',
        source_prediction_sha256=sha(OUT/'da3-full-predictions.npz'),
        support='At least one +/-3 training-neighbor agreement: relative depth<0.10, cycle<2px, RGB L1<0.15, nonoccluded',
        edges='Exclude log-depth range>0.08 in5x5 pixel window, erode resulting support by3x3',
        sparse_gate='Nonreserved robust SfM observations only: if DA3 relative error>0.10 exclude radius4processedpixels',
        confidence='Reuse original DA2 source-confidence/motion mask multiplied by new validity. DA3 raw confidence is not a probability and not used.',
        scale='Arbitrary SfM units, not metres',depth_resolution=[int(depths.shape[2]),int(depths.shape[1])],
        supervision='Coarse shape constraint, not facial detail supervision',appearance_holdout_excluded=True,
        independent_ground_truth=False,hyperparameter_sweep=False)
    write_json(dest/'protocol.json',protocol)
    rows=[];eval_rows=[]
    for i,name in enumerate(names):
        depth=depths[i];h,w=depth.shape;c=cams[name]
        log=np.log(depth)
        span=cv2.dilate(log,np.ones((5,5),np.uint8))-cv2.erode(log,np.ones((5,5),np.uint8))
        valid=(support[i]&(span<.08)&(depth>.2)&np.isfinite(depth)).astype('uint8')
        selected=[r for r in observations if r['frame']==name]
        xy=np.array([r['xy'] for r in selected])*[w/c['width'],h/c['height']]
        values=sample(depth,xy)
        sparse_rejected=0
        for j,record in enumerate(selected):
            if not record['prior_fit_reserved'] and abs(values[j]/record['z']-1)>.10:
                cv2.circle(valid,tuple(np.rint(xy[j]).astype(int)),4,0,-1);sparse_rejected+=1
        valid=cv2.erode(valid,np.ones((3,3),np.uint8)).astype(bool)
        weight=original_weight[i]*valid
        weight[weight<=.25]=0
        inverse=1/depth
        np.savez_compressed(dest/(name+'.npz'),inverse_depth=inverse.astype('float32'),weight=weight.astype('float32'))
        box=np.rint(np.array(boxes[name])*[w/c['width'],h/c['height'],w/c['width'],h/c['height']]).astype(int)
        head=np.zeros((h,w),bool);x0,y0,x1,y1=box;head[y0:y1,x0:x1]=True
        rows.append(dict(frame=name,sha256=sha(dest/(name+'.npz')),source_size=[c['width'],c['height']],
            prior_size=[w,h],sparse_rejected_observations=sparse_rejected,
            head_crop_valid_fraction=float((weight[head]>.25).mean()),
            outside_head_valid_fraction=float((weight[~head]>.25).mean())))
        sampled_weight=sample(weight,xy)
        for j,record in enumerate(selected):
            if record['prior_fit_reserved']:
                eval_rows.append(dict(frame=name,point_id=record['point_id'],head_crop=record['head_crop'],
                    accepted=bool(sampled_weight[j]>.25),error=float(abs(values[j]/record['z']-1))))
        if name in ['000014','000042']:
            photo=cv2.resize(np.asarray(Image.open(BASE/f'hover-sfm-001/dataset/images/{name}.jpg').convert('RGB')),(w,h))
            masked=photo.copy();masked[weight<=.25]//=5
            panel=Image.new('RGB',(2*w,h+36),'white');panel.paste(Image.fromarray(photo),(0,36));panel.paste(Image.fromarray(masked),(w,36))
            draw=ImageDraw.Draw(panel);draw.text((8,10),'Source image',fill='black');draw.text((w+8,10),'Accepted geometry supervision',fill='black')
            panel.save(dest/f'support-{name}.png')
    summary={}
    for region in ['head_crop','outside_head_crop']:
        rr=[r for r in eval_rows if r['head_crop']==(region=='head_crop')];kept=[r for r in rr if r['accepted']]
        summary[region]=dict(observations=len(rr),accepted_observations=len(kept),
            accepted_fraction=len(kept)/len(rr),error_all=stats([r['error'] for r in rr]),
            error_accepted=stats([r['error'] for r in kept]))
    manifest=dict(status='complete',owner='shuolee',protocol=protocol,records=rows,
        appearance_holdout_excluded=True,training_started=False,accepted_as_best=False,
        reserved_sparse_development=summary,
        scope='Eligible for a bounded matched-control refinement only. Correlated SfM checks are not independent validation.')
    write_json(dest/'manifest.json',manifest)
    print(json.dumps(dict(frames=len(rows),mean_head_coverage=float(np.mean([r['head_crop_valid_fraction'] for r in rows])),
        mean_outside_coverage=float(np.mean([r['outside_head_valid_fraction'] for r in rows])),reserved_sparse=summary),indent=2))


if __name__=='__main__':main()
