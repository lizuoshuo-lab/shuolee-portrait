"""Compare opaque-ray center-depth dispersion with common masks across all seeds.

Read-only development diagnostic, not a surface/floaters ground-truth score.
RGB error and dispersion association must not be interpreted as causality.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch

from depth_ray_moments import render_moments
from local_refinement import BASE,ROOT,read_model,cameras,render,image_tensor,SUBJECT_BOX


def stats(values):
    if not len(values):return dict(pixels=0)
    return dict(pixels=int(len(values)),mean=float(np.mean(values)),median=float(np.median(values)),
        p90=float(np.quantile(values,.9)),above005_fraction=float(np.mean(values>.05)),
        above010_fraction=float(np.mean(values>.1)))


def region_mask(shape,box):
    h,w=shape;x0,y0,x1,y1=[round(a*b) for a,b in zip(box,[w,h,w,h])]
    mask=np.zeros(shape,dtype=bool);mask[y0:y1,x0:x1]=True
    return mask


def heatmap(values,valid,maximum=.15):
    from matplotlib import colormaps
    rgb=(colormaps['magma'](np.clip(values/maximum,0,1))[...,:3]*255).round().astype('uint8')
    rgb[~valid]=[180,180,180]
    return Image.fromarray(rgb)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='depth-spread-evaluation-001');a=p.parse_args()
    out=ROOT/a.output;out.mkdir(exist_ok=False);torch.set_num_threads(4);cv2.setNumThreads(4)
    validation=json.loads((ROOT/'depth-spread-research-001/gpu-validation.json').read_text())
    assert validation['status']=='passed' and validation['gpu']['status']=='passed'
    names={f'{key}{seed}':f'{prefix}-18000-s{seed}' for key,prefix in
        [('combined','native-motion-mip-depthmask'),('depth003','native-depthguide003'),('depth010','native-depthguide010')]
        for seed in range(3)}
    frozen={};models={}
    for key,name in names.items():
        manifest=json.loads((ROOT/name/'manifest.json').read_text());assert manifest['status']=='complete'
        path=Path(manifest['baked_ply']);frozen[key]=dict(run=name,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        models[key]=read_model(path)
    protocol=dict(started=time.time(),models=frozen,frames='all8 repeatedly inspected development views',
        moments='alpha-normalized first/second moments of camera-z at Gaussian centers',
        relative_spread='sqrt(E[z^2]-E[z]^2)/E[z]; dimensionless',
        inverse_spread='sqrt(E[(1/z)^2]-E[1/z]^2)/E[1/z]',
        ndc_pairwise_proxy='A^2 Var(m), m=far/(far-near)*(1-near/z), near=.2,far=1000; not ray/surfel depth',
        mask='same pixel intersection alpha>.95 across all9 models; no per-model selection',
        smooth_mask='real RGB maximum adjacent channel difference<.05, followed by3x3 erosion; same across models',
        subject='previous frozen bounding boxes, include background; head is top third of box, not a segmentation',
        rgb_error='mean absolute RGB difference in [0,1], evaluated on the same common masks',
        bins='four subject-ROI quartiles of combined seed0 dispersion, shared across all models in each view',
        sources=['https://arxiv.org/html/2403.17888v3#A1','https://github.com/hbb1/diff-surfel-rasterization'],
        caveats=['3D Gaussian-center spread is not true surface thickness or proof of floaters',
            'multiple depths at occlusion boundaries can be physically valid',
            'color edges do not exhaust geometry discontinuities',
            'statistics are descriptive; correlated pixels/frames are not independent samples',
            'all models and references are unchanged; this diagnostic introduces no new training loss'])
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));rows={key:[] for key in names};checks=[]
    for c in [c for c in cameras() if c['is_test']]:
        h,w=c['height'],c['width'];gt=image_tensor(BASE/'hover-sfm-001/dataset/images'/f'{c["img_name"]}.jpg')
        gt_array=gt.cpu().numpy();delta=np.zeros((h,w),dtype='float32')
        dx=np.max(np.abs(gt_array[:,1:]-gt_array[:,:-1]),-1);dy=np.max(np.abs(gt_array[1:]-gt_array[:-1]),-1)
        delta[:,1:]=np.maximum(delta[:,1:],dx);delta[:,:-1]=np.maximum(delta[:,:-1],dx)
        delta[1:]=np.maximum(delta[1:],dy);delta[:-1]=np.maximum(delta[:-1],dy)
        smooth=cv2.erode((delta<.05).astype('uint8'),np.ones((3,3),np.uint8)).astype(bool)
        common=np.ones((h,w),dtype=bool);cache={};target=out/c['img_name'];target.mkdir()
        for key,model in models.items():
            moment,diagnostic=render_moments(model,c,mode='mip')
            rgb,alpha,_=render(model,c,mode='mip',depth=True)
            valid=alpha[...,0]>.95
            reference_error=float((moment['mean_z'][valid]-rgb[...,3][valid]).abs().max())
            # Both are float32 accumulated; allow depth-scaled round-off only.
            relative_error=float(((moment['mean_z'][valid]-rgb[...,3][valid]).abs()/rgb[...,3][valid].clamp_min(.2)).max())
            assert relative_error<2e-5
            checks.append(dict(frame=c['img_name'],model=key,depth_mean_max_absolute_error=reference_error,
                depth_mean_max_relative_error=relative_error,**diagnostic))
            common&=(moment['alpha'].cpu().numpy()>.95)
            cache[key]={field:moment[field].float().cpu().numpy() for field in
                ['relative_std','relative_inverse_std','ndc_pairwise_proxy']}
            cache[key]['rgb_error']=(rgb[...,:3].clamp(0,1)-gt).abs().mean(-1).cpu().numpy()
            if key.endswith('0'):
                Image.fromarray((rgb[...,:3].clamp(0,1).cpu().numpy()*255).round().astype('uint8')).save(target/f'{key}-rgb.png')
        box=SUBJECT_BOX[int(c['img_name'])];subject=region_mask((h,w),box)
        head=region_mask((h,w),(box[0],box[1],box[2],box[1]+(box[3]-box[1])/3))
        regions=dict(full=common,subject=common&subject,head=common&head,
            full_smooth=common&smooth,subject_smooth=common&subject&smooth,head_smooth=common&head&smooth)
        limits=np.quantile(cache['combined0']['relative_std'][regions['subject']],[0,.25,.5,.75,1])
        bins=[regions['subject']&(cache['combined0']['relative_std']>=limits[i])&
            (cache['combined0']['relative_std']<(limits[i+1] if i<3 else np.inf)) for i in range(4)]
        for key,data in cache.items():
            row=dict(frame=c['img_name'],common_opacity_fraction=float(common.mean()),regions={},
                baseline_subject_quartile_limits=limits.tolist())
            for region,mask in regions.items():
                row['regions'][region]=dict(relative_spread=stats(data['relative_std'][mask]),
                    inverse_spread=stats(data['relative_inverse_std'][mask]),
                    mean_ndc_proxy=float(np.mean(data['ndc_pairwise_proxy'][mask])),
                    rgb_mae=float(np.mean(data['rgb_error'][mask])))
            row['baseline_spread_quartiles']=[dict(pixels=int(mask.sum()),
                mean_rgb_mae=float(np.mean(data['rgb_error'][mask])),
                mean_relative_spread=float(np.mean(data['relative_std'][mask]))) for mask in bins]
            rows[key].append(row)
            if key.endswith('0'):
                heatmap(data['relative_std'],common).save(target/f'{key}-spread.png')
                heatmap(data['rgb_error'],common,maximum=.1).save(target/f'{key}-error.png')
                np.savez_compressed(target/f'{key}-maps.npz',relative_std=data['relative_std'],
                    relative_inverse_std=data['relative_inverse_std'],common_mask=common,rgb_mae=data['rgb_error'])
        Image.fromarray((gt_array*255).round().astype('uint8')).save(target/'gt.png')
        Image.fromarray((common*255).astype('uint8')).save(target/'common-mask.png')
        print(json.dumps(dict(frame=c['img_name'],common_fraction=float(common.mean()),
            subject_spread={key:rows[key][-1]['regions']['subject']['relative_spread']['median'] for key in rows})),flush=True)
    aggregate={key:{region:dict(mean_view_median=float(np.mean([r['regions'][region]['relative_spread']['median'] for r in data])),
        mean_view_p90=float(np.mean([r['regions'][region]['relative_spread']['p90'] for r in data])),
        mean_view_above010=float(np.mean([r['regions'][region]['relative_spread']['above010_fraction'] for r in data])),
        mean_view_rgb_mae=float(np.mean([r['regions'][region]['rgb_mae'] for r in data])))
        for region in rows[key][0]['regions']} for key,data in rows.items()}
    assert all(hashlib.sha256(Path(v['path']).read_bytes()).hexdigest()==v['sha256'] for v in frozen.values())
    (out/'summary.json').write_text(json.dumps(dict(status='complete',protocol=protocol,aggregate=aggregate,rows=rows,
        validation=checks,confirmed_floaters=False,models_unchanged=True),indent=2))


if __name__=='__main__':main()
