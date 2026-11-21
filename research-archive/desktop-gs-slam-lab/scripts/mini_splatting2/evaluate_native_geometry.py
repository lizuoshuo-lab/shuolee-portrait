"""Frozen sparse-SfM agreement and predicted-depth contradiction diagnostics.

Neither source is independent measured depth; this is not floater-removal GT.
The unchanged criteria allow comparisons across different Gaussian counts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from local_refinement import LAB,BASE,ROOT,read_model,cameras,render,SUBJECT_BOX


def summarize(errors):
    if not len(errors):return dict(points=0)
    return dict(points=len(errors),median_relative_error=float(np.median(errors)),
        p90_relative_error=float(np.quantile(errors,.9)),over20pct_fraction=float((errors>.2).mean()))


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('names',nargs='+');a=p.parse_args()
    out=ROOT/a.output;out.mkdir(exist_ok=False)
    torch.set_num_threads(4);cv2.setNumThreads(4)
    sys.path.insert(0,str(LAB/'repos/hierarchical-3d-gaussians/preprocess'))
    import read_write_model as rw
    _,images,points=rw.read_model(str(BASE/'hover-sfm-001/dataset/sparse/0'),ext='.bin')
    byname={i.name:i for i in images.values()};cams=cameras();paths={};frozen={}
    for name in a.names:
        m=json.loads((ROOT/name/'manifest.json').read_text());assert m['status']=='complete'
        paths[name]=Path(m.get('baked_ply',m['ply']))
        frozen[name]=dict(path=str(paths[name]),sha256=hashlib.sha256(paths[name].read_bytes()).hexdigest())
    protocol=dict(started=time.time(),models=frozen,
        depth='alpha-normalized expected Gaussian-center camera z',minimum_alpha=.95,
        sparse_points='reprojection error <=1px, track length>=4; report all and point_id%5==0 separately',
        contradiction=dict(confidence=.85,minimum_views=8,front_fraction=.9,max_surface_views=1,
            surface_tolerance=.15,front_threshold=.8,extent='3 times max baked standard deviation'),
        caveats=['SfM is shared with model initialization and camera calibration, not independent geometry GT',
            'point_id%5==0 was excluded from depth scale/shift fitting, but remains in shared SfM initialization',
            'monocular depth agreement is partly an in-training diagnostic for guided models, not independent acceptance',
            'contradiction counts are not established floaters, and no Gaussian is modified by this audit'])
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));results={}
    for name,path in paths.items():
        model=read_model(path);mode='mip' if 'baked' in path.name else 'classic';rows=[]
        for c in [c for c in cams if c['is_test']]:
            pred,alpha,_=render(model,c,mode=mode,depth=True)
            zm=pred[...,3].cpu().numpy();am=alpha[...,0].cpu().numpy()
            im=byname[c['img_name']+'.jpg'];ids=np.array(im.point3D_ids)
            valid=np.array([p>=0 and p in points and points[p].error<=1 and len(points[p].image_ids)>=4 for p in ids])
            ids=ids[valid];xy=im.xys[valid].astype('float32')
            xyz=np.array([points[p].xyz for p in ids]);R=rw.qvec2rotmat(im.qvec);z=(xyz@R.T+im.tvec)[:,2]
            observed=cv2.remap(zm,xy[:,0,None],xy[:,1,None],cv2.INTER_LINEAR).ravel()
            opacity=cv2.remap(am,xy[:,0,None],xy[:,1,None],cv2.INTER_LINEAR).ravel()
            good=(z>.2)&(observed>.2)&(opacity>.95)
            rel=np.abs(observed-z)/np.maximum(z,1e-8)
            box=SUBJECT_BOX[int(c['img_name'])]
            person=(xy[:,0]>=box[0]*c['width'])&(xy[:,0]<box[2]*c['width'])&(xy[:,1]>=box[1]*c['height'])&(xy[:,1]<box[3]*c['height'])
            rows.append(dict(frame=c['img_name'],all=summarize(rel[good]),prior_fit_reserved=summarize(rel[good&(ids%5==0)]),
                subject_roi=summarize(rel[good&person]),projected_sparse_points=len(ids)))
        n=len(model['means']);observed=torch.zeros(n,device='cuda');front=torch.zeros_like(observed);surface=torch.zeros_like(observed)
        extent=3*model['scales'].exp().max(-1).values
        for c in [c for c in cams if not c['is_test']]:
            with np.load(ROOT/'depth-priors'/f'{c["img_name"]}.npz') as d:
                prior=torch.tensor(np.stack([d['inverse_depth'],d['weight']]).astype('float32'),device='cuda')[None]
            motion=torch.tensor(np.asarray(Image.open(ROOT/'motion-masks-001'/f'{c["img_name"]}.png').convert('L')).copy(),device='cuda')
            prior[0,1]*=(motion==0)
            V=c['viewmat'][0];pc=model['means']@V[:3,:3].T+V[:3,3];z=pc[:,2]
            u=pc[:,0]/z.clamp_min(.001)*c['fx']+c['width']/2;v=pc[:,1]/z.clamp_min(.001)*c['fy']+c['height']/2
            grid=torch.stack([2*u/(c['width']-1)-1,2*v/(c['height']-1)-1],-1)[None,None]
            sample=F.grid_sample(prior,grid,align_corners=True,padding_mode='zeros')[0,:,0]
            depth=sample[0].clamp_min(1e-5).reciprocal()
            valid=(z>.2)&(u>3)&(u<c['width']-4)&(v>3)&(v<c['height']-4)&(sample[1]>.85)
            observed+=valid;front+=valid&((z+extent)<.8*depth);surface+=valid&((z-depth).abs()<.15*depth)
        selected=(observed>=8)&(front/observed.clamp_min(1)>=.9)&(surface<=1)
        opacity=model['opacities'].sigmoid();count=int(selected.sum())
        contradiction=dict(count=count,total=n,fraction=count/n,
            opacity_mass_fraction=float(opacity[selected].sum()/opacity.sum()),
            above005_opacity_count=int((selected&(opacity>.05)).sum()),confirmed_floaters=False)
        result=dict(rows=rows,prior_contradiction=contradiction,
            view_mean_median_relative_error=float(np.mean([r['all']['median_relative_error'] for r in rows])),
            reserved_mean_median_relative_error=float(np.mean([r['prior_fit_reserved']['median_relative_error'] for r in rows])),
            subject_mean_median_relative_error=float(np.mean([r['subject_roi']['median_relative_error'] for r in rows])))
        results[name]=result;(out/f'{name}.json').write_text(json.dumps(result,indent=2))
        print(name,json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
        del model;torch.cuda.empty_cache()
    assert all(hashlib.sha256(paths[k].read_bytes()).hexdigest()==frozen[k]['sha256'] for k in paths)
    (out/'summary.json').write_text(json.dumps(dict(status='complete',models=results,protocol=protocol),indent=2))

if __name__=='__main__':main()
