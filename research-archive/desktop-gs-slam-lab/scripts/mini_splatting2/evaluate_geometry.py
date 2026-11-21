"""SfM agreement and controlled novel paths. Novel paths have no image GT."""
import json
from pathlib import Path
import sys
import numpy as np
import cv2
import torch
from local_refinement import LAB,BASE,ROOT,PLY,read_model,cameras,render,write_json,save_image


def restore(name):
    model=read_model()
    if name=='original':return model,'classic',None
    # Our own checkpoint config stores torch.__version__ as TorchVersion.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        ckpt=torch.load(ROOT/name/'checkpoint.pt',map_location='cuda',weights_only=True)
    for k,v in ckpt['params'].items():model[k].data.copy_(v)
    return model,ckpt['config']['mode'],ckpt['filter_3d']


@torch.no_grad()
def main():
    sys.path.insert(0,str(LAB/'repos/hierarchical-3d-gaussians/preprocess'))
    import read_write_model as rw
    _,images,points=rw.read_model(str(BASE/'hover-sfm-001/dataset/sparse/0'),ext='.bin')
    byname={i.name:i for i in images.values()};cams=cameras()
    audit=json.loads((ROOT/'capture-audit.json').read_text());center=np.array(audit['focus']);up=np.array(audit['approximate_up'])
    ref=cams[24];vec=np.array(ref['position'])-center;radius=np.linalg.norm(vec)
    horizontal=vec-up*np.dot(vec,up);horizontal/=np.linalg.norm(horizontal)
    elevation=np.arcsin(np.dot(vec,up)/radius)
    out=ROOT/'geometry-evaluation';out.mkdir(exist_ok=True)
    paths=[]
    for kind in ['distance','elevation']:
        for i,t in enumerate(np.linspace(0,1,60)):
            ratio=1+.5*np.sin(t*2*np.pi) if kind=='distance' else 1
            delta=np.deg2rad(25)*np.sin(t*2*np.pi) if kind=='elevation' else 0
            p=center+radius*ratio*(np.cos(elevation+delta)*horizontal+np.sin(elevation+delta)*up)
            forward=center-p;forward/=np.linalg.norm(forward)
            right=np.cross(forward,up);right/=np.linalg.norm(right)
            down=np.cross(forward,right);R=np.stack([right,down,forward],axis=1)
            V=np.eye(4);V[:3,:3]=R.T;V[:3,3]=-R.T@p
            c=dict(ref);c['viewmat']=torch.tensor(V,dtype=torch.float32,device='cuda')[None]
            paths.append(dict(kind=kind,index=i,ratio=float(ratio),delta_deg=float(np.rad2deg(delta)),position=p.tolist(),rotation=R.tolist(),camera=c))
    write_json(out/'paths.json',dict(scope='qualitative extrapolation only, no GT; all models share exact cameras',
        frames=[{k:v for k,v in p.items() if k!='camera'} for p in paths]))
    names=['original','classic-refine-s0-r1','mip-refine-s0','depth-refine-s0','mip-depth-refine-s0']
    summaries={}
    for name in names:
        model,mode,filt=restore(name);modelout=out/name;modelout.mkdir(exist_ok=True);rows=[]
        for c in [c for c in cams if c['is_test']]:
            pred,alpha,_=render(model,c,mode=mode,filt=filt,depth=True)
            zmap=pred[...,3].cpu().numpy();amap=alpha[...,0].cpu().numpy()
            im=byname[c['img_name']+'.jpg'];pids=im.point3D_ids
            valid=np.array([p>=0 and p in points and points[p].error<=1 and len(points[p].image_ids)>=4 for p in pids])
            xyz=np.array([points[p].xyz for p in pids[valid]])
            xy=im.xys[valid].astype('float32');R=rw.qvec2rotmat(im.qvec);z=(xyz@R.T+im.tvec)[:,2]
            rendered=cv2.remap(zmap,xy[:,0,None],xy[:,1,None],cv2.INTER_LINEAR).ravel()
            support=cv2.remap(amap,xy[:,0,None],xy[:,1,None],cv2.INTER_LINEAR).ravel()
            good=(z>.2)&(rendered>.2)&(support>.95)
            rel=np.abs(rendered[good]-z[good])/z[good]
            rows.append(dict(frame=c['img_name'],evaluated_points=int(good.sum()),median_relative_z_error=float(np.median(rel)),
                             p90_relative_z_error=float(np.quantile(rel,.9)),over_20pct_fraction=float((rel>.2).mean())))
        for p in paths:
            pred,_,_=render(model,p['camera'],divisor=2,mode=mode,filt=filt)
            d=modelout/p['kind'];d.mkdir(exist_ok=True)
            save_image(d/f'{p["index"]:03d}.png',pred)
        summaries[name]=dict(rows=rows,mean_median_relative_z_error=float(np.mean([r['median_relative_z_error'] for r in rows])),
                             mean_p90_relative_z_error=float(np.mean([r['p90_relative_z_error'] for r in rows])))
        print(name,summaries[name]['mean_median_relative_z_error'],flush=True)
    write_json(out/'sfm-depth-agreement.json',dict(models=summaries,
         evidence_type='agreement with robust sparse COLMAP points in held-out RGB frames, not independently measured dense depth',
         limitations='same SfM calibration used by the reconstruction; misses floaters in untextured or unobserved regions',
         depth_render='alpha-normalized expected camera z, not a surface intersection or median depth'))


if __name__=='__main__':main()
