"""Common estimated-geometry masks for no-GT path continuity diagnostics.

These diagnostics do not certify geometry and can favor blur. Human A/B
judgments remain separate; no synthetic subjective score is invented.
"""
import json
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'
OUT=ROOT/'difix-novel-paths-001'


def load_rgb(kind,name):return np.asarray(Image.open(OUT/kind/(name+'.png')).convert('RGB'),dtype='float32')/255


def correspondence(target,source,zt,zs,at,ass):
    h,w=zt.shape; yy,xx=np.mgrid[:h,:w]
    points=np.stack([(xx-w/2)/target['fx']*zt,(yy-h/2)/target['fy']*zt,zt],-1).reshape(-1,3)
    V=np.array(target['viewmat']);W=np.array(source['viewmat'])
    world=(points-V[:3,3])@V[:3,:3]
    other=world@W[:3,:3].T+W[:3,3]
    z=other[:,2].reshape(h,w)
    u=(other[:,0]/np.maximum(other[:,2],1e-6)*source['fx']+w/2).reshape(h,w).astype('float32')
    v=(other[:,1]/np.maximum(other[:,2],1e-6)*source['fy']+h/2).reshape(h,w).astype('float32')
    def sample(x):return cv2.remap(x,u,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    valid=(z>.2)&(u>2)&(u<w-3)&(v>2)&(v<h-3)&(at>.95)&(sample(ass)>.95)&(np.abs(sample(zs)-z)/np.maximum(z,.2)<.05)
    return sample,valid


def main():
    cv2.setNumThreads(4)
    report=json.loads((OUT/'summary.json').read_text());assert report['status']=='complete_pending_visual_and_temporal_review'
    cams=json.loads((OUT/'cameras.json').read_text());rows=[];texture=[]
    for kind in ['elevation','distance']:
        path=[c for c in cams if c['kind']==kind]
        for gap in [1,5]:
            for i in range(len(path)-gap):
                target,source=path[i],path[i+gap]
                with np.load(OUT/'depth'/(target['name']+'.npz')) as a,np.load(OUT/'depth'/(source['name']+'.npz')) as b:
                    sample,valid=correspondence(target,source,a['depth'],b['depth'],a['alpha'],b['alpha'])
                row=dict(kind=kind,gap=gap,frame=target['index'],common_valid_fraction=float(valid.mean()))
                for arm in ['input','output']:
                    rgb=load_rgb(arm,target['name']);other=sample(load_rgb(arm,source['name']))
                    errors=np.mean(np.abs(rgb-other),-1)
                    row[arm]=dict(warp_mae=float(errors[valid].mean()) if valid.any() else None,
                                  warp_p95=float(np.quantile(errors[valid],.95)) if valid.any() else None)
                rows.append(row)
        for c in path:
            entry=dict(kind=kind,frame=c['index'])
            for arm in ['input','output']:
                gray=cv2.cvtColor(load_rgb(arm,c['name']),cv2.COLOR_RGB2GRAY)
                gx=cv2.Sobel(gray,cv2.CV_32F,1,0);gy=cv2.Sobel(gray,cv2.CV_32F,0,1)
                entry[arm]=dict(edge_energy=float(np.mean(gx*gx+gy*gy)))
            texture.append(entry)
    aggregate={}
    for kind in ['elevation','distance']:
        aggregate[kind]={}
        for gap in [1,5]:
            subset=[r for r in rows if r['kind']==kind and r['gap']==gap]
            aggregate[kind][str(gap)]={arm:float(np.mean([r[arm]['warp_mae'] for r in subset if r[arm]['warp_mae'] is not None])) for arm in ['input','output']}
            aggregate[kind][str(gap)]['common_valid_fraction']=float(np.mean([r['common_valid_fraction'] for r in subset]))
    result=dict(status='complete',created=time.time(),aggregate=aggregate,rows=rows,texture=texture,
        geometry_updated=False,no_gt=True,human_ratings_collected=0,
        interpretation='Lower warp error means greater agreement under the original estimated geometry; blur can lower this error. Edge energy can increase from noise. Neither replaces subjective clarity or proves correct unseen geometry.')
    target=OUT/'continuity.json';assert not target.exists();target.write_text(json.dumps(result,indent=2))
    print(json.dumps(aggregate),flush=True)


if __name__=='__main__':main()
