"""Image-only feature tracks checked against fixed calibrated epipolar geometry.

No learned/estimated Gaussian depth is used. Residuals can reflect camera error,
tracking error, reflections, rolling shutter or nonrigid motion; not proof of motion.
"""
import ast
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

LAB=Path(__file__).resolve().parents[2];ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'
BASE=LAB/'outputs/mini-splatting2-computer00-001'


def fundamental(a,b):
    r1,r2=np.array(a['rotation']),np.array(b['rotation'])
    p1,p2=np.array(a['position']),np.array(b['position'])
    rotation=r2.T@r1;t=r2.T@(p1-p2)
    cross=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
    def intrinsic(c):return np.array([[c['fx'],0,c['width']/2],[0,c['fy'],c['height']/2],[0,0,1]])
    return np.linalg.inv(intrinsic(b)).T@cross@rotation@np.linalg.inv(intrinsic(a))


def summary(values):
    if not len(values):return dict(tracks=0)
    return dict(tracks=len(values),median_px=float(np.median(values)),p90_px=float(np.quantile(values,.9)),
        over1px_fraction=float(np.mean(values>1)),over2px_fraction=float(np.mean(values>2)))


def main():
    out=ROOT/'head-epipolar-audit-001';out.mkdir(exist_ok=False);cv2.setNumThreads(4)
    cameras=sorted(json.loads((BASE/'viewer/cameras.json').read_text()),key=lambda c:c['img_name'])
    assignment=next(n for n in ast.parse(Path(__file__).with_name('local_refinement.py').read_text()).body
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SUBJECT_BOX' for t in n.targets))
    boxes=ast.literal_eval(assignment.value);offsets=[-6,-3,-1,1,3,6]
    pairs=[(index,index+offset) for index in range(0,len(cameras),8) for offset in offsets if 0<=index+offset<len(cameras)]
    protocol=dict(started=time.time(),pairs=pairs,offsets=offsets,units='pixels at calibrated757x1347',
        selection='all8 development anchors with fixed offsets; all pairs retained',
        tracking='Shi-Tomasi corners max5000 quality.01 distance5; pyramidalLK31x31 levels4; forward/backward<.5px',
        appearance_filter='11x11 normalized cross-correlation>.90; patch standard deviation>3/255',
        epipolar='Sampson distance from frozen camera intrinsics/extrinsics; NOT used for selecting tracks',
        regions='fixed source-frame head(top third)/torso(remaining box)/outside-person; regions include some background',
        no_gaussian_depth_used=True,no_camera_update=True,
        caveats=['Residual can be caused by tracking, calibration, reflection, rolling shutter or motion',
            'Outside-person reference can itself contain moving people',
            'This is not independent camera GT or a motion classifier',
            'Region texture and available corner counts differ; report counts and all pairs'])
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));cache={};hashes={}
    def load(index):
        if index not in cache:
            path=BASE/'hover-sfm-001/dataset/images'/f'{cameras[index]["img_name"]}.jpg'
            cache[index]=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
            assert cache[index] is not None;hashes[cameras[index]['img_name']]=hashlib.sha256(path.read_bytes()).hexdigest()
        return cache[index]
    rows=[]
    for source,target in pairs:
        a,b=load(source),load(target);h,w=a.shape
        corners=cv2.goodFeaturesToTrack(a,maxCorners=5000,qualityLevel=.01,minDistance=5,blockSize=5)
        q,status,_=cv2.calcOpticalFlowPyrLK(a,b,corners,None,winSize=(31,31),maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01))
        reverse,backstatus,_=cv2.calcOpticalFlowPyrLK(b,a,q,None,winSize=(31,31),maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01))
        p=corners[:,0];q=q[:,0]
        good=(status[:,0]>0)&(backstatus[:,0]>0)&(np.linalg.norm(reverse[:,0]-p,axis=1)<.5)
        good&=(q[:,0]>8)&(q[:,0]<w-9)&(q[:,1]>8)&(q[:,1]<h-9)&(p[:,0]>8)&(p[:,0]<w-9)&(p[:,1]>8)&(p[:,1]<h-9)
        ncc=np.full(len(p),-1,dtype='float32')
        for idx in np.where(good)[0]:
            patch1=cv2.getRectSubPix(a,(11,11),tuple(p[idx])).astype('float32')
            patch2=cv2.getRectSubPix(b,(11,11),tuple(q[idx])).astype('float32')
            std1,std2=patch1.std(),patch2.std()
            if min(std1,std2)>3:
                ncc[idx]=np.mean((patch1-patch1.mean())*(patch2-patch2.mean()))/(std1*std2)
        good&=ncc>.90;f=fundamental(cameras[source],cameras[target])
        x1=np.c_[p,np.ones(len(p))];x2=np.c_[q,np.ones(len(q))]
        fx=x1@f.T;ftx=x2@f
        distance=np.abs((x2*fx).sum(-1))/np.sqrt((fx[:,:2]**2).sum(-1)+(ftx[:,:2]**2).sum(-1)+1e-20)
        box=boxes[int(cameras[source]['img_name'])];x0,y0,x1b,y1=[v*s for v,s in zip(box,[w,h,w,h])]
        person=(p[:,0]>=x0)&(p[:,0]<x1b)&(p[:,1]>=y0)&(p[:,1]<y1)
        head=person&(p[:,1]<y0+(y1-y0)/3)
        regions=dict(head=head,torso=person&~head,outside_person=~person)
        row=dict(source=cameras[source]['img_name'],target=cameras[target]['img_name'],offset=target-source,
            detected=len(p),tracking_accepted=int(good.sum()),regions={name:summary(distance[mask&good]) for name,mask in regions.items()})
        rows.append(row)
        np.savez_compressed(out/f'{source:06d}-{target:06d}.npz',source_xy=p[good],target_xy=q[good],
            sampson_distance=distance[good],head=head[good],person=person[good],ncc=ncc[good])
        print(json.dumps(row),flush=True)
    aggregate={name:dict(pairs_with_10plus_tracks=sum(r['regions'][name]['tracks']>=10 for r in rows),
        total_tracks=sum(r['regions'][name]['tracks'] for r in rows),
        mean_pair_median_px=float(np.mean([r['regions'][name]['median_px'] for r in rows if r['regions'][name]['tracks']>=10])),
        mean_pair_over1px_fraction=float(np.mean([r['regions'][name]['over1px_fraction'] for r in rows if r['regions'][name]['tracks']>=10])))
        for name in ['head','torso','outside_person']}
    (out/'summary.json').write_text(json.dumps(dict(status='complete',protocol=protocol,rows=rows,aggregate=aggregate,
        image_sha256=hashes,proves_subject_motion=False,models_modified=False),indent=2))
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__':main()
