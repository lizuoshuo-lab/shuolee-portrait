"""One preregistered descriptor-based background validation fallback; no fit."""
from pathlib import Path
import datetime as dt, hashlib, json, time
import cv2
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw
H=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/portrait-alignment-open-source-116/hsfm')
O=H.parent;B=O.parent.parent;J=H/'joint-diagnostic-reserved'
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(J/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
    assert not (J/'background-result.json').exists(),'One fixed descriptor attempt only'
    t0=time.monotonic();cv2.setNumThreads(4)
    rec=read(H/'input-pilot/inputs.json')['records'];rr={r['raw_frame']:r for r in rec};ids=sorted(rr)
    reg=dict(created=now(),reason='KLT validation across12-13frame endpoint gaps has6 background tracks in1block after fixed prior-bank exclusion. Foreground coverage passed. This blocks camera fitting, not evidence that HSfM failed.',
        one_bounded_fallback=True,method='OpenCV SIFT,4000maxfeatures, default detector thresholds; mutual L2ratio0.7, exact three-view descriptor cycle',
        region='Background outside target mask eroded11pixels at original resolution; exclude prior-bank neighborhoods before feature detection',
        masks_not_scene_truth='Other moving people remain; inspect original RGB before using as validation. Not physicalGT.',
        no_camera_model_or_residual_admission=True,prior_exclusion_px=6,
        acceptance='>=10new background tracks spanning>=3triplets, manually review first/middle/last supported blocks',
        stop='If coverage fails, stop P3camera pilot as unavailable under this frozen diagnostic; no threshold sweep or new download',
        frozen_foreground_sha256=sha(J/'validation-tracks.npz'),script_sha256=sha(Path(__file__)),budget_cpu_seconds=600)
    if (J/'background-protocol.json').exists():
        assert not (J/'background-tracks.npz').exists()
        save('background-path-repair.json',dict(updated=now(),reason='Initial attempt failed before reading any image: one excess parent in source root path. No inference, matches or results existed.',previous_script_sha256=read(J/'background-protocol.json')['script_sha256'],repaired_script_sha256=sha(Path(__file__))))
    else:save('background-protocol.json',reg)
    S=B/'current/full-scene-localization-074';intr=np.array(read(S/'cameras.json')['intrinsics'])
    K=np.array(rec[0]['K_from_existing_manifest']);K[:2,2]-=.5
    Kr=np.array([[intr[0],0,intr[1]-.5],[0,intr[0],intr[2]-.5],[0,0,1.]])
    p1=np.load(O/'pixsfm/independent-validation/tracks.npz')['observations'].copy()
    p1[:,1:]=cv2.undistortPoints(p1[:,1:].reshape(-1,1,2),Kr,np.array([intr[3],0,0,0,0.]),P=K).reshape(-1,2)
    old=np.load(B/'current/shared-point-bundle-convergence-101/input.npz')['observations']
    sift=cv2.SIFT_create(nfeatures=4000);features={}
    for f in ids:
        r=rr[f];assert sha(Path(r['image']))==r['image_sha256']
        img=cv2.imread(r['image'],0);fg=cv2.imread(r['mask'],0)>127
        mask=cv2.erode((~fg).astype(np.uint8)*255,np.ones((11,11),np.uint8))
        for xy in np.concatenate([p1[p1[:,0]==f-1,1:],old[old[:,0]==f-1,1:]],axis=0):cv2.circle(mask,tuple(np.rint(xy).astype(int)),7,0,-1)
        kp,de=sift.detectAndCompute(img,mask);features[f]=(np.array([k.pt for k in kp]),de)
    bf=cv2.BFMatcher(cv2.NORM_L2);maps={}
    def match(a,b):
        if (a,b) not in maps:
            ab=bf.knnMatch(features[a][1],features[b][1],k=2);ba=bf.knnMatch(features[b][1],features[a][1],k=2)
            aa={m.queryIdx:m.trainIdx for m,n in ab if m.distance<.7*n.distance}
            bb={m.queryIdx:m.trainIdx for m,n in ba if m.distance<.7*n.distance}
            maps[a,b]={i:j for i,j in aa.items() if bb.get(j)==i}
        return maps[a,b]
    tracks=[];stats=[]
    for i in range(1,len(ids)-1):
        a,b,c=ids[i-1:i+2];ab=match(a,b);bc=match(b,c);ac=match(a,c)
        n=0
        for ia,ib in ab.items():
            ic=bc.get(ib)
            if ic is None or ac.get(ia)!=ic:continue
            tracks.append((a,b,c,np.stack([features[a][0][ia],features[b][0][ib],features[c][0][ic]])));n+=1
        stats.append(dict(frames=[a,b,c],cycle_tracks=n))
        assert time.monotonic()-t0<600
    if tracks:np.savez_compressed(J/'background-tracks.npz',frames=[x[:3] for x in tracks],xy=np.stack([x[3] for x in tracks]),groups=np.zeros(len(tracks),int))
    support=sorted(set(t[1] for t in tracks));selection=[]
    for f in sorted(set(support[i] for i in [0,len(support)//2,len(support)-1])) if support else []:
        # Three spatially separated examples per block, selected without camera error.
        indices=[j for j,t in enumerate(tracks) if t[1]==f]
        for j in np.unique(np.round(np.linspace(0,len(indices)-1,min(3,len(indices)))).astype(int)):selection.append(indices[j])
    panels=[]
    for ti in selection:
        a,b,c,xy=tracks[ti];im=Image.new('RGB',(720,266),'white');d=ImageDraw.Draw(im)
        d.text((8,4),f'New background probe{ti} | middle held | image-only mutualSIFTcycle',fill='black')
        for col,(f,q) in enumerate(zip((a,b,c),xy)):
            box=np.rint(np.r_[q-70,q+70]).astype(int);crop=Image.open(rr[f]['image']).convert('RGB').crop(tuple(box)).resize((224,224),Image.Resampling.LANCZOS)
            im.paste(crop,(240*col+8,36));d.text((240*col+8,20),f'raw{f:06d}',fill='black');p=(q-box[:2])*1.6+[240*col+8,36]
            d.ellipse((p[0]-5,p[1]-5,p[0]+5,p[1]+5),outline='#00ccff',width=2)
        panels.append(im)
    for k in range(0,len(panels),3):
        ps=panels[k:k+3];im=Image.new('RGB',(720,266*len(ps)),'white')
        for j,p in enumerate(ps):im.paste(p,(0,j*266))
        im.save(J/f'background-review-{k//3+1}.jpg',quality=95)
    result=dict(updated=now(),tracks=len(tracks),blocks=support,coverage_passed=len(tracks)>=10 and len(support)>=3,
        counts=stats,feature_counts={f:len(features[f][0]) for f in ids},seconds=time.monotonic()-t0,visual_samples=selection,
        sha256=sha(J/'background-tracks.npz') if tracks else None,fit_started=False,visual_review='pending')
    save('background-result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
