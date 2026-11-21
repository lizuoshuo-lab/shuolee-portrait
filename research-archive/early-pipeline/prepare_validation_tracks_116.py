"""Freeze image-only measurements before P1 camera refinement; CPU only."""
from pathlib import Path
import collections
import datetime as dt
import hashlib
import json
import os
import signal
import struct
import time
import traceback
import cv2
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw, ImageFont

B=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
S=B/'current/full-scene-localization-074'
P=B/'current/avatar-repository-pair-114/smplx/data'
O=B/'current/portrait-alignment-open-source-116/pixsfm'
R=O/'independent-validation'
LABELS=['background','face_lower','face_upper','torso','legs_shoes']

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8');t.replace(p)
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def loc(p):
    s=str(p).replace('\\','/');return Path('/mnt/'+s[0].lower()+'/'+s[3:]) if len(s)>1 and s[1]==':' else Path(s)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def patch(im,p):
    a=cv2.getRectSubPix(im,(13,13),tuple(map(float,p))).astype(np.float64);a-=a.mean()
    return a/max(np.linalg.norm(a),1e-8)
def raw_from_rect(xy,K,intr):
    q=(xy+.5-K[:2,2])/[K[0,0],K[1,1]]
    return q*(1+intr[3]*(q*q).sum(-1,keepdims=True))*intr[0]+intr[1:3]-.5
def alarm(*_):raise TimeoutError('Registered 30-minute independent observation budget reached')

def main():
    assert not R.exists(),'Do not regenerate an existing validation bank after looking at results.'
    assert not (O/'ba-vgg-candidate').exists(),'Validation must be frozen before the new BA candidate.'
    reg=read(O/'validation-preregistration.json');assert sha(__file__)==reg['script_sha256']
    R.mkdir();start=time.monotonic();signal.signal(signal.SIGALRM,alarm);signal.alarm(1800)
    cv2.setNumThreads(4)
    def status(stage,**data):save(R/'progress.json',dict(updated=now(),pid=os.getpid(),state='running',stage=stage,**data))
    try:
        source=read(S/'cameras.json');intr=np.array(source['intrinsics']);rows=read(P/'training-inputs.json')['records'];K=np.array(rows[0]['K'])
        assert [r['source_frame_index'] for r in rows]==list(range(1,176))
        old=B/'current/shared-point-bundle-convergence-101/input.npz'
        with np.load(old) as z:oldobs=z['observations'].copy()
        oldobs[:,1:]=raw_from_rect(oldobs[:,1:],K,intr)
        oldtrees=[cKDTree(oldobs[oldobs[:,0]==i,1:]) for i in range(175)]
        face=np.load(P/'observations.npz')['keypoints'][:,23:91]
        yy,xx=np.mgrid[:1352,:760];rawgrid=np.stack([xx*2+.5,yy*2+.5],-1).reshape(-1,2)
        n=(rawgrid+.5-intr[1:3])/intr[0];u=n.copy()
        for _ in range(12):u=n/(1+intr[3]*(u*u).sum(1,keepdims=True))
        rect=u*[K[0,0],K[1,1]]+K[:2,2]-.5
        mx=rect[:,0].reshape(1352,760).astype(np.float32);my=rect[:,1].reshape(1352,760).astype(np.float32)
        images=[];regions=[];hashes={}
        status('decode_images_and_image_defined_regions')
        for i,row in enumerate(rows):
            path=S/'images'/f'raw_{i+1:06d}.png';raw=cv2.imread(str(path),0);assert raw.shape==(2704,1520)
            hashes[path.name]=sha(path);images.append(cv2.resize(raw,(760,1352),interpolation=cv2.INTER_AREA))
            mask=cv2.imread(str(loc(row['mask'])),0);assert mask.shape==(2694,1514)
            fg=cv2.remap(mask,mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)>127
            inside=cv2.erode(fg.astype(np.uint8),np.ones((7,7),np.uint8))>0
            outside=cv2.erode((~fg).astype(np.uint8),np.ones((11,11),np.uint8))>0
            y=np.where(mask>127)[0];top,ht=y.min(),np.ptp(y)+1
            region=np.full(fg.shape,-1,np.int8);region[outside]=0
            region[inside&(my>=top+.23*ht)&(my<top+.64*ht)]=3
            region[inside&(my>=top+.64*ht)]=4
            low=face[i].min(0);high=face[i].max(0);split=np.median(face[i,27:36,1])
            facial=inside&(mx>=low[0]-5)&(mx<=high[0]+5)&(my>=low[1]-5)&(my<=high[1]+5)
            region[facial&(my>=split)]=1;region[facial&(my<split)]=2;regions.append(region)
            if (i+1)%25==0:status('decode_images_and_image_defined_regions',frames=i+1)
        lk=dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.001),minEigThreshold=1e-5)
        tracks=[];rejects=collections.Counter();counts=[]
        for seed in reg['zero_based_seeds']:
            for group in range(5):
                cand=cv2.goodFeaturesToTrack(images[seed],maxCorners=reg['max_corners_per_region'][group],qualityLevel=.01,minDistance=5,
                    mask=(regions[seed]==group).astype(np.uint8)*255,blockSize=5)
                if cand is None:counts.append(dict(seed=seed,group=group,seeds=0));continue
                xy=cand[:,0]*2+.5;distance=oldtrees[seed].query(xy)[0]
                keep=distance>reg['old_bank_seed_exclusion_raw_px'];rejects['seed_near_old_observation']+=int((~keep).sum());cand=cand[keep]
                counts.append(dict(seed=seed,group=group,seeds=len(cand)))
                if not len(cand):continue
                histories=[{seed:p[0].copy()} for p in cand]
                for direction in [-1,1]:
                    active=np.arange(len(cand));prev=cand.copy()
                    for step in range(1,13):
                        f=seed+direction*step;pf=f-direction
                        if not 0<=f<175:break
                        nxt,st,_=cv2.calcOpticalFlowPyrLK(images[pf],images[f],prev,None,**lk)
                        back,sb,_=cv2.calcOpticalFlowPyrLK(images[f],images[pf],nxt,None,**lk)
                        valid=(st[:,0]>0)&(sb[:,0]>0)&(np.linalg.norm(back[:,0]-prev[:,0],axis=1)<=.5)
                        rejects['adjacent_forward_backward']+=int((~valid).sum())
                        # Direct seed-to-current and reverse checks do not use cameras or geometry.
                        if step%4==0:
                            direct,sd,_=cv2.calcOpticalFlowPyrLK(images[seed],images[f],cand[active],None,**lk)
                            rev,sr,_=cv2.calcOpticalFlowPyrLK(images[f],images[seed],direct,None,**lk)
                            cyc=(sd[:,0]>0)&(sr[:,0]>0)&(np.linalg.norm(direct[:,0]-nxt[:,0],axis=1)<=.75)&(np.linalg.norm(rev[:,0]-cand[active,0],axis=1)<=.75)
                            rejects['direct_seed_cycle']+=int((valid&~cyc).sum());valid&=cyc
                        for j in np.flatnonzero(valid):
                            x,y=np.rint(nxt[j,0]).astype(int)
                            if not(7<=x<753 and 7<=y<1345 and regions[f][y,x]==group):
                                valid[j]=False;rejects['image_region_or_border']+=1;continue
                            if np.sum(patch(images[pf],prev[j,0])*patch(images[f],nxt[j,0]))<.85:
                                valid[j]=False;rejects['adjacent_patch_ncc']+=1
                        for j in np.flatnonzero(valid):histories[active[j]][f]=nxt[j,0].copy()
                        active=active[valid];prev=nxt[valid]
                        if not len(active):break
                for j,h in enumerate(histories):
                    if sum(f<seed for f in h)<5 or sum(f>seed for f in h)<5 or len(h)<15:
                        rejects['insufficient_bidirectional_support']+=1;continue
                    ob=np.array([(f,*h[f]) for f in sorted(h)],float);ob[:,1:]=ob[:,1:]*2+.5
                    held=np.zeros(len(ob),bool);held[len(ob)//3:2*len(ob)//3]=True
                    tracks.append(dict(seed=seed,group=group,observations=ob,held=held,seed_old_distance=float(distance[keep][j])))
            status('track_images_only',seed=seed,tracks=len(tracks));print(json.dumps(dict(seed=seed,tracks=len(tracks))),flush=True)
        assert tracks,'No accepted image tracks; stop without camera optimization.'
        off=np.cumsum([0]+[len(t['observations']) for t in tracks]);ob=np.concatenate([t['observations'] for t in tracks]);held=np.concatenate([t['held'] for t in tracks])
        np.savez_compressed(R/'tracks.npz',observations=ob,offsets=off,held=held,groups=[t['group'] for t in tracks],seeds=[t['seed'] for t in tracks])
        # Remove every source 3D track whose observed center is close to this bank, before BA selection.
        # This prevents direct reuse of the validation material tracks; shared image/CNN receptive fields remain a limitation.
        trees={i+1:cKDTree(ob[ob[:,0]==i,1:]+.5) for i in range(175) if np.any(ob[:,0]==i)}
        excluded=set();support={};status('reserve_disjoint_source_track_ids')
        with (S/'refined_sparse/images.bin').open('rb') as f:
            for _ in range(struct.unpack('<Q',f.read(8))[0]):
                iid,*_=struct.unpack('<i7di',f.read(64));name=bytearray()
                while True:
                    ch=f.read(1)
                    if ch==b'\0':break
                    assert ch;name.extend(ch)
                n=struct.unpack('<Q',f.read(8))[0];p=np.frombuffer(f.read(n*24),dtype=[('xy','<f8',(2,)),('pid','<i8')])
                assert int(Path(name.decode()).stem.split('_')[-1])==iid
                if iid in trees:
                    close=trees[iid].query(p['xy'])[0]<=reg['source_track_exclusion_raw_px']
                    excluded.update(int(x) for x in p['pid'][close] if x>=0)
        np.save(R/'excluded-sfm-point-ids.npy',np.array(sorted(excluded),np.int64))
        for group,label in enumerate(LABELS):
            ix=[i for i,t in enumerate(tracks) if t['group']==group]
            support[label]=dict(tracks=len(ix),observations=sum(len(tracks[i]['observations']) for i in ix),held_observations=sum(int(tracks[i]['held'].sum()) for i in ix),
                 represented_seed_blocks=sorted(set(tracks[i]['seed'] for i in ix)))
        # Fixed coverage review: first admitted track per region in the first, middle, last supported block.
        selection=[]
        for group,label in enumerate(LABELS):
            blocks=sorted(set(t['seed'] for t in tracks if t['group']==group))
            if not blocks:continue
            for seed in sorted(set(blocks[j] for j in [0,len(blocks)//2,len(blocks)-1])):
                tid=next(i for i,t in enumerate(tracks) if t['group']==group and t['seed']==seed);selection.append(tid)
        font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',17);panels=[]
        for tid in selection:
            t=tracks[tid];a=t['observations'];ii=np.unique(np.round(np.linspace(0,len(a)-1,5)).astype(int))
            panel=Image.new('RGB',(1250,300),'white');d=ImageDraw.Draw(panel)
            d.text((8,5),f'New validation track {tid} | {LABELS[t["group"]]} | seed raw {t["seed"]+1:06d}',font=font,fill='black')
            for col,k in enumerate(ii):
                fr,x,y=a[k];box=np.round([x-65,y-65,x+65,y+65]).astype(int)
                with Image.open(S/'images'/f'raw_{int(fr)+1:06d}.png') as im:crop=im.convert('RGB').crop(tuple(box)).resize((234,234),Image.Resampling.LANCZOS)
                panel.paste(crop,(col*250+8,45));px,py=(np.array([x,y])-box[:2])*1.8+[col*250+8,45]
                d.ellipse((px-6,py-6,px+6,py+6),outline='#00b6ff',width=2)
                d.text((col*250+8,27),f'raw {int(fr)+1:06d}'+(' | held' if t['held'][k] else ' | point fit'),font=font,fill='black')
            panels.append(panel)
        figures=[]
        for startidx in range(0,len(panels),5):
            sub=panels[startidx:startidx+5];board=Image.new('RGB',(1250,300*len(sub)),'white')
            for j,im in enumerate(sub):board.paste(im,(0,300*j))
            name=f'validation-tracks-{startidx//5+1}.jpg';board.save(R/name,quality=95);figures.append(name)
        data=dict(completed=now(),elapsed_seconds=time.monotonic()-start,tracks=len(tracks),observations=len(ob),held_observations=int(held.sum()),
            groups=support,rejections=dict(rejects),seed_counts=counts,source_images_sha256=hashes,
            previous_validation_bank_sha256=sha(old),tracks_sha256=sha(R/'tracks.npz'),excluded_sfm_point_count=len(excluded),
            exclusion_sha256=sha(R/'excluded-sfm-point-ids.npy'),visual_sample_track_ids=selection,figures=figures,
            visual_review='pending',source_coordinates='OpenCV centers on original distorted1520x2704 images; plus0.5 gives COLMAP coordinates.',
            coverage_screen_passed=all(support[g]['tracks']>=10 and len(support[g]['represented_seed_blocks'])>=3 for g in LABELS),
            camera_error_used_in_feature_admission=False,camera_optimization_started=False,
            independence='New image-only observations reserved before P1 BA, >6rawpx from old validation observations at seed. Every nearby original SfM point ID is excluded from the candidate solve. Same original video is reused; this is not new-image or physical-GT testing. CNN receptive fields may overlap.',
            limits=['KLT is a correspondence estimate, not ground truth. Forward/backward, cycle, region and patch checks cannot rule out all reflection, occlusion, deformation or drift.',
                    'A15-25frame track tests local view consistency; it cannot certify wide-baseline correspondence or full360degree geometry.'])
        save(R/'result.json',data);save(R/'progress.json',dict(updated=now(),state='completed',pid=os.getpid(),stage='frozen_validation_awaits_visual_review'))
        print(json.dumps({k:data[k] for k in ['tracks','observations','groups','excluded_sfm_point_count','coverage_screen_passed','elapsed_seconds']}),flush=True)
    except BaseException as e:
        save(R/'failure.json',dict(updated=now(),error=str(e),traceback=traceback.format_exc()));save(R/'progress.json',dict(updated=now(),state='failed',pid=os.getpid(),error=str(e)));raise
    finally:signal.alarm(0)

if __name__=='__main__':main()
