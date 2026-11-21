"""One bounded native-pixel background motion experiment, not a fitting input."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np
from PIL import Image

from update_leafwoman129_state import ROOT, NATIVE, atomic, read, sha

BASE=ROOT/'leafwoman-fit40-132'
OUT=BASE/'background-motion135'
VIDEO=Path('D:/00/工作/数据集/video22.mp4')
CONTACT=NATIVE/'source-pose-review/source-contacts.json'
W,H=2704,1520
K=np.array([[1120.8319106618264,0,1352.8626255714278],
            [0,1119.820634890859,763.7770175739016],[0,0,1.]])
I=np.linalg.inv(K)
E_THRESH=1.5
H_THRESH=3.
T_LIMIT=180


def decodes(indices):
    expression='+'.join(r'eq(n\,%d)'%i for i in indices)
    command=[shutil.which('ffmpeg'),'-nostdin','-v','error','-hwaccel','none','-threads','2','-noautorotate',
             '-i',str(VIDEO),'-filter_threads','1','-vf','select='+expression,'-fps_mode','passthrough',
             '-frames:v',str(len(indices)),'-pix_fmt','rgb24','-f','rawvideo','pipe:1']
    with subprocess.Popen(command,stdout=subprocess.PIPE) as process:
        try:
            for i in indices:
                buf=process.stdout.read(W*H*3)
                if len(buf)!=W*H*3:raise RuntimeError(f'Incomplete native decode at index {i}')
                yield np.frombuffer(buf,np.uint8).reshape(H,W,3)
            if process.wait(timeout=10):raise RuntimeError('FFmpeg failed')
        finally:
            if process.poll() is None:process.kill();process.wait()


def features(image, mask, sift, band):
    key,desc=sift.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),mask)
    order=sorted(range(len(key)),key=lambda j:(-key[j].response,*key[j].pt,key[j].size))
    grid=defaultdict(list);kept=[]
    for j in order:
        x,y=key[j].pt;radius=6*key[j].size
        if x+radius >= band[0] and x-radius <= band[1]:continue
        cx,cy=int(x//8),int(y//8)
        if any((x-u)**2+(y-v)**2 < 64 for dx in (-1,0,1) for dy in (-1,0,1)
               for u,v in grid[(cx+dx,cy+dy)]):continue
        grid[(cx,cy)].append((x,y));kept.append(j)
        if len(kept)==2400:break
    xy=np.array([key[j].pt for j in kept],dtype=np.float64).reshape(-1,2)
    if not len(kept):return xy,np.empty((0,128),np.float32)
    d=desc[kept];return xy,np.sqrt(d/np.maximum(d.sum(1,keepdims=True),1e-12))


def mutual(a,b):
    if min(len(a),len(b))<2:return {}
    bf=cv2.BFMatcher(cv2.NORM_L2)
    fw,bw=(bf.knnMatch(x,y,k=2) for x,y in ((a,b),(b,a)))
    back={u.queryIdx:u.trainIdx for u,v in bw if u.distance < .75*v.distance}
    return {u.queryIdx:u.trainIdx for u,v in fw if u.distance < .75*v.distance and back.get(u.trainIdx)==u.queryIdx}


def tracks(matches,n):
    roots={}
    def find(x):
        roots.setdefault(x,x)
        if roots[x]!=x:roots[x]=find(roots[x])
        return roots[x]
    for i in range(n-2):
        a,b,c=matches[i,i+1],matches[i+1,i+2],matches[i,i+2]
        for x,y in a.items():
            z=b.get(y)
            if z is None or c.get(x)!=z:continue
            f=find((i,x))
            roots[find((i+1,y))]=f;roots[find((i+2,z))]=f
    groups=defaultdict(list)
    for node in roots:groups[find(node)].append(node)
    clean=[sorted(v) for v in groups.values() if len(v)>=3 and len(v)==len({i for i,_ in v})]
    clean.sort()
    return [(dict(nodes),int(hashlib.sha256(repr(nodes).encode()).hexdigest()[:8],16)%5==0)
            for nodes in clean],len(groups)-len(clean)


def homogeneous(p):return np.c_[p,np.ones(len(p))]
def norm(p):
    h=homogeneous(p)@I.T;return h[:,:2]/h[:,2:]
def apply_h(mat,p):
    h=homogeneous(p)@mat.T
    with np.errstate(invalid='ignore',divide='ignore'):return h[:,:2]/h[:,2:]
def stats(e):
    good=e[np.isfinite(e)]
    return dict(n=len(e),finite=len(good),median=float(np.median(good)) if len(good) else None,
                p90=float(np.quantile(good,.9)) if len(good) else None)
def cells(a):return len({(int(x*4/W),int(y*3/H)) for x,y in a})


def diagnose(a,b,held):
    fit=~held;h=int(held.sum());f=int(fit.sum())
    row=dict(fit_count=f,held_count=h,held_cells=[cells(a[held]),cells(b[held])],admitted=False)
    if f<50 or h<12 or min(row['held_cells'])<3:
        row['reason']='insufficient_independent_or_spatial_support';return row
    cv2.setRNGSeed(135)
    x,y=norm(a),norm(b)
    E,mask=cv2.findEssentialMat(x[fit],y[fit],np.eye(3),method=cv2.RANSAC,prob=.999,
        threshold=E_THRESH/np.mean([K[0,0],K[1,1]]),maxIters=4000)
    Hm,_=cv2.findHomography(a[fit],b[fit],cv2.RANSAC,H_THRESH,maxIters=4000,confidence=.999)
    if E is None or E.shape!=(3,3) or Hm is None or abs(np.linalg.det(Hm))<1e-12:
        row['reason']='missing_unique_E_or_H';return row
    n,R,t,_=cv2.recoverPose(E,x[fit],y[fit],np.eye(3),mask=mask.copy())
    t=t[:,0]
    A,B=a[held],b[held]
    F=I.T@E@I
    fa=homogeneous(A)@F.T;fb=homogeneous(B)@F
    sampson=abs((homogeneous(B)*fa).sum(1))/np.maximum(np.sqrt((fa[:,:2]**2+fb[:,:2]**2).sum(1)),1e-15)
    transfer=np.maximum(np.linalg.norm(apply_h(Hm,A)-B,axis=1),np.linalg.norm(apply_h(np.linalg.inv(Hm),B)-A,axis=1))
    X=cv2.triangulatePoints(np.eye(3,4),np.c_[R,t],x[held].T,y[held].T).T
    with np.errstate(invalid='ignore',divide='ignore'):
        X=X[:,:3]/X[:,3:];q=X@R.T+t
        error=np.maximum(np.linalg.norm((X@K.T)[:,:2]/X[:,2:]-A,axis=1),
                         np.linalg.norm((q@K.T)[:,:2]/q[:,2:]-B,axis=1))
        ray2=X+R.T@t
        angle=np.degrees(np.arctan2(np.linalg.norm(np.cross(X,ray2),axis=1),(X*ray2).sum(1)))
    good=np.isfinite(error)&(X[:,2]>0)&(q[:,2]>0)&(angle>1)&(error<=3)
    nonH=good&(transfer>H_THRESH)
    row.update(fit_cheirality=int(n),sampson_all_held_px=stats(sampson),H_max_symmetric_all_held_px=stats(transfer),
        H_held_supported=int((transfer<=H_THRESH).sum()),positive_depth_parallax_reprojection_held=int(good.sum()),
        non_H_held_supported=int(nonH.sum()),R_relative=R.tolist(),unit_t_direction=t.tolist(),
        held_all_used_no_residual_filter=True)
    row['admitted']=bool(n>=35 and good.sum()>=max(12,.60*h) and nonH.sum()>=12 and
                         min(cells(A[nonH]),cells(B[nonH]))>=3)
    row['reason']='pair_only_conditional_motion' if row['admitted'] else 'translation_not_well_supported_outside_H'
    return row


def run():
    started=time.monotonic();cv2.setNumThreads(2)
    p=read(BASE/'fit-plan.json');contacts=read(CONTACT);by={x['frame']:x for x in contacts['records']}
    digest=sha(VIDEO);assert digest==p['source_video_sha256']==contacts['video_sha256']
    rows=p['front_arc'];names=[r['frame'] for r in rows];indices=[r['source_frame_index'] for r in rows]
    assert len(rows)==24 and indices==sorted(set(indices)) and not p['rear_arc']
    assert all(by[r['frame']]['source_frame_index']==r['source_frame_index'] and by[r['frame']]['source_sha256']==r['source_sha256'] for r in rows)
    boxes=[by[f]['paths']['body']['crop_original_xyxy'] for f in names]
    band=[min(b[0] for b in boxes)-64,max(b[2] for b in boxes)+64]
    assert 0 < band[0] < band[1] < W
    OUT.mkdir(exist_ok=False)
    sources={str(q):sha(q) for q in (VIDEO,CONTACT,BASE/'fit-plan.json',Path(__file__))}
    plan=dict(updated=datetime.now(timezone.utc).isoformat(),source_sha256=sources,frames=names,indices=indices,
        fixed_K_integer_centers=K.tolist(),excluded_x_full_height=band,time_budget_seconds=T_LIMIT,
        feature_policy='Native RGB FFmpeg; verify each frozen source crop. RootSIFT 2400 cap, radius8px dedup, full person corridor+descriptor exclusion.',
        split_policy='Adjacency and skip-one matches only. Mutual Lowe .75; closed triangles; ambiguous whole families rejected; hash reserves 1/5 complete families before geometry.',
        pair_policy='Same FIT for E and H; normalized1.5px E,3px H,4000iters. >=50fit,12held,3heldcells. Fit cheirality35; >=60% held positive-depth+>1deg+<=3px; >=12 such non-H matches on >=3cells.',
        no_retry=True,no_fit_start=True,physical_camera_admitted=False)
    atomic(OUT/'plan.json',plan)
    xy=[];desc=[];raster=[];mask=np.ones((H,W),np.uint8)*255;mask[:,band[0]:band[1]]=0
    sift=cv2.SIFT_create(nfeatures=5000)
    for im,f,b in zip(decodes(indices),names,boxes):
        old=Path(by[f]['paths']['body']['path'].replace('/mnt/c/','C:/',1))
        assert sha(old)==by[f]['paths']['body']['sha256']
        x0,y0,x1,y1=b
        with Image.open(old) as image:recorded=np.asarray(image.convert('RGB'))
        d=np.abs(im[y0:y1,x0:x1].astype(np.int16)-recorded.astype(np.int16))
        mean=float(d.mean());p99=float(np.quantile(d,.99))
        assert mean<=1.1 and p99<=3,(f,mean,p99)
        raster.append(dict(frame=f,mean_RGB_abs=mean,p99_RGB_abs=p99,byte_equal=not bool(d.any())))
        a,b=features(im,mask,sift,band);xy.append(a);desc.append(b)
        assert time.monotonic()-started<T_LIMIT
    print('VIDEO22_BG_FEATURES',len(xy),sum(map(len,xy)),flush=True)
    matches={(i,j):mutual(desc[i],desc[j]) for i in range(len(xy)) for j in range(i+1,min(i+3,len(xy)))}
    families,rejected=tracks(matches,len(xy));results={};arrays={}
    for (i,j),m in matches.items():
        pairs=[(d[i],d[j],is_held) for d,is_held in families if i in d and j in d and m.get(d[i])==d[j]]
        a=np.asarray([xy[i][x] for x,_,_ in pairs]).reshape(-1,2)
        b=np.asarray([xy[j][y] for _,y,_ in pairs]).reshape(-1,2)
        h=np.asarray([v for _,_,v in pairs],bool)
        result=diagnose(a,b,h);result.update(pair=[names[i],names[j]],raw_mutual_matches=len(m));results[i,j]=result
        arrays[f'pair_{i:02d}_{j:02d}']=np.column_stack((a,b,h))
        assert time.monotonic()-started<T_LIMIT
    loops=[]
    for i in range(len(xy)-2):
        a,b,c=(results[k] for k in ((i,i+1),(i+1,i+2),(i,i+2)))
        if not all(r['admitted'] for r in (a,b,c)):continue
        R=np.asarray(c['R_relative']).T@np.asarray(b['R_relative'])@np.asarray(a['R_relative'])
        theta=float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1))))
        loops.append(dict(frames=names[i:i+3],closure_degrees=theta,pass_1deg=theta<=1))
    assert sources=={q:sha(q) for q in sources}
    np.savez_compressed(OUT/'matched_pixels.npz',**arrays)
    out=dict(updated=datetime.now(timezone.utc).isoformat(),state='fixed_K_pair_diagnostic_only',
        seconds=time.monotonic()-started,frames=names,feature_counts=list(map(len,xy)),raster_verification=raster,
        whole_track_families=len(families),held_families=sum(h for _,h in families),ambiguous_rejected=rejected,
        pair_count=len(results),supported_pairs=sum(r['admitted'] for r in results.values()),
        supported_adjacent_pairs=sum(r['admitted'] for (i,j),r in results.items() if j==i+1),
        pair_results=list(results.values()),rotation_loops=loops,plan_sha256=sha(OUT/'plan.json'),
        matched_pixels_sha256=sha(OUT/'matched_pixels.npz'),new_fitting_input_written=False,
        real_static_identity_not_independently_labeled=True,gravity_scale_K_calibration_accepted=False,
        no_camera_chain_or_interpolation=True,rear_frames=0,optimizer_updates=0,training_started=False,quality_accepted=False)
    atomic(OUT/'result.json',out)
    print('VIDEO22_BG_DIAGNOSTIC',out['supported_pairs'],out['supported_adjacent_pairs'],len(loops),round(out['seconds'],1),flush=True)


if __name__=='__main__':run()
