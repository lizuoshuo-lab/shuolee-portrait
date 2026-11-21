"""One fixed-K close-bark plus outer-scene conditional camera pair check."""
from collections import defaultdict
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from official_smplifyx_entry123 import sha
from prepare_leafwoman_bark142 import BOX, OUT, atomic, read
from probe_video22_background135 import BASE, K, I, cells, homogeneous, norm, apply_h, mutual, tracks, stats


def bark_sift(rgb, polygon, sift):
    height,width=rgb.shape[:2]
    mask=np.zeros((height,width),np.uint8)
    cv2.fillConvexPoly(mask,np.asarray(polygon,dtype=np.int32)-BOX[:2],255)
    dist=cv2.distanceTransform(mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    kp,desc=sift.detectAndCompute(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),mask)
    order=sorted(range(len(kp)),key=lambda j:(-kp[j].response,*kp[j].pt,kp[j].size))
    seen=defaultdict(list); kept=[]
    for j in order:
        x,y=kp[j].pt
        if dist[round(y),round(x)] <= 6*kp[j].size+1: continue
        a,b=int(x//8),int(y//8)
        if any((x-u)**2+(y-v)**2<64 for u0 in (-1,0,1) for v0 in (-1,0,1) for u,v in seen[a+u0,b+v0]):
            continue
        kept.append(j);seen[a,b].append((x,y))
    xy=np.asarray([kp[j].pt for j in kept],dtype=float).reshape(-1,2)+BOX[:2]
    d=desc[kept] if kept else np.empty((0,128),np.float32)
    return xy,np.sqrt(d/np.maximum(d.sum(1,keepdims=True),1e-12))


def bark_cells(xy, polygon):
    top,bottom=polygon[:2],polygon[[3,2]]
    y=(xy[:,1]-top[0,1])/(bottom[0,1]-top[0,1])
    lo,hi=(top[:,0]+y[:,None]*(bottom[:,0]-top[:,0])).T
    x=(xy[:,0]-lo)/(hi-lo)
    return len(set(zip(np.clip((x*2).astype(int),0,1),np.clip((y*3).astype(int),0,2))))


def assess(a,b,R,t,F,H):
    fa=homogeneous(a)@F.T;fb=homogeneous(b)@F
    sa=np.abs((homogeneous(b)*fa).sum(1))/np.maximum(np.sqrt((fa[:,:2]**2+fb[:,:2]**2).sum(1)),1e-15)
    h=np.maximum(np.linalg.norm(apply_h(H,a)-b,axis=1),np.linalg.norm(apply_h(np.linalg.inv(H),b)-a,axis=1))
    P=cv2.triangulatePoints(np.eye(3,4),np.c_[R,t],norm(a).T,norm(b).T).T
    with np.errstate(divide='ignore',invalid='ignore'):
        X=P[:,:3]/P[:,3:];Y=X@R.T+t
        err=np.maximum(np.linalg.norm((X@K.T)[:,:2]/X[:,2:]-a,axis=1),np.linalg.norm((Y@K.T)[:,:2]/Y[:,2:]-b,axis=1))
        cam2=X+R.T@t
        angle=np.degrees(np.arctan2(np.linalg.norm(np.cross(X,cam2),axis=1),(X*cam2).sum(1)))
    good=(X[:,2]>0)&(Y[:,2]>0)&np.isfinite(err)&(err<=3)
    parallax=good&(angle>1)
    return dict(sampson_px=stats(sa),symmetric_H_transfer_px=stats(h),reprojection_px=stats(err),
        positive_depth_reproject_count=int(good.sum()),good_parallax_count=int(parallax.sum()),
        non_H_good_parallax_count=int((parallax&(h>3)).sum()),H_support_count=int((h<=3).sum()),
        positive_depth_reprojection_ratio=float(good.mean()),no_held_residual_selection=True)


def pair(a,b,near,outer,polygons):
    tree=near.astype(bool); ext=outer[:,4].astype(bool)
    A,B=outer[:,:2],outer[:,2:4]
    counts=dict(bark_fit=int((~tree).sum()),bark_held=int(tree.sum()),outer_fit=int((~ext).sum()),outer_held=int(ext.sum()))
    coverage=dict(bark_held_cells=[bark_cells(a[tree],polygons[0]),bark_cells(b[tree],polygons[1])],
                  outer_held_cells=[cells(A[ext]),cells(B[ext])])
    row=dict(**counts,**coverage,admitted=False)
    if (counts['bark_fit']<12 or counts['bark_held']<6 or counts['outer_fit']<24 or counts['outer_held']<8
        or min(coverage['bark_held_cells'])<3 or min(coverage['outer_held_cells'])<3):
        return dict(row,reason='support_or_spread_insufficient')
    x=np.concatenate([a[~tree],A[~ext]]);y=np.concatenate([b[~tree],B[~ext]])
    p,q=norm(x),norm(y)
    cv2.setRNGSeed(142)
    E,mask=cv2.findEssentialMat(p,q,np.eye(3),method=cv2.RANSAC,prob=.999,
        threshold=1.5/np.mean([K[0,0],K[1,1]]),maxIters=4000)
    H,_=cv2.findHomography(x,y,cv2.RANSAC,3.,maxIters=4000,confidence=.999)
    if E is None or E.shape!=(3,3) or H is None or abs(np.linalg.det(H))<1e-12:
        return dict(row,reason='no_unique_pose_or_homography')
    count,R,t,pose_mask=cv2.recoverPose(E,p,q,np.eye(3),mask=mask.copy());t=t[:,0]
    tx,ty,tz=t
    cross=np.array([[0,-tz,ty],[tz,0,-tx],[-ty,tx,0]])
    F=I.T@cross@R@I
    nr,fr=assess(a[tree],b[tree],R,t,F,H),assess(A[ext],B[ext],R,t,F,H)
    good_near=nr['positive_depth_reprojection_ratio']>=.6 and nr['good_parallax_count']>=max(4,.6*tree.sum())
    good_outer=fr['positive_depth_reprojection_ratio']>=.6
    good_err=max(nr['sampson_px']['p90'],fr['sampson_px']['p90'])<=3.
    nonplanar=nr['non_H_good_parallax_count']+fr['non_H_good_parallax_count']>=6
    use=bool(count>=.6*len(x) and good_near and good_outer and good_err and nonplanar)
    return dict(row,admitted=use,reason='conditional_pair' if use else 'pose_or_nonplanar_validation_failed',
        bark_validation=nr,outer_validation=fr,fit_cheirality_count=int(count),R_relative=R.tolist(),unit_t_direction=t.tolist(),
        fit_bark_positive_count=int(np.count_nonzero(pose_mask[:counts['bark_fit']])),
        fit_outer_positive_count=int(np.count_nonzero(pose_mask[counts['bark_fit']:])))


def run():
    start=time.monotonic();cv2.setNumThreads(2)
    assert not (OUT/'plan.json').exists()
    source=read(OUT/'source.json');rois=read(OUT/'bark-rois.json')
    assert rois['source_sha256']==sha(OUT/'source.json') and source['strip_sha256']==sha(OUT/'native-strips.npz')
    assert rois['review_image_sha256']==sha(OUT/'bark-roi-review.png')
    bg=BASE/'background-motion135'; previous=read(bg/'result.json'); split=read(bg/'split-verification.json')
    assert split['no_same_keypoint_in_two_splits'] and split['nearest_fit_to_held_native_px']>=8
    assert source['old_background_plan_sha256']==sha(bg/'plan.json')==previous['plan_sha256']
    assert previous['matched_pixels_sha256']==sha(bg/'matched_pixels.npz')
    polygons=np.asarray([r['native_polygon'] for r in rois['bounds']])
    with np.load(OUT/'native-strips.npz',allow_pickle=False) as z:
        frames=z['frame_ids'].tolist(); rgb=z['rgb']
    assert frames==[r['frame'] for r in rois['bounds']]==previous['frames'] and len(frames)==24
    assert polygons[:,:,0].min()>BOX[0] and polygons[:,:,0].max()<BOX[2]
    policy=dict(frames=frames,source_sha256=sha(OUT/'source.json'),rois_sha256=sha(OUT/'bark-rois.json'),
        runner_sha256=sha(__file__),outside_pixels_sha256=sha(bg/'matched_pixels.npz'),outside_split_sha256=sha(bg/'split-verification.json'),
        fixed_K=K.tolist(),source_only_roi_actually_reviewed=True,
        extractor='SIFT default, descriptor footprint within static bark polygon (6*size+1), 8px dedup, mutual 0.75.',
        track_split='Reuse 135 triangle-cycle clean whole-family graph, stable hash 1/5 held. Use archived outer splits unchanged. No cross-domain pixel overlap.',
        test='One combined E/H candidate per original adjacent/skip-one pair. Bark FIT≥12, HELD≥6, outer FIT≥24 HELD≥8; both held scene groups span ≥3 predeclared cells. '
             'RANSAC1.5px E and 3px H, both groups ≥60% positive with ≤3px reprojection, ≥60% bark support parallax>1deg, '
             '≥6 non-H held with positive depth and parallax, held Sampson P90≤3px. Do not refit from holdout, sweep, or interpolate.',
        limits='Bark is one nearby static structure, not a measured depth plane. Outer-scene points reused, not new blinded measurements. '
               'No full camera path without independent connected support for all24 and target-excluded multiview checks.')
    atomic(OUT/'plan.json',policy)
    extractor=cv2.SIFT_create();xy,desc=zip(*(bark_sift(image,poly,extractor) for image,poly in zip(rgb,polygons,strict=True)))
    edges={(i,j):mutual(desc[i],desc[j]) for i in range(24) for j in range(i+1,min(i+3,24))}
    families,ambiguous=tracks(edges,24)
    near=np.inf
    for i,points in enumerate(xy):
        x=np.asarray([points[d[i]] for d,h in families if i in d and not h]);y=np.asarray([points[d[i]] for d,h in families if i in d and h]);
        if len(x) and len(y): near=min(near,float(np.linalg.norm(x[:,None]-y[None],axis=2).min()))
    assert near>=8
    rows=[]; arrays={}; status={}
    with np.load(bg/'matched_pixels.npz',allow_pickle=False) as existing:
        assert set(existing.files)=={f'pair_{i:02d}_{j:02d}' for i,j in edges}
        for (i,j),m in edges.items():
            key=f'pair_{i:02d}_{j:02d}'
            old=existing[key]
            assert np.all(((old[:,0]<BOX[0])|(old[:,0]>BOX[2]))&((old[:,2]<BOX[0])|(old[:,2]>BOX[2])))
            accepted=[(d[i],d[j],h) for d,h in families if i in d and j in d and m.get(d[i])==d[j]]
            a=np.asarray([xy[i][u] for u,v,h in accepted]).reshape(-1,2)
            b=np.asarray([xy[j][v] for u,v,h in accepted]).reshape(-1,2)
            held=np.asarray([h for u,v,h in accepted],bool)
            row=pair(a,b,held,old,polygons[[i,j]])
            row.update(frames=[frames[i],frames[j]],mutual_count=len(m)); rows.append(row)
            arrays[key]=np.column_stack([a,b,held]);status[i,j]=row
    assert time.monotonic()-start<120
    graph={i:set() for i in range(24)}
    for (i,j),row in status.items():
        if row['admitted']: graph[i].add(j);graph[j].add(i)
    components=[];remaining=set(graph)
    while remaining:
        todo=[min(remaining)]; group=set()
        while todo:
            i=todo.pop()
            if i in group: continue
            group.add(i);todo.extend(sorted(graph[i]-group))
        remaining-=group;components.append([frames[i] for i in sorted(group)])
    np.savez_compressed(OUT/'bark-matched-pixels.npz',**arrays)
    receipt=dict(plan_sha256=sha(OUT/'plan.json'),seconds=time.monotonic()-start,points=[len(p) for p in xy],
        frames=frames,raw_source_crop_matches=source['raw_body_crop_matches'],pairs=rows,
        conditioned_pairs=sum(r['admitted'] for r in rows),actual_pose_estimates=sum('bark_validation' in r for r in rows),
        whole_bark_families=len(families),bark_held_families=sum(h for _,h in families),ambiguities=ambiguous,
        min_same_frame_fit_to_held_distance_px=near if np.isfinite(near) else None,components=components,
        pixels_sha256=sha(OUT/'bark-matched-pixels.npz'),camera_path_written=False,training_admitted=False,quality_accepted=False)
    atomic(OUT/'result.json',receipt)
    print('BARK142',receipt['points'],len(families),receipt['actual_pose_estimates'],receipt['conditioned_pairs'],list(map(len,components)),flush=True)


if __name__=='__main__':
    run()
