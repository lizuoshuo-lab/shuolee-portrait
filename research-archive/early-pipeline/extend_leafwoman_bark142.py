"""Extend the registered 022 bark families through dense native video frames."""
from collections import defaultdict
from pathlib import Path
import time

import cv2
import numpy as np

from official_smplifyx_entry123 import sha
from prepare_leafwoman_bark142 import BOX, OUT, atomic, read
from probe_leafwoman_bark142 import BASE, pair
from track_leafwoman_torso141 import decode
from track_native_continuity122 import controls, step


def families(z):
    parent={};role={}

    def root(n):
        parent.setdefault(n,n)
        if parent[n]!=n: parent[n]=root(parent[n])
        return parent[n]

    for key in sorted(z.files):
        i,j=map(int,key.split('_')[1:])
        for a,b,c,d,h in z[key]:
            p,q=(i,a,b),(j,c,d)
            for x in (p,q):
                assert x not in role or role[x]==bool(h)
                role[x]=bool(h)
            parent[root(q)]=root(p)
    groups=defaultdict(list)
    for k in role: groups[root(k)].append(k)
    result=[]
    for group in sorted(sorted(g) for g in groups.values()):
        assert len({q[0] for q in group})==len(group)>=3 and len({role[q] for q in group})==1
        result.append((group,role[group[0]]))
    return result


def run():
    cv2.setNumThreads(2); started=time.monotonic()
    init=OUT/'dense-extension-bounded-plan.json';done=OUT/'dense-extension-result.json'
    assert not init.exists() and not done.exists()
    first=read(OUT/'result.json');source=read(OUT/'source.json')
    assert first['pixels_sha256']==sha(OUT/'bark-matched-pixels.npz') and first['actual_pose_estimates']==0
    assert sha(OUT/'plan.json')==first['plan_sha256']
    with np.load(OUT/'bark-matched-pixels.npz',allow_pickle=False) as f: material=families(f);keys=f.files.copy()
    assert len(material)==first['whole_bark_families'] and sum(h for _,h in material)==first['bark_held_families']
    frames=first['frames']; indices=np.asarray([x['native_index'] for x in source['frames']]);stop=int(indices[-1])+1
    assert len(indices)==24 and source['raw_body_crop_matches']==24
    roi=read(OUT/'bark-rois.json');poly=np.asarray([r['native_polygon'] for r in roi['bounds']],dtype=np.float32)
    assert [r['frame'] for r in roi['bounds']]==frames
    dense_poly=np.column_stack([np.interp(np.arange(stop),indices,v) for v in poly.reshape(24,8).T]).reshape(stop,4,2).astype(np.float32)
    seed=np.asarray([indices[g[len(g)//2][0]] for g,_ in material]); seed_xy=np.asarray([g[len(g)//2][1:] for g,_ in material])
    starts=np.asarray([indices[max(0,g[0][0]-2)] for g,_ in material])
    stops=np.asarray([indices[min(23,g[-1][0]+2)] for g,_ in material])
    from track_leafwoman_torso141 import OUT as TORSO
    old=read(TORSO/'plan.json'); skip=set(old['skipped_source_indices']); assert not skip&set(indices)
    atomic(init,dict(first_result_sha256=sha(OUT/'result.json'),old_track_pixels_sha256=first['pixels_sha256'],
        initial_reverse_failure_sha256=sha(OUT/'dense-extension-attempt-1.json'),
        decode_repair='One forward RGB crop decode; bounded gray crop in-memory cache reused for reverse. No ffmpeg reverse buffer.',
        roi_sha256=sha(OUT/'bark-rois.json'),runner_sha256=sha(__file__),reuses_whole_family_roles=True,
        policy='At each family median original view, follow existing native LK through dense frames both ways; '
               'stop at first failure, polygon margin16px. Track only up to two roster slots outside original support. '
               'At original support require all saved anchors within2px of LK. No exact-eight reserve pixels decoded to tracking. '
               'Discard both entire families on any <8px same-frame collision; no role reassignment. Same final pair gate as round142.',
        no_crop_resize=True,integer_origin=list(BOX[:2]),skipped_source_indices=sorted(skip),original_family_count=len(material)))
    ctrl=controls()
    n=len(material);uv=np.full((stop,n,2),np.nan,np.float32);valid=np.zeros((stop,n),bool)
    uv[seed,np.arange(n)]=seed_xy;valid[seed,np.arange(n)]=True
    origin=np.asarray(BOX[:2]);cache=np.load(OUT/'native-strips.npz',allow_pickle=False)
    strips=cache['rgb']; cache.close();roster={n:i for i,n in enumerate(indices)}
    gray_frames=np.empty((stop,BOX[3]-BOX[1],BOX[2]-BOX[0]),np.uint8)
    assert gray_frames.nbytes < 72*1024**2
    selected=[k for k in range(stop) if k not in skip]
    for reverse in (False,True):
        seq=selected[::-1] if reverse else selected
        stream=((i,None) for i in seq) if reverse else decode(seq,BOX,False)
        active=np.zeros(n,bool);previous=None
        for i,rgb in stream:
            assert time.monotonic()-started<120
            if not reverse:
                if i in roster:assert np.array_equal(rgb,strips[roster[i]]),(i,'dense RGB strip changed')
                gray_frames[i]=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
            gray=gray_frames[i]
            active &= (i>=starts)&(i<=stops)
            g=np.flatnonzero(active)
            if len(g) and previous is not None:
                q,_,_,_,why=step(previous,gray,uv[prev,g]-origin)
                xy=q+origin
                pass_mask=np.asarray([cv2.pointPolygonTest(dense_poly[i],tuple(map(float,v)),True)>=16 if np.isfinite(v).all() else False for v in xy])&(why==0)
                uv[i,g[pass_mask]]=xy[pass_mask]; valid[i,g[pass_mask]]=True;active[g[~pass_mask]]=False
            new=np.flatnonzero(seed==i)
            for j in new:
                active[j]=cv2.pointPolygonTest(dense_poly[i],tuple(map(float,seed_xy[j])),True)>=16
                valid[i,j]=active[j]
            previous,prev=gray,i
        print('BARK142_DENSE_DIRECTION',reverse,len(seq),flush=True)
    assert not valid[list(skip)].any()
    # Check the anchors of a whole family, including both sides of the seed.
    keep=np.ones(n,bool)
    for j,(g,_) in enumerate(material):
        slots=np.asarray([indices[p[0]] for p in g]);target=np.asarray([p[1:] for p in g])
        keep[j]=valid[slots,j].all() and np.max(np.linalg.norm(uv[slots,j]-target,axis=1))<=2
    consistent=int(keep.sum()); colliding=set()
    for i in range(stop):
        g=np.flatnonzero(valid[i]&keep)
        q=uv[i,g]
        distances=np.linalg.norm(q[:,None]-q[None,:],axis=2)
        row,col=np.nonzero(np.triu(distances<8,1));colliding.update(g[row]);colliding.update(g[col])
    keep[list(colliding)]=False
    held=np.asarray([h for _,h in material],bool)
    rows=[]
    oldbg=BASE/'background-motion135/matched_pixels.npz'
    assert sha(oldbg)==read(OUT/'plan.json')['outside_pixels_sha256']
    with np.load(oldbg,allow_pickle=False) as old:
        for k in sorted(keys):
            i,j=map(int,k.split('_')[1:]);x,y=indices[[i,j]]
            mask=keep&valid[x]&valid[y]
            row=pair(uv[x,mask].astype(float),uv[y,mask].astype(float),held[mask],old[k],poly[[i,j]])
            rows.append(dict(row,frames=[frames[i],frames[j]]))
    np.savez_compressed(OUT/'dense-family-pixels.npz',uv=uv[indices],valid=valid[indices],retained=keep,held=held,frame_ids=np.asarray(frames))
    result=dict(plan_sha256=sha(init),family_pixels_sha256=sha(OUT/'dense-family-pixels.npz'),seconds=time.monotonic()-started,
        control=ctrl,gray_crop_cache_bytes=gray_frames.nbytes,retained_original_anchor_consistent_families=consistent,colliding_families_excluded=len(colliding),
        independent_families_retained=int(keep.sum()),retained_held=int((keep&held).sum()),pairs=rows,
        actual_pose_estimates=sum('bark_validation' in r for r in rows),admitted_pairs=sum(r['admitted'] for r in rows),
        camera_path_written=False,training_started=False,quality_accepted=False)
    atomic(done,result)
    print('BARK142_DENSE',consistent,len(colliding),int(keep.sum()),result['actual_pose_estimates'],result['admitted_pairs'],flush=True)


if __name__=='__main__':
    run()
