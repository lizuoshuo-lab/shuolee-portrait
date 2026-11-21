"""T6: extend frozen material observations through actual native adjacent frames.

One CPU run, no redetection/reacquisition, camera estimates, or training.  Optical
flow acceptance is a numerical consistency check, not physical point identity.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse, hashlib, json, signal, time, traceback
import cv2
import numpy as np

H=Path(__file__).resolve().parent
R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
I=R/'video-update-20261027'; S=R/'native-supervision'; O=R/'native-continuity'
G=I/'clip03-grid-coverage/array-cache-repair'
PARAMS=dict(winSize=(31,31),maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.001),
    flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,minEigThreshold=1e-4)
FB_LIMIT=.75
MARGIN=16

def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)

def seeds():
    bg=read(I/'clip03-roof-review/reviewed-support.json')['tracks']
    vi=read(G/'visual-index.json');rv=read(G/'visual-review.json')
    out=[]
    for t in bg:
        out.append(dict(t,group='static_scene_candidate',material=t['material']))
    for r in rv['rows']:
        if r['row']>=51:
            t=vi['rows'][r['row']]
            out.append(dict(t,group='person_diagnostic_only',material=r['reviewed_material']))
    if len(bg)!=55 or len(out)!=87: raise ValueError('Frozen seed selection changed')
    seen={}
    for t in out:
        key=str(t['family'])
        if key in seen: raise ValueError('Duplicate physical family in seed list')
        seen[key]=t['split']
        t['seed']=min(t['observations'],key=lambda x:(abs(x['frame']-232),x['frame']))
        t['seed_rule']='Existing observation nearest fixed middle frame232; earlier frame breaks ties, before flow or residuals.'
    return out

def register():
    if O.exists(): raise RuntimeError('T6 exists: inspect, never repeat')
    prior=read(S/'review.json')
    if not prior['actual_review_complete']: raise RuntimeError('T5 review missing')
    files=[Path(__file__),S/'review.json',S/'native-frames.json',S/'frame-plan.json',S/'frame-clock.json',
        S/'decode-receipt.json',I/'clip03-roof-review/reviewed-support.json',G/'visual-index.json',G/'visual-review.json',
        I/'clip03-scene-support-plan.json']
    st=seeds();frames=read(S/'native-frames.json')['frames']
    if [x['frame_id'] for x in frames]!=list(range(464)): raise ValueError('Full temporal plan mismatch')
    O.mkdir()
    save(O/'seeds.json',st);files.append(O/'seeds.json')
    save(O/'plan.json',dict(stage='T6',registered=now(),hypothesis=
        'Actual 24Hz observations may bridge the large16frame gaps and distinguish temporal tracking drift from camera-model residuals. This tests observation continuity, not a newK or static-body model.',
        changed_variable='One adjacent-frame native optical-flow propagation from each existing fixed material seed, across the newly frozen full464frames. The old SIFT anchors and whole-family splits remain unchanged and are withheld from propagation except the one selected seed.',
        source_sha256=read(S/'plan.json')['source']['sha256'],frame_count=464,
        seed_counts=dict(static_scene=55,scene_fit=40,scene_heldout=15,person_diagnostic=32),
        seed_rule='Nearest existing observation to232 per family, earlier tie. Propagate once both directions, stop on first numerical rejection; no re-detection, reseeding, snapping to anchors or gap interpolation.',
        operation='OpenCV pyramidal Lucas-Kanade on unchanged fullnative uint8 grayscale, as used by previous SIFT (cv2.imread COLOR then BGR2GRAY). Internal pyramids only; all output coordinates remain original1520x2704 integer-center convention.',
        parameters=dict(window=31,pyramid_max_level=3,max_iterations=40,epsilon=.001,min_eigenvalue=1e-4,
            forward_backward_limit_px=FB_LIMIT,frame_margin_px=MARGIN,opencv_threads=2),
        comparison='All nonseed existing SIFT anchors are comparison-only: report native Euclidean disagreement and availability, never use to correct flow or choose parameters. SIFT is not subpixel groundtruth. Report face, clothes, feet and static negatives separately.',
        physical_limits='Bidirectional numerical consistency can follow the wrong repeated texture. Propagated paths are correlated observations, not independent heldout measurements; original heldout-family status preserved. No background/body staticity inferred from track smoothness.',
        failure_policy='Every attempted point, forward/backward result, status, and firstfailure is retained. Unsupported slots remain missing. No deleting by camera residual or promoting rejected points.',
        review='Inspect all87 chronological native crop strips using deterministic seed/ends/firstfailure and existing anchors; review native context for face and drift/termination cases. No drawn labels/markers on images; mapping in JSON/captions.',
        budget=dict(runs=1,CPU_wall_seconds=900,GPU=0,new_cameras=0,GS_steps=0),
        stop='Any source/hash/identity/control failure or walllimit: preserve partial arrays and failure. Do not restart, relax thresholds, or claim failed tracking proves physical discontinuity.',
        input_hashes={str(p):sha(p) for p in files},K_changed=False,pose_read_or_changed=False,training_admitted=False))
    q=read(R/'queue-state.json');q.update(updated=now(),phase='T6_native_temporal_continuity_registered',running=None,
        next='Execute registered T6 once on CPU: unchanged material seeds/splits, actual all464 frames, no camera or training.');save(R/'queue-state.json',q)
    print('T6_REGISTERED',len(st),flush=True)

def inside(x,w,h):
    return np.isfinite(x).all(1)&(x[:,0]>=MARGIN)&(x[:,0]<w-MARGIN)&(x[:,1]>=MARGIN)&(x[:,1]<h-MARGIN)

def step(a,b,p):
    p=np.asarray(p,np.float32).reshape(-1,1,2)
    q,s,e=cv2.calcOpticalFlowPyrLK(a,b,p,None,**PARAMS)
    safe=q.copy();finite=np.isfinite(q).all((1,2));safe[~finite]=p[~finite]
    z,t,be=cv2.calcOpticalFlowPyrLK(b,a,safe,None,**PARAMS)
    q=q[:,0];z=z[:,0];fb=np.linalg.norm(z-p[:,0],axis=1)
    reason=np.zeros(len(p),np.uint8)
    reason|=(~s.ravel().astype(bool)).astype(np.uint8)
    reason|=(~t.ravel().astype(bool)).astype(np.uint8)*2
    reason|=(~inside(q,a.shape[1],a.shape[0])).astype(np.uint8)*4
    reason|=((~np.isfinite(fb))|(fb>FB_LIMIT)).astype(np.uint8)*8
    reason|=((~np.isfinite(e.ravel()))|(e.ravel()<1e-4)|(~np.isfinite(be.ravel()))|(be.ravel()<1e-4)).astype(np.uint8)*16
    return q,z,fb,e.ravel(),reason

def controls():
    rng=np.random.default_rng(1226)
    a=cv2.GaussianBlur(rng.integers(0,256,(240,320),np.uint8),(5,5),.8)
    pts=np.array([[x,y] for y in [50,100,150,190] for x in [50,100,150,200,250]],np.float32)
    shift=np.array([.5,-.75],np.float32)
    b=cv2.warpAffine(a,np.array([[1,0,shift[0]],[0,1,shift[1]]],np.float32),(320,240),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    q,z,fb,e,reason=step(a,b,pts);err=np.linalg.norm(q-pts-shift,axis=1)
    if reason.any() or float(err.max())>.08: raise AssertionError('Native xy/subpixel translation control')
    blank=np.full_like(a,128);_,_,_,_,bad=step(blank,blank,pts)
    if not np.all(bad!=0): raise AssertionError('Textureless negative control')
    return dict(native_xy_translation_max_px=float(err.max()),forward_backward_max_px=float(fb.max()),
        tested_points=len(pts),textureless_rejected=int(np.count_nonzero(bad)),physical_accuracy_claim=False)

def run():
    p=read(O/'plan.json')
    if (O/'progress.json').exists(): raise RuntimeError('T6 already started: inspect actual PID/log')
    for path,h in p['input_hashes'].items():
        if sha(path)!=h: raise ValueError('Frozen input changed: '+path)
    def expired(*_): raise TimeoutError('Registered T6 CPU walllimit')
    signal.signal(signal.SIGALRM,expired);signal.alarm(p['budget']['CPU_wall_seconds'])
    cv2.setNumThreads(2);start=time.monotonic();st=read(O/'seeds.json');fr=read(S/'native-frames.json')['frames'];n=len(st)
    uv=np.full((464,n,2),np.nan,np.float32);back=np.full_like(uv,np.nan)
    fb=np.full((464,n),np.nan,np.float32);eig=np.full_like(fb,np.nan)
    state=np.full((464,n),-1,np.int16);reason=np.zeros((464,n),np.uint8)
    def snapshot(label,**kw):
        save(O/'progress.json',dict(stage='T6',state=label,pid=os.getpid(),updated=now(),seconds=time.monotonic()-start,**kw))
    def arrays(): np.savez_compressed(O/'tracks.npz',uv=uv,backward_uv=back,forward_backward=fb,min_eigenvalue=eig,state=state,rejection_bits=reason)
    def image(f):
        path=Path(fr[f]['rgb']['path'])
        if sha(path)!=fr[f]['rgb']['sha256']: raise ValueError('Frozen native image changed at'+str(f))
        im=cv2.imread(str(path),cv2.IMREAD_COLOR)
        if im is None or im.shape!=(2704,1520,3): raise ValueError('Native raster differs')
        return cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    try:
        snapshot('controls');c=controls();save(O/'controls.json',c)
        for j,t in enumerate(st):uv[t['seed']['frame'],j]=t['seed']['uv'];state[t['seed']['frame'],j]=2
        lo=min(t['seed']['frame'] for t in st);hi=max(t['seed']['frame'] for t in st)
        for direction,seq in [('forward',range(lo,463)),('backward',range(hi,0,-1))]:
            current=image(seq.start);active=np.zeros(n,bool)
            for f in seq:
                active|=np.array([t['seed']['frame']==f for t in st])
                dest=f+(1 if direction=='forward' else -1);nxt=image(dest)
                ids=np.flatnonzero(active)
                if len(ids):
                    q,z,err,e,why=step(current,nxt,uv[f,ids]);uv[dest,ids]=q;back[dest,ids]=z
                    fb[dest,ids]=err;eig[dest,ids]=e;reason[dest,ids]=why
                    state[dest,ids]=np.where(why==0,1,0);active[ids[why!=0]]=False
                current=nxt
                if dest%32==0 or dest in [0,463]:
                    arrays();snapshot('propagating',direction=direction,frame=dest,active=int(active.sum()))
                    print('T6_FLOW',direction,dest,int(active.sum()),flush=True)
        arrays()
        comparisons=[]
        for j,t in enumerate(st):
            for a in t['observations']:
                f=a['frame'];seed=f==t['seed']['frame'];valid=state[f,j]>0
                comparisons.append(dict(track_index=j,track=t['track'],family=t['family'],split=t['split'],group=t['group'],material=t['material'],
                    frame=f,seed=seed,raw_sift_uv=a['uv'],flow_uv=uv[f,j].tolist() if np.isfinite(uv[f,j]).all() else None,
                    numerical_flow_valid=bool(valid),state=int(state[f,j]),
                    disagreement_px=float(np.linalg.norm(uv[f,j]-a['uv'])) if valid else None))
        save(O/'anchor-comparison.json',comparisons)
        coverage=[]
        for f in range(464):
            row=dict(frame=f,time_seconds=fr[f]['time_seconds'])
            for name,fun in [('scene_fit',lambda t:t['group']=='static_scene_candidate' and t['split']=='fit'),
                             ('scene_heldout',lambda t:t['group']=='static_scene_candidate' and t['split']=='heldout'),
                             ('nonground_fit',lambda t:t['group']=='static_scene_candidate' and t['split']=='fit' and t['nonground']),
                             ('face',lambda t:t['material']=='face_surface_coarse')]:
                ids=[j for j,t in enumerate(st) if fun(t)];row[name]=int(np.count_nonzero(state[f,ids]>0))
            coverage.append(row)
        save(O/'coverage.json',coverage)
        def summary(group):
            rows=[r for r in comparisons if not r['seed'] and group(r)]
            errs=[r['disagreement_px'] for r in rows if r['disagreement_px'] is not None]
            return dict(anchors=len(rows),available=len(errs),median_px=float(np.median(errs)) if errs else None,
                max_px=max(errs) if errs else None,over1px=sum(e>=1 for e in errs))
        result=dict(stage='T6',state='complete_pending_material_review',updated=now(),seconds=time.monotonic()-start,
            frames=464,seed_families=n,controls=c,attempted_slots=int(np.count_nonzero(state>=0)),numerically_valid_slots=int(np.count_nonzero(state>0)),
            static_scene_fit_min=min(r['scene_fit'] for r in coverage),frames_with_8_scene_fit=sum(r['scene_fit']>=8 for r in coverage),
            missing_scene_fit_frames=[r['frame'] for r in coverage if r['scene_fit']==0],
            disagreement={g:summary(fn) for g,fn in [('scene',lambda r:r['group']=='static_scene_candidate'),
                ('face',lambda r:r['material']=='face_surface_coarse'),('clothing',lambda r:r['material'] in ['white_clothing_in_head_roi','shorts']),
                ('feet',lambda r:r['material'] in ['shoe','sock'])]},
            state_codes={'-1':'not_observed_or_stopped','0':'first_rejected_attempt_retained','1':'numeric_flow_consistent','2':'original_seed'},
            rejection_bits={'1':'forward_solver_status','2':'backward_solver_status','4':'outside_full_patch_margin','8':'forward_backward_limit','16':'low_texture'},
            input_hashes_unchanged=all(sha(path)==h for path,h in p['input_hashes'].items()),
            K_changed=False,old_poses_or_xyz_read=False,cameras_registered=0,person_fit=False,training=False,physical_admission=False,
            limits=['Small forward/backward residual does not certify correct physical identity or subpixel localization.',
                'Nonseed anchors were not used to correct flow. Old SIFT anchors themselves have uncertainty.',
                'Flow trajectories are correlated proposals; whole-family holdout split stays frozen.',
                'Loss of texture/field-of-view or accumulated drift is not proof of physical video discontinuity.'])
        if not result['input_hashes_unchanged']:raise ValueError('Source changed during execution')
        save(O/'result.json',result);save(O/'progress.json',dict(state='complete_pending_material_review',pid=None,updated=now()))
        signal.alarm(0);print(json.dumps(result),flush=True)
    except BaseException as e:
        arrays();save(O/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc()))
        save(O/'progress.json',dict(state='failed_preserved',pid=None,updated=now()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run']);a=p.parse_args();globals()[a.action]()
