"""N1: one native temporal material-observation pass, without camera/3D fitting."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, os, subprocess, time, traceback
import cv2
import numpy as np
from PIL import Image

R = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
I = R / 'video-update-20261027'
PLAN = I / 'native-temporal-material-plan.json'
O = I / 'native-temporal-material'

def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(path.read_text())
def save(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()
def digest(data): return hashlib.sha256(data).hexdigest()

def register():
    if PLAN.exists() or O.exists(): raise RuntimeError('N1 exists; inspect instead of repeating')
    video = read(I / 'result.json')['videos'][1]
    assert video['sha256'].startswith('32be3063')
    save(PLAN, dict(stage='N1', registered=now(), source_sha256=sha(Path(__file__)),
        input=video, anchor_frame=278, decoded_frames=557, width=1520, height=2704,
        hypothesis='New native temporal continuity may preserve material identity better than independent wide-baseline SIFT. One anchor, every real decoded frame, two temporal directions; do not infer physical truth from a face bounding box.',
        causal_scope='New encoding and matching method both changed. This diagnoses correspondence continuity; cannot attribute improvement solely to resolution or K.',
        camera_use='NONE: no K, distortion, extrinsics, old poses or 3D points are loaded or estimated. Existing K candidates unchanged.',
        domain='Already rectified/cropped native MP4. No autorotation, geometric resize, second undistortion or training-image warp. LK pyramid/interpolation is measurement internals only; outputs native integer-center coordinates.',
        face_seeds=[
            dict(name='image_left_eye_outer_candidate', uv=[714,1167]),
            dict(name='image_left_eye_inner_candidate', uv=[730,1164]),
            dict(name='image_right_eye_inner_candidate', uv=[753,1164]),
            dict(name='image_right_eye_outer_candidate', uv=[771,1164]),
            dict(name='nose_left_detail_candidate', uv=[730,1194]),
            dict(name='nose_base_detail_candidate', uv=[745,1198]),
            dict(name='mouth_image_left_candidate', uv=[727,1213]),
            dict(name='mouth_image_right_candidate', uv=[758,1211]),
            dict(name='lower_lip_candidate', uv=[743,1221])],
        seed_review='Approximate candidate pixels selected from actual native frame278 face crop [680,1110,865,1280]. Names are appearance hypotheses; eyelid/lip motion, highlight and aperture bias not ruled out. No manual point is ground truth.',
        regions=[
            dict(name='ground_candidate', rect=[32,2250,1488,2672], max_corners=240, min_distance=28, purpose='camera_fit_or_holdout_candidate_only'),
            dict(name='clothing_candidate', rect=[720,1325,845,1590], max_corners=50, min_distance=18, purpose='human_validation_only'),
            dict(name='footwear_candidate', rect=[680,2010,880,2190], max_corners=20, min_distance=18, purpose='human_validation_only')],
        seed_detector=dict(method='GFTT', quality_level=.02, block_size=7, integer_roi_offset=True),
        tracking=dict(method='OpenCV pyramidal LK', window=[21,21], max_level=3, iterations=30, epsilon=.01,
            min_eigen_threshold=.0001, adjacent_forward_backward_limit_px=.5, image_margin=12,
            stop_track_on_first_failed_validity_check=True, reseeding=False,
            direct_12_frame_path_difference='Recorded for all available IDs, never used to delete observations or tune thresholds'),
        holdout='All human tracks held out from camera fitting. Background seeds within48px joined into spatial families before tracking; sha256(sourceSHA|family_members)%5==0 held out as a whole family. No split changes after tracking.',
        review_frames=[0,70,139,208,254,266,278,290,302,348,417,486,556],
        preregistered_future_training_frames=list(range(0,557,4)),
        budget=dict(attempts=1, CPU_wall_seconds=600, opencv_threads=4, GPU=0, decoded_gray_memory_GB=2.4),
        stop_conditions=['Input hash/dimensions/frame count mismatch; no inference allowed.',
            'All current tracks lost in a direction: stop that direction without reseeding.',
            'Retain every attempted UV/status/error including rejected observations; do not cherry-pick reprojection errors.',
            'Insufficient physical semantic support or only near-planar ground: no camera/training admission.'],
        limitations=['Cycle consistency is not absolute localization accuracy or physical identity.',
            'One anchor cannot see newly unoccluded surfaces; lost/occluded observations remain missing.',
            'Ground candidates do not establish staticness or nonplanar depth.',
            'Measurement uncertainty remains empirical, not a calibrated subpixel ground truth.']))
    print('N1_NATIVE_TEMPORAL_REGISTERED', flush=True)

def lk(a, b, points, cfg):
    q, st, error = cv2.calcOpticalFlowPyrLK(a, b, points.reshape(-1,1,2).astype(np.float32), None,
        winSize=tuple(cfg['window']), maxLevel=cfg['max_level'],
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, cfg['iterations'], cfg['epsilon']),
        minEigThreshold=cfg['min_eigen_threshold'])
    return q.reshape(-1,2), st.ravel().astype(bool), error.ravel()

def synthetic_control(cfg):
    rng = np.random.default_rng(122)
    a = cv2.GaussianBlur(rng.integers(0,256,(128,128),dtype=np.uint8),(5,5),1.)
    delta = np.array([.35,-.2],np.float32)
    b = cv2.warpAffine(a,np.array([[1,0,delta[0]],[0,1,delta[1]]],np.float32),(128,128))
    p = np.array([[40,40],[88,40],[40,88],[88,88]],np.float32)
    q, s, _ = lk(a,b,p,cfg); back, sb, _ = lk(b,a,q,cfg)
    err = np.linalg.norm(q-p-delta,axis=1); fb=np.linalg.norm(back-p,axis=1)
    assert s.all() and sb.all() and err.max()<.1 and fb.max()<.1
    return dict(known_shift=delta.tolist(),max_translation_error_px=float(err.max()),max_fb_px=float(fb.max()),
        source='synthetic random texture only',physical_calibration=False)

def seed_points(gray, plan):
    seeds = [dict(id=i,group='face_candidate',name=s['name'],uv=s['uv'],split='heldout_human',family='face_all') for i,s in enumerate(plan['face_seeds'])]
    for region in plan['regions']:
        x0,y0,x1,y1=region['rect']; roi=gray[y0:y1,x0:x1]
        p=cv2.goodFeaturesToTrack(roi,maxCorners=region['max_corners'],qualityLevel=.02,minDistance=region['min_distance'],blockSize=7)
        if p is None: continue
        for xy in p.reshape(-1,2):
            uv=xy+np.array([x0,y0]);i=len(seeds)
            seeds.append(dict(id=i,group=region['name'],name=f'{region["name"]}_{i:04d}',uv=uv.tolist(),split='heldout_human',family=f'human_{region["name"]}'))
    ground=[s for s in seeds if s['group']=='ground_candidate']
    parent={s['id']:s['id'] for s in ground}
    def find(i):
        while parent[i]!=i: i=parent[i]
        return i
    for ia,a in enumerate(ground):
        for b in ground[ia+1:]:
            if np.linalg.norm(np.array(a['uv'])-b['uv'])<48:
                p,q=find(a['id']),find(b['id']);parent[max(p,q)]=min(p,q)
    families={}
    for s in ground: families.setdefault(find(s['id']),[]).append(s['id'])
    for members in families.values():
        h=digest((plan['input']['sha256']+'|'+','.join(map(str,sorted(members)))).encode())
        split='heldout_background' if int(h[:8],16)%5==0 else 'fit_background'
        for i in members: seeds[i].update(split=split,family=h)
    return seeds

def patch(image,uv,size=96):
    # Integer crop is presentation only, no resampling of image or recorded UV.
    x,y=np.rint(uv).astype(int);x0=x-size//2;y0=y-size//2
    out=np.full((size,size,3),225,np.uint8)
    xa,ya=max(0,x0),max(0,y0);xb,yb=min(image.shape[1],x0+size),min(image.shape[0],y0+size)
    if xb>xa and yb>ya: out[ya-y0:yb-y0,xa-x0:xb-x0]=image[ya:yb,xa:xb]
    return out

def run():
    plan=read(PLAN); assert plan['source_sha256']==sha(Path(__file__))
    if O.exists(): raise RuntimeError('N1 already has output; inspect instead of rerun')
    O.mkdir(); start=time.monotonic(); frames=None; proc=None
    def progress(stage,**extra):
        payload=dict(updated=now(),pid=os.getpid(),stage=stage,elapsed_seconds=time.monotonic()-start)
        payload.update(extra)
        save(O/'progress.json',payload)
    def budget():
        if time.monotonic()-start>plan['budget']['CPU_wall_seconds']: raise TimeoutError('N1 fixed budget exceeded')
    try:
        cv2.setNumThreads(plan['budget']['opencv_threads'])
        save(O/'synthetic-control.json',synthetic_control(plan['tracking']))
        source=Path(plan['input']['path']); assert sha(source)==plan['input']['sha256']
        n,h,w=plan['decoded_frames'],plan['height'],plan['width'];anchor=plan['anchor_frame']
        frames=np.empty((n,h,w),np.uint8); rgbs={}; gray_hashes=[]
        ff=['/usr/bin/ffmpeg','-nostdin','-hide_banner','-loglevel','error','-threads','4','-noautorotate','-i',str(source),'-map','0:v:0','-vsync','0','-pix_fmt','rgb24','-f','rawvideo','pipe:1']
        save(O/'decode-command.json',dict(command=ff,geometric_resampling=False,colors='ffmpeg rgb24 -> OpenCV RGB2GRAY, photometric conversion only'))
        with (O/'decode.log').open('wb') as log:
            proc=subprocess.Popen(ff,stdout=subprocess.PIPE,stderr=log)
            for f in range(n):
                buf=bytearray()
                while len(buf)<h*w*3:
                    block=proc.stdout.read(h*w*3-len(buf))
                    if not block: raise RuntimeError(f'Short native decode at {f}')
                    buf.extend(block)
                rgb=np.frombuffer(buf,np.uint8).reshape(h,w,3)
                frames[f]=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);gray_hashes.append(digest(frames[f].tobytes()))
                if f in plan['review_frames']:
                    rgbs[f]=rgb.copy();Image.fromarray(rgb).save(O/f'frame{f:06d}.png')
                if f%50==0:progress('decode',frame=f,ffmpeg_pid=proc.pid);print('decode',f,flush=True)
                budget()
            assert not proc.stdout.read(1), 'More decoded frames than manifest'
            assert proc.wait(timeout=20)==0
        seeds=seed_points(frames[anchor],plan);save(O/'seeds.json',dict(frame=anchor,seeds=seeds,physical_identity_accepted=False))
        count=len(seeds);uv=np.full((n,count,2),np.nan,np.float32);valid=np.zeros((n,count),bool)
        fb=np.full((n,count),np.nan,np.float32);photo=np.full((n,count),np.nan,np.float32)
        flags=np.zeros((n,count),np.uint8);uv[anchor]=np.array([s['uv'] for s in seeds]);valid[anchor]=True;flags[anchor]=15
        comparisons=[];cfg=plan['tracking'];margin=cfg['image_margin']
        for direction in [-1,1]:
            for f in range(anchor+direction,-1 if direction<0 else n,direction):
                prev=f-direction;ids=np.flatnonzero(valid[prev])
                if not len(ids): break
                p=uv[prev,ids];q,s,err=lk(frames[prev],frames[f],p,cfg)
                back,sb,_=lk(frames[f],frames[prev],q,cfg)
                errfb=np.linalg.norm(back-p,axis=1);inside=(q[:,0]>=margin)&(q[:,0]<w-margin)&(q[:,1]>=margin)&(q[:,1]<h-margin)&np.isfinite(q).all(1)
                ok=s&sb&inside&(errfb<cfg['adjacent_forward_backward_limit_px'])
                uv[f,ids]=q;fb[f,ids]=errfb;photo[f,ids]=err
                flags[f,ids]=s.astype(np.uint8)+2*sb.astype(np.uint8)+4*inside.astype(np.uint8)+8*(errfb<.5).astype(np.uint8)
                valid[f,ids]=ok
                if abs(f-anchor)%12==0:
                    ref=f-12*direction;common=np.flatnonzero(valid[f]&valid[ref])
                    if len(common):
                        direct,ds,_=lk(frames[ref],frames[f],uv[ref,common],cfg)
                        for j,track in enumerate(common):
                            comparisons.append(dict(frame=f,reference_frame=ref,track=int(track),direct_status=bool(ds[j]),
                                direct_uv=direct[j].tolist(),sequential_uv=uv[f,track].tolist(),difference_px=float(np.linalg.norm(direct[j]-uv[f,track]))))
                if abs(f-anchor)%30==0:progress('track',direction=direction,frame=f,alive=int(ok.sum()));print('track',direction,f,int(ok.sum()),flush=True)
                budget()
        np.savez_compressed(O/'observations.npz',uv=uv,valid=valid,forward_backward_px=fb,photometric_l1=photo,status_bits=flags)
        save(O/'path-comparison.json',dict(selection='Every available track at preregistered12-frame checkpoints; not a filter',records=comparisons))
        save(O/'frame-manifest.json',dict(source_sha256=plan['input']['sha256'],frames=[dict(frame=f,gray_sha256=x,preregistered_training_frame=f in plan['preregistered_future_training_frames']) for f,x in enumerate(gray_hashes)],geometry_preserved=True))
        review=[]
        # All human tracks; a deterministic pre-error sample of background tracks.
        review_ids=[s['id'] for s in seeds if s['group']=='face_candidate']
        review_ids += [s['id'] for group in ['ground_candidate','clothing_candidate','footwear_candidate'] for s in [x for x in seeds if x['group']==group][:6]]
        for track in review_ids:
            picks=[f for f in plan['review_frames'] if np.isfinite(uv[f,track]).all()]
            if not picks: continue
            canvas=np.full((96,96*len(picks),3),225,np.uint8)
            for k,f in enumerate(picks):canvas[:,96*k:96*(k+1)]=patch(rgbs[f],uv[f,track])
            fname=f'patch_{track:04d}_{seeds[track]["group"]}.png';Image.fromarray(canvas).save(O/fname)
            review.append(dict(track=track,file=fname,frames=picks,uv=[uv[f,track].tolist() for f in picks],valid=[bool(valid[f,track]) for f in picks],padding='neutral only outside source bounds',native_patch_size=96))
        save(O/'patch-index.json',dict(rows=review,selection='All9 face seeds and first6 each ground/clothing/footwear in detector order, independent of residuals; includes failed last observations when on review frame.',source_pixels_rescaled=False,text_on_images=False))
        summaries=[]
        for group in sorted(set(s['group'] for s in seeds)):
            ids=np.array([s['id'] for s in seeds if s['group']==group]);traces=[]
            for track in ids:
                hits=np.flatnonzero(valid[:,track]);traces.append(dict(track=int(track),first=int(hits.min()),last=int(hits.max()),valid_frames=int(len(hits)),split=seeds[track]['split']))
            vals=np.array([c['difference_px'] for c in comparisons if c['track'] in ids and c['direct_status']])
            summaries.append(dict(group=group,seeds=len(ids),valid_observations=int(valid[:,ids].sum()),
                reached_first_frame=int(valid[0,ids].sum()),reached_last_frame=int(valid[-1,ids].sum()),
                adjacent_failed_attempts=int((np.isfinite(fb[:,ids])&~valid[:,ids]).sum()),
                direct_path_checks=int(len(vals)),direct_path_median_px=float(np.median(vals)) if len(vals) else None,
                direct_path_max_px=float(vals.max()) if len(vals) else None,direct_path_ge_1px=int((vals>=1).sum()),tracks=traces))
        assert sha(source)==plan['input']['sha256']
        result=dict(stage='N1',updated=now(),state='completed_pending_actual_material_review',source_sha256=plan['input']['sha256'],
            decoded_frames=n,anchor=anchor,total_seeds=count,groups=summaries,elapsed_seconds=time.monotonic()-start,
            K_or_poses_estimated=False,old_3D_loaded=False,training_started=False,physical_face_accepted=False,
            uncertainty='Adjacent and12-frame consistency are self-consistency diagnostics, not independent absolute reprojection/physical material truth. No measurement-bias bound established.',
            limitation='Single-anchor visibility does not cover unseen surfaces; ground depth/staticness not certified.')
        save(O/'result.json',result);progress('completed_pending_actual_material_review',pid=None)
        print(json.dumps({k:v for k,v in result.items() if k!='groups'}),flush=True)
    except BaseException as e:
        if proc and proc.poll() is None:proc.terminate()
        save(O/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc()))
        progress('failed',pid=None,error=repr(e));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--register',action='store_true');args=p.parse_args()
    register() if args.register else run()
