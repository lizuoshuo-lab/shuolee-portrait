"""N5: bounded new-video background F / fixed conditional-K E diagnostic."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, time, os, sys, traceback
import numpy as np
import cv2
from PIL import Image

R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
I=R/'video-update-20261027';O=I/'background-geometry';PLAN=I/'background-geometry-plan.json'
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

def register():
    assert not PLAN.exists() and not O.exists()
    source=read(I/'result.json')['videos'][1]
    assert source['sha256'].startswith('32be3063')
    K=read(I/'output-intrinsics-audit/result.json')['variants'][0]['K']
    p=dict(stage='N5',registered=now(),source=source,frames=[230,246,262,278,294,310,326],
        user_steering='Proceed with approximate supplied intrinsics to diagnose gross mapping/trajectory errors; subpixel is the final objective, not a requirement for starting a coarse diagnostic.',
        hypothesis='Compare uncalibrated pinhole epipolar F and fixed conditional-K calibrated E on identical native background observations and frozen heldout families. Large F/E disagreement suggests a calibration/central-camera inconsistency but does not uniquely identify its cause.',
        controls='Same new native frames, same observations/splits, same1px robust threshold; F and E have different DOF, so even heldout advantage is diagnostic rather than a unique causal proof. Not a rerun of old121; no old poses or3D are loaded.',
        K_integer_centers=K,K_source='N4 H0 approximate firstYAML rectifiedK + explicit half-pixel resize and ideal center crop. Never optimized or promoted as certified.',
        region='Outer background plus upper distant terrain and bottom ground. Exclude expanded central person rectangle [350,900,1100,2400]. These are candidate background materials, not automatically static.',
        excluded_rect=[350,900,1100,2400],
        detector=dict(method='nativeSIFT+RootSIFT',features_per_frame=6000,contrast=.02,edge_threshold=10,sigma=1.6),
        matching=dict(mutual=True,ratio=.75,pair_steps=[1,2],three_view_cycles_required=True,family_merge_radius_px=8),
        splits='Connected feature tracks and within-frame<=8px near duplicates grouped as whole physical-candidate families before any fit; SHA256(source|family)%5 held out. No error-based relabelling.',
        geometry=dict(seed=1225,robust_threshold_px=1.0,min_iterations=100,max_iterations=10000,min_fit=50,min_heldout=12),
        budget=dict(extraction_runs=1,geometry_runs=1,extraction_CPU_seconds=240,geometry_CPU_seconds=180,GPU=0),
        review='Before geometry, inspect all7 native thumbnails and24 deterministic three-view material rows across image regions and both splits; save unknown/contradiction flags without deleting by geometric error.',
        acceptance='This is coarse diagnosis only. Report every heldout Sampson error and spatial bins, not physical point reprojection ground truth; no fullframe or human training admission.',
        stop='Decode/identity mismatch or time budget -> preserve failure. Insufficient tracks or clearly invalid background support -> no camera promotion. No threshold scan.',
        new_undistortion=False,old_camera_or_geometry=False,training_started=False)
    save(PLAN,p);print('N5 registered',flush=True)

def extract():
    assert not O.exists(), 'Existing extraction must not repeat'
    O.mkdir();p=read(PLAN);start=time.monotonic();cv2.setNumThreads(4);cv2.setRNGSeed(1225)
    video=Path(p['source']['path']);assert sha(video)==p['source']['sha256']
    frames={};cap=cv2.VideoCapture(str(video));cap.set(cv2.CAP_PROP_ORIENTATION_AUTO,0)
    for fi in range(p['frames'][-1]+1):
        ok,img=cap.read();assert ok
        if fi in p['frames']:
            assert img.shape[:2]==(2704,1520)
            frames[fi]=img;cv2.imwrite(str(O/f'frame_{fi:06d}.png'),img)
    cap.release();sift=cv2.SIFT_create(nfeatures=6000,contrastThreshold=.02,edgeThreshold=10,sigma=1.6)
    observations=[];des=[];offset=[0];nodes=[];images=[]
    for i,fi in enumerate(p['frames']):
        mask=np.ones((2704,1520),np.uint8)*255;x0,y0,x1,y1=p['excluded_rect'];mask[y0:y1,x0:x1]=0
        kp,d=sift.detectAndCompute(cv2.cvtColor(frames[fi],cv2.COLOR_BGR2GRAY),mask)
        uv=np.array([k.pt for k in kp],np.float32);d=np.sqrt(d/(np.sum(d,axis=1,keepdims=True)+1e-12))
        observations.append(uv);des.append(d);nodes.extend([(i,j) for j in range(len(kp))]);offset.append(len(nodes))
        np.savez_compressed(O/f'features_{fi:06d}.npz',uv=uv,descriptor=d)
        images.append(Image.fromarray(cv2.cvtColor(frames[fi],cv2.COLOR_BGR2RGB)).resize((304,541),Image.Resampling.LANCZOS))
        save(O/'progress.json',dict(state='extracting',pid=os.getpid(),frames_done=i+1,updated=now(),GPU=0))
    montage=Image.new('RGB',(304*7,541),'white')
    for i,im in enumerate(images):montage.paste(im,(304*i,0))
    montage.save(O/'frame_overview.png')
    pairs={};bf=cv2.BFMatcher(cv2.NORM_L2)
    for i in range(7):
        for step in [1,2]:
            j=i+step
            if j>=7:continue
            ab=bf.knnMatch(des[i],des[j],k=2);ba=bf.knnMatch(des[j],des[i],k=2)
            f={m.queryIdx:m.trainIdx for m,n in ab if m.distance<.75*n.distance}
            b={m.queryIdx:m.trainIdx for m,n in ba if m.distance<.75*n.distance}
            pairs[i,j]={a:z for a,z in f.items() if b.get(z)==a}
    parent=list(range(len(nodes)))
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    accepted=set()
    for i in range(5):
        for a,b in pairs[i,i+1].items():
            c=pairs[i+1,i+2].get(b)
            if c is not None and pairs[i,i+2].get(a)==c:
                ids=[offset[i]+a,offset[i+1]+b,offset[i+2]+c]
                for t in ids:accepted.add(t)
                union(ids[0],ids[1]);union(ids[1],ids[2])
    tracks={}
    for a in sorted(accepted):tracks.setdefault(find(a),[]).append(nodes[a])
    tracks=[t for t in tracks.values() if len(set(i for i,j in t))==len(t)]
    # Families are a separate union of whole tracks; same-frame nearby SIFT copies cannot cross splits.
    tp=list(range(len(tracks)))
    def tf(a):
        while tp[a]!=a:tp[a]=tp[tp[a]];a=tp[a]
        return a
    buckets={}
    for ti,t in enumerate(tracks):
        for i,j in t:
            uv=observations[i][j];cell=tuple(np.floor(uv/8).astype(int))
            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    for tj,other in buckets.get((i,cell[0]+dx,cell[1]+dy),[]):
                        if np.linalg.norm(uv-other)<=8:
                            a,b=tf(ti),tf(tj);tp[max(a,b)]=min(a,b)
            buckets.setdefault((i,*cell),[]).append((ti,uv))
    rows=[]
    for ti,t in enumerate(tracks):
        fam=tf(ti);h=hashlib.sha256((p['source']['sha256']+'|N5family|'+str(fam)).encode()).hexdigest()
        rows.append(dict(track=ti,family=fam,split='heldout' if int(h[:8],16)%5==0 else 'fit',
            observations=[dict(frame=p['frames'][i],feature=j,uv=observations[i][j].tolist()) for i,j in t]))
    save(O/'observations.json',dict(source_sha256=p['source']['sha256'],pixel_convention='integer_pixel_centers',tracks=rows,
        raw_pairs=[dict(frames=[p['frames'][i],p['frames'][j]],matches=list(d.items())) for (i,j),d in pairs.items()]))
    # Deterministic spatial and split coverage, selected before any geometric residual exists.
    chosen=[]
    for split in ['fit','heldout']:
        for region in ['upper','left','right','bottom']:
            pool=[]
            for t in rows:
                if t['split']!=split:continue
                uv=t['observations'][len(t['observations'])//2]['uv'];x,y=uv
                reg='upper' if y<1300 else ('bottom' if y>=2400 else ('left' if x<350 else 'right'))
                if reg==region:pool.append(t)
            order=sorted(pool,key=lambda t:hashlib.sha256(f'{t["track"]}|N5review'.encode()).hexdigest())
            chosen.extend(order[:3])
    review=[]
    for ti,t in enumerate(chosen):
        obs=t['observations'];pick=[obs[0],obs[len(obs)//2],obs[-1]];canvas=Image.new('RGB',(288,96),(90,90,90))
        for ci,o in enumerate(pick):
            x,y=np.rint(o['uv']).astype(int);img=Image.fromarray(cv2.cvtColor(frames[o['frame']],cv2.COLOR_BGR2RGB))
            # Pure crop and layout, no warping or annotation. Edge bounds remain explicit in JSON.
            canvas.paste(img.crop((x-48,y-48,x+48,y+48)),(ci*96,0))
        name=f'review_{ti:02d}_track_{t["track"]}.png';canvas.save(O/name)
        review.append(dict(file=name,track=t['track'],family=t['family'],split=t['split'],observations=pick))
    save(O/'visual-index.json',dict(rows=review,all_frames=p['frames'],selection='frozen before geometry'))
    assert time.monotonic()-start<=240
    result=dict(stage='N5_observations',updated=now(),elapsed_seconds=time.monotonic()-start,source_sha256=p['source']['sha256'],
        frames=p['frames'],features=[len(x) for x in observations],cycle_tracks=len(rows),
        fit_tracks=sum(t['split']=='fit' for t in rows),heldout_tracks=sum(t['split']=='heldout' for t in rows),
        review_rows=len(review),geometry_started=False,training_started=False,physical_identity_validated=False,
        observation_sha256=sha(O/'observations.json'))
    save(O/'extraction-result.json',result);save(O/'progress.json',dict(state='pending_material_review',pid=None,updated=now()))
    print(json.dumps(result),flush=True)

def sampson(a,b,F):
    aa=np.c_[a,np.ones(len(a))];bb=np.c_[b,np.ones(len(b))];fa=aa@F.T;fb=bb@F
    return np.sum(bb*fa,axis=1)/np.sqrt(np.maximum(1e-20,np.sum(fa[:,:2]**2+fb[:,:2]**2,axis=1)))
def stats(v):
    v=np.abs(v);return dict(n=len(v),median=float(np.median(v)),p90=float(np.percentile(v,90)),max=float(v.max()),fraction_lt1=float(np.mean(v<1))) if len(v) else dict(n=0)
def geometry():
    assert not (O/'geometry-result.json').exists()
    review=read(O/'visual-review.json');assert review['allow_exploratory_geometry'] and not review['physical_admission']
    p=read(PLAN);e=read(O/'extraction-result.json');assert sha(O/'observations.json')==e['observation_sha256']
    sys.path.insert(0,'/opt/gs-slam-lab/camera-contract120-tools/poselib-2.0.5');import poselib
    obs=read(O/'observations.json')['tracks'];start=time.monotonic();K=np.array(p['K_integer_centers']);inv=np.linalg.inv(K)
    face_path=I/'native-temporal-material/observations.npz'
    assert sha(face_path)==p['face_candidate_control']['observation_sha256']
    face=np.load(face_path)
    cam=dict(model='PINHOLE',width=1520,height=2704,params=[K[0,0],K[1,1],K[0,2],K[1,2]])
    results=[];allF=[];allE=[]
    for i in range(6):
        for step in [1,2]:
            if i+step>=7:continue
            fa,fb=p['frames'][i],p['frames'][i+step];items=[]
            for t in obs:
                d={o['frame']:o for o in t['observations']}
                if fa in d and fb in d:items.append((t,d[fa]['uv'],d[fb]['uv']))
            fit=[v for v in items if v[0]['split']=='fit'];held=[v for v in items if v[0]['split']=='heldout']
            row=dict(frames=[fa,fb],fit=len(fit),heldout=len(held))
            if len(fit)<50 or len(held)<12:row['state']='insufficient_support';results.append(row);continue
            a=np.array([v[1] for v in fit]);b=np.array([v[2] for v in fit]);ha=np.array([v[1] for v in held]);hb=np.array([v[2] for v in held])
            cv2.setRNGSeed(1225)
            F,mask=cv2.findFundamentalMat(a,b,cv2.FM_RANSAC,1.,.999,10000)
            pose,info=poselib.estimate_relative_pose(a,b,cam,cam,dict(max_epipolar_error=1.,min_iterations=100,max_iterations=10000,seed=1225),{})
            if F is None or F.shape!=(3,3) or pose is None:row['state']='solver_failed';results.append(row);continue
            t=pose.t;skew=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
            FE=inv.T@skew@pose.R@inv;ef=sampson(ha,hb,F);ee=sampson(ha,hb,FE)
            radius=np.linalg.norm(ha-[K[0,2],K[1,2]],axis=1)/(np.hypot(1520,2704)/2)
            row.update(state='estimated_diagnostic',F=F.tolist(),E_R=pose.R.tolist(),E_t_direction=pose.t.tolist(),
                F_fit_inliers=int(mask.sum()),E_fit_inliers=int(info['num_inliers']),
                F_heldout=stats(ef),E_heldout=stats(ee),
                heldout_observations=[dict(track=v[0]['track'],family=v[0]['family'],a=v[1],b=v[2],radial_fraction=float(r),F_signed_sampson_px=float(f),E_signed_sampson_px=float(e)) for v,r,f,e in zip(held,radius,ef,ee)])
            ids=[z for z in range(9) if bool(face['valid'][fa,z]) and bool(face['valid'][fb,z])]
            xa=face['uv'][fa,ids];xb=face['uv'][fb,ids]
            fface=sampson(xa,xb,F);eface=sampson(xa,xb,FE)
            row['unverified_face_candidates']=[dict(track=z,a=a.tolist(),b=b.tolist(),F_signed_sampson_px=float(f),E_signed_sampson_px=float(e),
                source='N1 frozen raw temporal candidates; never used in camera fit',physical_reprojection_gt=False) for z,a,b,f,e in zip(ids,xa,xb,fface,eface)]
            results.append(row);allF.extend(ef);allE.extend(ee)
            save(O/'geometry-progress.json',dict(state='running',updated=now(),pid=os.getpid(),pairs=len(results)))
            assert time.monotonic()-start<=180
    save(O/'geometry-result.json',dict(stage='N5_geometry',updated=now(),pairs=results,
        aggregate_F_heldout=stats(np.array(allF)),aggregate_fixedK_E_heldout=stats(np.array(allE)),
        K_integer_centers=K.tolist(),K_optimized=False,full_camera_trajectory=False,physical_calibration_accepted=False,
        error_kind='Sampson epipolar consistency, NOT independent material-point reprojection error or face accuracy',
        observation_sha256=e['observation_sha256'],elapsed_seconds=time.monotonic()-start,training_started=False))
    save(O/'geometry-progress.json',dict(state='complete_pending_review',updated=now(),pid=None));print('N5 GEOMETRY COMPLETE',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['register','extract','geometry']);args=parser.parse_args()
    try:globals()[args.stage]()
    except Exception as exc:
        if O.exists():save(O/(args.stage+'-failure.json'),dict(updated=now(),error=repr(exc),traceback=traceback.format_exc()))
        raise
