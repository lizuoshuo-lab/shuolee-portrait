"""One bounded image-only audit; never writes a camera, source image or point cloud."""
from pathlib import Path
import datetime as dt, hashlib, json, os, sys, time, traceback
import cv2
import numpy as np
from PIL import Image

B = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
O = B/'current/portrait-causal-diagnosis-117'
D = O/'D2-material-correspondence'
J = B/'current/portrait-alignment-open-source-116/hsfm/joint-diagnostic-reserved'
M = B/'current/avatar-repository-pair-114/smplx/data/training-inputs.json'
KP = M.parent/'observations.npz'
LABELS = ['background', 'face_lower', 'face_upper', 'torso', 'legs_shoes']
EDGES = [(0,1),(1,0),(1,2),(2,1),(0,2),(2,0)]
SEMANTIC = [(23+30,'nose_tip'),(23+48,'mouth_left'),(23+54,'mouth_right'),
            (23+39,'eye_inner_left'),(23+42,'eye_inner_right'),
            (23+36,'eye_outer_left'),(23+45,'eye_outer_right'),
            (23,'jaw_end_left'),(23+8,'chin_contour'),(23+16,'jaw_end_right'),
            (5,'shoulder_left'),(6,'shoulder_right'),(9,'wrist_left'),
            (10,'wrist_right'),(15,'ankle_left'),(16,'ankle_right')]

def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, v):
    p=Path(p); q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8'); q.replace(p)
def stat(x):
    x=np.asarray(x); x=x[np.isfinite(x)]
    return dict(count=int(len(x)), median=float(np.median(x)) if len(x) else None,
                p90=float(np.quantile(x,.9)) if len(x) else None, max=float(x.max()) if len(x) else None)
def progress(stage, **kw):
    save(D/'progress.json', dict(updated=now(), pid=os.getpid(), stage=stage, **kw)); print(stage, kw, flush=True)

def register():
    assert not D.exists(), 'Existing audit must not be reset.'
    assert read(O/'queue-state.json')['state']=='D1_shared_K_rejected_D2_pending'
    rows=read(M)['records']; assert [r['source_frame_index'] for r in rows]==list(range(1,176))
    with np.load(J/'frozen-validation.npz') as z: frames, groups=z['frames'].copy(), z['groups'].copy()
    with np.load(KP) as z:
        shapes={k:list(z[k].shape) for k in z.files}
        assert z['keypoints'].shape==(175,133,2) and z['scores'].shape==(175,133)
    triples=np.unique(frames, axis=0).tolist()
    selected=[]
    for g in range(5):
        ids=np.flatnonzero(groups==g)
        selected.extend(ids.tolist() if g in (1,2) else ids[np.linspace(0,len(ids)-1,6,dtype=int)].tolist())
    D.mkdir()
    save(D/'preregistration.json', dict(created=now(), variable='Correspondence validation only; immutable074 cameras, all175 identities retained',
      inputs={str(p):sha(p) for p in [M,KP,J/'frozen-validation.npz']},
      source_images={str(f):rows[f-1]['source_sha256'] for f in sorted(set(frames.ravel().tolist()))},
      observations='All231 previously selected material triplets. Reused validation, neither fresh held-out GT nor new calibration data.',
      semantic='Compare16 same-index anatomical/contour labels per identical temporal triple. These labels are not material ground truth.',
      semantic_indices=SEMANTIC, semantic_score_min=.6, semantic_shapes=shapes, triples=triples,
      coordinate_contract='Rectified1514x2694, OpenCV integer pixel centers. No resize, crop-camera or distortion changes. KP and material bank already in same image space.',
      algorithm=dict(lk_window=21, lk_levels=3, lk_iterations=40, lk_epsilon=.001, lk_min_eigenvalue=1e-5,
                     initial_flow='Existing image correspondence, never projected camera positions. Conditional refinement, not independent rematching.',
                     edges=EDGES, cycles=[[0,1,2,0],[0,2,1,0]],
                     patch_width=21, local_search_radius=8, secondary_peak_exclude_radius=3),
      thresholds=dict(max_fb_px=1., max_cycle_px=1.5, max_endpoint_disagreement_px=2., min_patch_ncc=.85,
                      min_unique_peak_gap=.05, max_peak_displacement_px=2., min_patch_std=3.),
      gate_meaning='A conservative image-consistency screen only. Failure does not by itself distinguish wrong correspondence, occlusion, reflection, surface appearance or motion. Passing does not prove material identity or calibrated camera.',
      camera_metrics='Read-only A/C triangulation and held B projection reported for every original material/semantic track. Never used to select, modify or reject source tracks.',
      visual_material_ids=selected, visual_semantic_triples=[[38,50,63],[88,100,113]],
      visual='Original patches:128x128 raster crop centered on each original observation, uniformly enlarged2x; no warping or embedded text. Each row hasA/B/C. Filename/JSON preserve indices, crop boxes and approximate semantic groups.',
      budget_seconds=3600, stop_on=['hash/identity/shape mismatch','nonfinite input','budget exceeded'],
      no_weight_download=True, no_gpu=True, no_camera_fit=True, no_gaussian_training=True,
      no_threshold_sweep=True, script_sha256=sha(__file__), root_cause_resolved=False))
    q=read(O/'queue-state.json'); q.update(updated=now(),stage='D2',state='D2_registered',active_pid=None,
        next_action='Run the preregistered fixed-camera image-only material and semantic audit once. No training.')
    save(O/'queue-state.json',q)
    print(json.dumps(dict(registered=True,triples=len(triples),semantic_shapes=shapes,material_visual_count=len(selected))))

def ncc(a,b):
    a=a.astype(float); b=b.astype(float);a-=a.mean();b-=b.mean()
    return float(np.sum(a*b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12))

def audit_bank(frames, xy, score, images, rows, name, t0):
    N=len(frames); end=np.full((N,6,2),np.nan); fb=np.full((N,6),np.nan)
    status=np.zeros((N,6),bool); peak=np.full((N,6),np.nan); gaps=peak.copy(); shifts=peak.copy(); corr=peak.copy()
    texture=np.zeros((N,6),bool)
    opts=dict(winSize=(21,21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.001),
              flags=cv2.OPTFLOW_USE_INITIAL_FLOW, minEigThreshold=1e-5)
    for triple in np.unique(frames,axis=0):
        ids=np.flatnonzero(np.all(frames==triple,axis=1))
        for e,(u,v) in enumerate(EDGES):
            p=xy[ids,u].astype(np.float32).reshape(-1,1,2);guess=xy[ids,v].astype(np.float32).reshape(-1,1,2)
            dst, st, _=cv2.calcOpticalFlowPyrLK(images[int(triple[u])],images[int(triple[v])],p,guess.copy(),**opts)
            ret, sr, _=cv2.calcOpticalFlowPyrLK(images[int(triple[v])],images[int(triple[u])],dst,p.copy(),**opts)
            end[ids,e]=dst[:,0];fb[ids,e]=np.linalg.norm(ret[:,0]-p[:,0],axis=1);status[ids,e]=(st[:,0]>0)&(sr[:,0]>0)
            for idx in ids:
                a,b=xy[idx,u],xy[idx,v]; im1,im2=images[int(triple[u])],images[int(triple[v])]
                valid=all(20<=pnt[0]<im1.shape[1]-20 and 20<=pnt[1]<im1.shape[0]-20 for pnt in [a,b])
                if not valid: continue
                templ=cv2.getRectSubPix(im1,(21,21),tuple(map(float,a)))
                patch=cv2.getRectSubPix(im2,(21,21),tuple(map(float,b)))
                region=cv2.getRectSubPix(im2,(37,37),tuple(map(float,b)))
                texture[idx,e]=templ.std()>=3 and patch.std()>=3;corr[idx,e]=ncc(templ,patch)
                field=cv2.matchTemplate(region,templ,cv2.TM_CCOEFF_NORMED)
                py,px=np.unravel_index(np.argmax(field),field.shape);peak[idx,e]=float(field[py,px]);shifts[idx,e]=float(np.hypot(px-8,py-8))
                yy,xx=np.mgrid[:17,:17];other=field[((xx-px)**2+(yy-py)**2)>9]
                gaps[idx,e]=float(peak[idx,e]-other.max())
        assert time.monotonic()-t0<3600
    cycles=np.full((N,2),np.nan); cycle_status=np.ones((N,2),bool)
    for triple in np.unique(frames,axis=0):
        ids=np.flatnonzero(np.all(frames==triple,axis=1)); start=xy[ids,0].astype(np.float32).reshape(-1,1,2)
        for k,order in enumerate([[0,1,2,0],[0,2,1,0]]):
            cur=start.copy()
            for u,v in zip(order[:-1],order[1:]):
                cur,st,_=cv2.calcOpticalFlowPyrLK(images[int(triple[u])],images[int(triple[v])],cur,
                         xy[ids,v].astype(np.float32).reshape(-1,1,2).copy(),**opts)
                cycle_status[ids,k]&=st[:,0]>0
            cycles[ids,k]=np.linalg.norm(cur[:,0]-start[:,0],axis=1)
    disagreement=np.stack([np.linalg.norm(end[:,e]-xy[:,v],axis=1) for e,(_,v) in enumerate(EDGES)],axis=1)
    errors=np.full(N,np.nan); positive=np.zeros(N,bool)
    for i,(fs,ps) in enumerate(zip(frames,xy)):
        projs=[];ws=[]
        for f in fs:
            r=rows[int(f)-1];K=np.asarray(r['K']).copy(); K[:2,2]-=.5; W=np.asarray(r['w2c']);ws.append(W);projs.append(K@W[:3])
        x=cv2.triangulatePoints(projs[0],projs[2],ps[0].reshape(2,1),ps[2].reshape(2,1))[:,0]
        if not np.isfinite(x).all() or abs(x[3])<1e-12:continue
        x/=x[3];positive[i]=all((w@x)[2]>0 for w in ws)
        p=projs[1]@x
        if abs(p[2])>1e-12:errors[i]=np.linalg.norm(p[:2]/p[2]-ps[1])
    passed=status.all(1)&cycle_status.all(1)&texture.all(1)&(fb.max(1)<=1.)&(cycles.max(1)<=1.5)&(disagreement.max(1)<=2.)&(corr.min(1)>=.85)&(gaps.min(1)>=.05)&(shifts.max(1)<=2.)
    records=[]
    for i in range(N):
        reasons=[]
        for failed,reason in [(not status[i].all(),'LK_status'),(not cycle_status[i].all(),'cycle_status'),
            (not texture[i].all(),'insufficient_patch_texture_or_boundary'),(fb[i].max()>1,'forward_backward'),
            (cycles[i].max()>1.5,'cycle'),(disagreement[i].max()>2,'changed_endpoint'),
            (not np.all(corr[i]>=.85),'patch_appearance'),(not np.all(gaps[i]>=.05),'ambiguous_local_peak'),
            (not np.all(shifts[i]<=2),'local_peak_displacement')]:
            if failed:reasons.append(reason)
        finite=lambda x:float(x) if np.isfinite(x) else None
        records.append(dict(id=i,frames=frames[i].tolist(),xy=xy[i].tolist(),score_min=float(score[i].min()),
            max_fb_px=finite(fb[i].max()),max_cycle_px=finite(cycles[i].max()),max_endpoint_disagreement_px=finite(disagreement[i].max()),
            min_patch_ncc=finite(corr[i].min()),min_peak_gap=finite(gaps[i].min()),max_peak_shift_px=finite(shifts[i].max()),
            positive_depth=bool(positive[i]),held_middle_px=finite(errors[i]),image_screen_passed=bool(passed[i]),reasons=reasons))
    np.savez_compressed(D/(name+'-measurements.npz'),frames=frames,xy=xy,refined_end=end,fb=fb,cycle=cycles,patch_ncc=corr,
        peak_gap=gaps,peak_shift=shifts,endpoint_disagreement=disagreement,held_middle_px=errors,positive=positive,screen=passed)
    save(D/(name+'-records.json'),records)
    return records

def summary(rec, ids):
    rr=[rec[i] for i in ids]
    fields=['max_fb_px','max_cycle_px','max_endpoint_disagreement_px','min_patch_ncc','min_peak_gap','max_peak_shift_px','held_middle_px']
    return dict(count=len(rr),screen_passed=sum(r['image_screen_passed'] for r in rr),
        stats={f:stat([r[f] if r[f] is not None else np.nan for r in rr]) for f in fields},
        reasons={reason:sum(reason in r['reasons'] for r in rr) for reason in sorted(set(v for r in rr for v in r['reasons']))})

def export_images(rec, selection, rows, prefix, manifest):
    dest=D/'review';dest.mkdir(exist_ok=True)
    for i in selection:
        r=rec[i]; panel=Image.new('RGB',(768,256),'white');boxes=[]
        for col,(f,xy) in enumerate(zip(r['frames'],r['xy'])):
            x,y=np.round(xy).astype(int);box=[int(x-64),int(y-64),int(x+64),int(y+64)];boxes.append(box)
            with Image.open(rows[f-1]['image']) as im:
                patch=im.convert('RGB').crop(box).resize((256,256),Image.Resampling.NEAREST)
            panel.paste(patch,(col*256,0))
        name=f'{prefix}_track{i:03d}_'+ '_'.join(f'{f:06d}' for f in r['frames'])+'.png'
        panel.save(dest/name)
        manifest.append(dict(file=name,bank=prefix,track=i,frames=r['frames'],crop_boxes=boxes,xy=r['xy'],text_burned=False,
            scale=2,warped=False,semantic_name=r.get('semantic_name'),group=r.get('group'),image_screen_passed=r['image_screen_passed']))

def run():
    assert not (D/'result.json').exists(),'Audit already completed.'
    reg=read(D/'preregistration.json');assert sha(__file__)==reg['script_sha256']
    for p,h in reg['inputs'].items(): assert sha(p)==h,p
    t0=time.monotonic(); cv2.setNumThreads(4);rows=read(M)['records']; images={}
    q=read(O/'queue-state.json');q.update(state='D2_running',active_pid=os.getpid(),active_log='D2-material-correspondence/run.log',updated=now());save(O/'queue-state.json',q)
    for f,h in reg['source_images'].items():
        p=Path(rows[int(f)-1]['image']);assert sha(p)==h
        im=cv2.imread(str(p),0);assert im.shape==(2694,1514);images[int(f)]=im
    progress('material_audit',images=len(images))
    with np.load(J/'frozen-validation.npz') as z:frames,xy,groups=[z[k].copy() for k in ['frames','xy','groups']]
    material=audit_bank(frames,xy,np.ones((len(frames),3)),images,rows,'material',t0)
    for r,g in zip(material,groups):r['group']=LABELS[g]
    save(D/'material-records.json',material)
    progress('semantic_audit',material_tracks=len(material))
    with np.load(KP) as z:kp,sc=z['keypoints'].copy(),z['scores'].copy()
    sf=[];sx=[];ss=[];sn=[]
    for triple in reg['triples']:
        for j,n in SEMANTIC:
            sf.append(triple);sx.append(kp[np.array(triple)-1,j]);ss.append(sc[np.array(triple)-1,j]);sn.append(n)
    semantic=audit_bank(np.asarray(sf),np.asarray(sx),np.asarray(ss),images,rows,'semantic',t0)
    for r,n in zip(semantic,sn):r['semantic_name']=n;r['confidence_admitted']=r['score_min']>=.6
    save(D/'semantic-records.json',semantic)
    manifest=[];export_images(material,reg['visual_material_ids'],rows,'material',manifest)
    sel=[i for i,r in enumerate(semantic) if r['frames'] in reg['visual_semantic_triples'] and i%len(SEMANTIC)<10]
    export_images(semantic,sel,rows,'semantic',manifest);save(D/'review/image-manifest.json',manifest)
    result=dict(updated=now(),state='completed_visual_review_pending',seconds=time.monotonic()-t0,
        material={lab:summary(material,np.flatnonzero(groups==g)) for g,lab in enumerate(LABELS)},
        semantic={n:summary(semantic,[i for i,r in enumerate(semantic) if r['semantic_name']==n and r['confidence_admitted']]) for _,n in SEMANTIC},
        semantic_confidence_rejected=sum(not r['confidence_admitted'] for r in semantic),
        sources_unchanged=all(sha(p)==h for p,h in reg['inputs'].items()),
        quality_accepted=False,root_cause_resolved=False,cameras_changed=False,gaussian_training_started=False,
        limits=['The old231bank was selected with related KLT/SIFT criteria. This is a sensitivity/identity audit, not a new independent accuracy benchmark.',
                'Same-location appearance can change with viewpoint and glasses reflection; a patch failure does not prove a wrong match.',
                'Anatomical/occluding landmarks need not be textured material points, so photometric matching cannot establish all semantic joint accuracy.',
                'No alternate camera is fitted. Baseline camera uncertainty and subject micromotion remain confounded.',
                '13sourceframes in this bank cannot validate all175cameras or training quality.'])
    assert result['sources_unchanged'];save(D/'result.json',result)
    q=read(O/'queue-state.json');q.update(updated=now(),state='D2_visual_review_pending',active_pid=None,
        result='D2-material-correspondence/result.json',next_action='Inspect preregistered source patches and semantic identities. Do not equate cycle consistency with material truth or launch training.')
    save(O/'queue-state.json',q);progress('completed_visual_review_pending',seconds=time.monotonic()-t0)
    print(json.dumps({g:dict(count=v['count'],screen_passed=v['screen_passed']) for g,v in result['material'].items()}))

if __name__=='__main__':
    try: register() if sys.argv[1]=='register' else run()
    except Exception as e:
        if D.exists():save(D/'failure.json',dict(updated=now(),pid=os.getpid(),error=repr(e),traceback=traceback.format_exc()))
        raise
