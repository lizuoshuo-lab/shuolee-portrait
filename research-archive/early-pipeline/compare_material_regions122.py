"""N8: fixed-camera, target-excluded comparison of existing native material tracks."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, sys, time, traceback
import numpy as np
from PIL import Image, ImageDraw
from analyze_residual_structure122 import dlt, project, controls

R = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
I = R/'video-update-20261027'; B = I/'background-geometry'
M = I/'native-temporal-material'; G = B/'shared-path/solver-repair'
P = B/'material-region-plan.json'; O = B/'material-region-comparison'
GROUPS = ['face_candidate', 'clothing_candidate', 'footwear_candidate', 'ground_candidate']
def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(p.read_text())
def save(p, v): p.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def register():
    assert not P.exists() and not O.exists()
    seeds = read(M/'seeds.json')['seeds']
    ids = {g: [s['id'] for s in seeds if s['group']==g] for g in GROUPS}
    samples = {g: a if g=='face_candidate' else [a[i] for i in np.linspace(0,len(a)-1,6).round().astype(int)] for g,a in ids.items()}
    inputs = [M/'observations.npz', M/'seeds.json', M/'visual-review.json', G/'result.json', B/'residual-structure/result.json']
    inputs += [B/f'frame_{f:06d}.png' for f in read(G/'result.json')['actual_registered_frames']]
    save(P, dict(stage='N8', registered=now(), source_sha256=sha(Path(__file__)),
        preceding_evidence='N7 face residual is coherent per view; background control is distant in image/depth. No unique cause proved.',
        question='Does the frozen face residual also appear on clothing/footwear, compared with existing ground tracks under exactly the same seven cameras?',
        actions=['Reuse all307 existing N1 tracks and original validity, no new feature detection/tracking.',
                 'For each valid observation in the frozen7 frames, triangulate from all other valid frames; no target self-inclusion. Require at least2 support frames; report every invalid/unsupported slot.',
                 'Describe residual mean/RMS and coherence by region and view. Do not estimate a corrective camera, motion or image warp.',
                 'Review raw96x96 native patches for every face seed and six uniformly indexed seeds from each other region; selection is frozen before residual computation.'],
        unchanged=['FirstK4000basis','SevenN5cposes','AllFIT/holdoutfamilies','Allimages','AllN1observations'],
        independence='Allhumantracks were heldout fromN5camera fitting. N1ground andN5SIFT may overlap inphysicalidentity, so ground here is a descriptive same-scene negativecontrol, NOT a new independent holdout score. Existing material ambiguity persists.',
        budget=dict(runs=1, CPU_seconds=120, GPU=0),
        source_counts={g:len(a) for g,a in ids.items()}, review_seed_ids=samples,
        stop='No residual-based pruning, no new BA/pose/geometry fitting fortraining, no thresholdsweep. Coherence cannot by itself distinguish pose error, coherent trackingbias, rolling-shutter or personmotion.',
        input_hashes={str(p):sha(p) for p in inputs}))
    print('N8_REGISTERED', flush=True)

def stats(values):
    a=np.asarray(values,dtype=float)
    if not len(a): return dict(n=0)
    return dict(n=len(a), median=float(np.median(a)), p90=float(np.percentile(a,90)), max=float(a.max()))

def run():
    assert not O.exists(); plan=read(P); assert plan['source_sha256']==sha(Path(__file__))
    for p,h in plan['input_hashes'].items(): assert sha(Path(p))==h
    O.mkdir(); start=time.monotonic(); save(O/'progress.json',dict(state='running_readonly',pid=os.getpid(),updated=now()))
    checks=controls(); seeds=read(M/'seeds.json')['seeds']; data=np.load(M/'observations.npz')
    uv=data['uv']; valid=data['valid']; base=read(G/'result.json'); frames=base['actual_registered_frames']
    K=np.array(base['K_integer_centers']); cams={c['frame']:np.array(c['w2c']) for c in base['cameras']}
    records=[]
    for s in seeds:
        track=s['id']; available=[f for f in frames if valid[f,track] and np.isfinite(uv[f,track]).all()]
        for f in frames:
            row=dict(track=track,group=s['group'],original_split=s['split'],family=s['family'],frame=f,
                observation_valid=bool(valid[f,track]), support=[j for j in available if j!=f])
            if f not in available:
                row.update(state='existing_tracking_invalid',evaluated=False)
            elif len(row['support'])<2:
                row.update(state='insufficient_support',evaluated=False,observed=uv[f,track].tolist())
            else:
                point=dlt(K,cams,[(j,uv[j,track]) for j in row['support']]); pred,z,_=project(K,cams[f],point)
                zs=[project(K,cams[j],point)[1] for j in row['support']]
                okay=bool(np.isfinite(pred).all() and z>0 and min(zs)>0)
                row.update(state='evaluated' if okay else 'invalid_geometry',evaluated=okay,observed=uv[f,track].tolist())
                if okay:
                    delta=pred-uv[f,track]
                    row.update(predicted=pred.tolist(),delta=delta.tolist(),error_px=float(np.linalg.norm(delta)),depth=float(z))
            records.append(row)
    # Exact parity with the previous face experiment checks frame, UV and pose identity.
    face_old=base['validation']['face_candidates']['observations']; index={(r['track'],r['frame']):r for r in records}
    parity=[]
    for old in face_old:
        cur=index[(old['track'],old['target'])]
        assert cur['evaluated']; parity.append(np.max(abs(np.array(cur['predicted'])-old['predicted'])))
    assert max(parity)<1e-6 and len(face_old)==57
    per_view=[]; summary={}
    for g in GROUPS:
        subset=[r for r in records if r['group']==g]; good=[r for r in subset if r['evaluated']]
        summary[g]=dict(tracks=len({r['track'] for r in subset}),slots=len(subset),
            status_counts={state:sum(r['state']==state for r in subset) for state in sorted({r['state'] for r in subset})},
            error_px=stats([r['error_px'] for r in good]),count_ge1=sum(r['error_px']>=1 for r in good))
        for f in frames:
            a=[r for r in good if r['frame']==f]
            if not a:
                per_view.append(dict(group=g,frame=f,n=0));continue
            d=np.array([r['delta'] for r in a]); mean=d.mean(axis=0); energy=np.sum(d*d)
            per_view.append(dict(group=g,frame=f,n=len(a),mean_delta=mean.tolist(),error_px=stats(np.linalg.norm(d,axis=1)),
                RMS=float(np.sqrt(energy/len(a))),centered_descriptive_RMS=float(np.sqrt(np.mean(np.sum((d-mean)**2,axis=1)))),
                common_energy_fraction=float(len(a)*np.dot(mean,mean)/energy)))
    # Every selected patch is native, unscaled, and remains visible even if its track failed.
    patch_index=[]; rows=[]
    for g in GROUPS:
        for tid in plan['review_seed_ids'][g]: rows.append((g,tid))
    frames_rgb={f:Image.open(B/f'frame_{f:06d}.png').convert('RGB') for f in frames}
    for page in range((len(rows)+5)//6):
        chunk=rows[page*6:(page+1)*6]; sheet=Image.new('RGB',(96*len(frames),192*len(chunk)),(235,235,235))
        indices=[]
        for rowidx,(g,tid) in enumerate(chunk):
            for col,f in enumerate(frames):
                ob=index[(tid,f)]; xy=uv[f,tid]
                entry=dict(row=rowidx,frame=f,track=tid,group=g,state=ob['state'])
                if np.isfinite(xy).all():
                    x,y=np.rint(xy).astype(int)-48; crop=frames_rgb[f].crop((x,y,x+96,y+96))
                    sheet.paste(crop,(col*96,rowidx*192))
                    mark=crop.copy(); draw=ImageDraw.Draw(mark); u,v=xy-[x,y]
                    draw.ellipse((u-2,v-2,u+2,v+2),outline=(0,255,80) if ob['observation_valid'] else (255,0,0))
                    if ob['evaluated']:
                        a,b=np.array(ob['predicted'])-[x,y];draw.line((a-3,b,a+3,b),fill=(255,40,180));draw.line((a,b-3,a,b+3),fill=(255,40,180))
                    sheet.paste(mark,(col*96,rowidx*192+96));entry['crop']=[int(x),int(y),int(x+96),int(y+96)]
                indices.append(entry)
        file=f'native_regions_{page:02d}.png';sheet.save(O/file);patch_index.append(dict(file=file,rows=chunk,observations=indices))
    result=dict(stage='N8',updated=now(),state='numeric_complete_pending_native_review',elapsed_seconds=time.monotonic()-start,
        controls=checks,face_prediction_parity_max_px=float(max(parity)),input_hashes=plan['input_hashes'],
        frames=frames,groups=summary,per_view=per_view,cameras_changed=False,K_changed=False,features_extracted=False,
        points_estimated_for_training=False,training_started=False,unique_cause_proven=False,
        limitations=plan['independence']+' Coherence statistics are descriptive; no corrective transforms were applied.')
    assert time.monotonic()-start<120
    save(O/'result.json',result);save(O/'observations.json',records);save(O/'patch-index.json',patch_index)
    for p,h in plan['input_hashes'].items():assert sha(Path(p))==h
    save(O/'progress.json',dict(state='completed_pending_review',updated=now(),pid=None))
    print(json.dumps(dict(groups=summary,per_view=per_view)),flush=True)

if __name__=='__main__':
    try: globals()[sys.argv[1]]()
    except Exception as e:
        if O.exists():save(O/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc()))
        raise
