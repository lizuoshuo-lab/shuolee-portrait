"""N12i: exhaustively review existing background families at face image height."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import json,hashlib,time,sys,traceback
import numpy as np
from PIL import Image
from shared_background122 import project,triangulate,camera_center
I=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video-update-20261027')
G=I/'clip03-grid-coverage/array-cache-repair';D=I/'clip03-fresh-geometry';F=I/'clip03-roof-review';M=I/'clip03-scene-support'
O=I/'clip03-head-level-support';P=I/'clip03-head-level-support-plan.json'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()

def selection():
    v=read(G/'visual-index.json');review=read(G/'visual-review.json');frames=v['columns']
    faces=[v['rows'][r['row']] for r in review['rows'] if r['reviewed_material']=='face_surface_coarse']
    # Image-height coverage selector only, independent of reconstructed depths/errors.
    bands={f:[min(o['uv'][1] for t in faces for o in t['observations'] if o['frame']==f)-128,
              max(o['uv'][1] for t in faces for o in t['observations'] if o['frame']==f)+128] for f in frames}
    raw=read(G/'observations.json')['tracks'];byfam={}
    for t in raw:
        if t['category']=='background' and t['camera_candidate']:
            byfam.setdefault(t['family'],[]).append(t)
    touched={fam:ts for fam,ts in byfam.items() if any(bands[o['frame']][0]<=o['uv'][1]<=bands[o['frame']][1] for t in ts for o in t['observations'])}
    prior={t['family']:t for t in read(G/'allocation.json')['candidate_families']}
    prior.update({t['family']:t for t in v['rows'] if t['category']=='background'})
    prior.update({t['family']:t for t in read(F/'reviewed-support.json')['tracks']})
    chosen=[]
    for fam,ts in sorted(touched.items()):
        split={t['split'] for t in ts};assert len(split)==1 and next(iter(split)) in ['fit','heldout']
        if fam in prior:
            t=next(t for t in ts if t['track']==prior[fam]['track']);rule='frozen_prior_representative'
        else:
            t=sorted(ts,key=lambda t:(-len(t['observations']),hashlib.sha256(f'{t["track"]}|N12i_rep'.encode()).hexdigest()))[0];rule='longest_existing_track_hash_tiebreak'
        touch=[dict(track=tt['track'],frame=o['frame'],uv=o['uv']) for tt in ts for o in tt['observations'] if bands[o['frame']][0]<=o['uv'][1]<=bands[o['frame']][1]]
        chosen.append(dict(**t,representative_rule=rule,head_band_touches=touch,existing_members=[tt['track'] for tt in ts],prior_category=prior.get(fam,{}).get('semantic_category')))
    return frames,bands,chosen,faces

def register():
    assert not O.exists() and not P.exists()
    frames,bands,rows,faces=selection()
    inputs=[G/n for n in ['observations.json','families.json','allocation.json','visual-index.json','visual-review.json']]+[F/'reviewed-support.json',D/'result.json']+[M/f'frame_{f:06d}.png' for f in frames]
    save(P,dict(stage='N12i',registered=now(),hypothesis='The existing tracks may omit or misidentify fixed background at the subject image height and comparable depth. Diagnose this coverage gap before attributing face residuals to motion.',
        variable='Review coverage of all existing eligible background families touching the frozen face image-height band. No extraction, matching, coordinate refinement, new region detector or BA.',
        selection='Perframe min/max y of the six frozen coarse face families plus/minus128nativepixels, full image width. Include every camera-eligible background family with any member observation in that band. Retain prior representative when available, else longest existing member with deterministic hash tie. Include all representative observations, not just band observations. No residual/depth-based selection.',
        frames=frames,image_height_bands=bands,families=len(rows),split_counts=dict(Counter(t['split'] for t in rows)),
        geometry='Use the completed N12g cameras only for provisional depth/parallax context. No initialization, optimization or new candidate. Face depth reference uses four FIT-labelled face families; they were never cameraFIT. Triangulate unchanged observations; positive depth recorded, no residual threshold or exclusion. Ratio0.5to2 is a descriptive comparable-depth interval, not a physical acceptance gate.',
        review='Every selected family:128nativepatch and256context for allavailable frames, plus original scene. Distinguish opaque fixedtexture, foliage, glass/reflection, silhouette, repetitive fence and mixeddepth. Image-height band is not proof of physical height. Preserve prior judgments unless explicit new image evidence justifies a change. No admitted camera points solely by derived depth.',
        budget=dict(runs=1,CPU_wall_seconds=120,max_review_families=256,GPU=0,training=0),
        stop='Over256families -> save fullenumeration/counts only and stop; no posthoc selective review to pass. Without physically credible independent static support, record missing evidence and do not rerun camera BA or infer head motion. Even apositive coarsecandidate doesnot certify subpixel localization.',
        input_hashes={str(p):sha(p) for p in inputs},code_hashes={str(p):sha(p) for p in [Path(__file__),Path(__file__).parent/'shared_background122.py']},K_changed=False,training=False))
    print(json.dumps(dict(registered=True,families=len(rows),splits=dict(Counter(t['split'] for t in rows)),bands=bands)),flush=True)

def run():
    assert not O.exists();p=read(P)
    for path,h in {**p['input_hashes'],**p['code_hashes']}.items():assert sha(Path(path))==h
    O.mkdir();start=time.monotonic();frames,bands,tracks,faces=selection();assert len(tracks)==p['families']
    save(O/'enumeration.json',dict(frames=frames,bands=bands,rows=tracks))
    if len(tracks)>256:
        save(O/'result.json',dict(state='review_budget_exceeded_preserved',families=len(tracks),camera_fit=False));return
    save(O/'progress.json',dict(state='prepare_existing_native_support',pid=os.getpid(),updated=now()))
    result=read(D/'result.json');K=np.array(result['K_integer_centers']);cams={r['frame']:np.array(r['w2c']) for r in result['cameras']}
    faceXYZ=[triangulate(K,cams,[(o['frame'],o['uv']) for o in t['observations']]) for t in faces if t['split']=='fit']
    assert len(faceXYZ)==4 and np.isfinite(faceXYZ).all()
    refs={f:float(np.median(project(K,cams[f],np.array(faceXYZ))[1])) for f in frames};assert min(refs.values())>0
    ims={f:Image.open(M/f'frame_{f:06d}.png').convert('RGB') for f in frames}
    rows=[];pages=[];contexts=[]
    for n,t in enumerate(tracks):
        assert time.monotonic()-start<120
        X=triangulate(K,cams,[(o['frame'],o['uv']) for o in t['observations']]);finite=bool(np.isfinite(X).all())
        depth=[project(K,cams[o['frame']],X[None])[1][0] for o in t['observations']] if finite else []
        positive=bool(depth) and min(depth)>0
        ratios=[float(z/refs[o['frame']]) for z,o in zip(depth,t['observations'])] if positive else []
        rays=[X-camera_center(cams[o['frame']]) for o in t['observations']] if finite else []
        angles=[np.degrees(np.arctan2(np.linalg.norm(np.cross(a,b)),np.dot(a,b))) for i,a in enumerate(rays) for b in rays[i+1:]]
        strip=Image.new('RGB',(128*len(frames),128),(80,80,80));ctx=Image.new('RGB',(256*len(frames),256),(80,80,80))
        for obs in t['observations']:
            x,y=np.rint(obs['uv']).astype(int);c=frames.index(obs['frame'])
            a=ims[obs['frame']].crop((x-64,y-64,x+64,y+64));b=ims[obs['frame']].crop((x-128,y-128,x+128,y+128))
            assert np.array_equal(np.array(a),np.array(b.crop((64,64,192,192))))
            strip.paste(a,(128*c,0));ctx.paste(b,(256*c,0))
        file=f'patch_{n:03d}_family_{t["family"]}.png';cf=f'context_{n:03d}_family_{t["family"]}.png';strip.save(O/file);ctx.save(O/cf)
        rows.append(dict(**t,row=n,file=file,context_file=cf,positive_provisional_depth=bool(positive),
            median_depth_ratio_to_face=float(np.median(ratios)) if ratios else None,depth_ratios=ratios,
            max_provisional_angle_degrees=float(max(angles)) if angles else None,physical_static_verified=False))
        if n%6==0:page=Image.new('RGB',(896,792),(235,235,235))
        page.paste(strip,(0,(n%6)*132))
        if n%6==5 or n==len(tracks)-1:fn=f'patch_page_{n//6:02d}.png';page.save(O/fn);pages.append(fn)
        if n%4==0:cp=Image.new('RGB',(1792,1040),(235,235,235))
        cp.paste(ctx,(0,(n%4)*260))
        if n%4==3 or n==len(tracks)-1:fn=f'context_page_{n//4:02d}.png';cp.save(O/fn);contexts.append(fn)
    for path,h in p['input_hashes'].items():assert sha(Path(path))==h
    save(O/'visual-index.json',dict(rows=rows,frames=frames,pages=pages,context_pages=contexts,source_pixels_unchanged=True,text_on_images=False,
        face_depth_reference=refs,geometry_context_only=True))
    out=dict(stage='N12i',state='complete_pending_actual_review',updated=now(),elapsed_seconds=time.monotonic()-start,families=len(rows),
        split_counts=dict(Counter(t['split'] for t in rows)),positive_depth=sum(t['positive_provisional_depth'] for t in rows),
        comparable_provisional_depth=sum(t['median_depth_ratio_to_face'] is not None and .5<=t['median_depth_ratio_to_face']<=2 for t in rows),
        all_families_retained=True,new_features=False,new_matches=False,K_changed=False,new_camera=False,training=False)
    save(O/'result.json',out);save(O/'progress.json',dict(state=out['state'],pid=None,updated=now()));print(json.dumps(out),flush=True)
if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except Exception as e:
        if O.exists():save(O/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),updated=now()));save(O/'progress.json',dict(state='failed_preserved',pid=None,updated=now()))
        raise
