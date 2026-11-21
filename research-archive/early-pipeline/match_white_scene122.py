"""N12b: single native observation extraction, spatial quota, frozen families. No geometry."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='4'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,time,sys,traceback
import cv2
import numpy as np
from PIL import Image

R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
I=R/'video-update-20261027'; M=I/'clip03-scene-support'; P=I/'clip03-native-matching-plan.json'; O=I/'clip03-native-matching'
FRAMES=[184,200,216,232,248,264,280]
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rank(s):return hashlib.sha256(str(s).encode()).hexdigest()

class Union:
    def __init__(self,n):self.p=list(range(n))
    def find(self,i):
        while self.p[i]!=i:self.p[i]=self.p[self.p[i]];i=self.p[i]
        return i
    def join(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.p[max(a,b)]=min(a,b)

def register():
    assert not P.exists() and not O.exists()
    review=read(M/'visual-review.json');assert review['person_exclusion_suitable_for_next_feature_probe']
    src=read(I/'clip03-scene-support-plan.json')
    files=[I/'clip03-scene-support-plan.json',M/'visual-review.json',M/'result.json']
    for f in FRAMES:files.extend([M/f'frame_{f:06d}.png',M/f'person_{f:06d}.png',M/f'exclusion_{f:06d}.png'])
    save(P,dict(stage='N12b',registered=now(),source_sha256=sha(Path(__file__)),input_video_sha256=src['source']['sha256'],frames=FRAMES,
        hypothesis='New built-scene capture may supply identifiable static support at different depths. Test observation identity before fixedK geometry. This is cross-capture pipeline validation, not a controlled one-factor attribution for the hat case.',
        pixel_contract='Native1520x2704 alreadyprocessed export. Existing exactPNG to uint8 grayscale; no resize, rotation, undistortion or image warp.',
        K_integer_centers=src['K_integer_centers'],K_changed=False,
        detector=dict(method='SIFT detect on fullnative gray once, then deterministic spatial quotas and compute descriptors once',nfeatures=0,contrast=.02,edge_threshold=10,sigma=1.6,
                      background_bands_y=[0,901,1802,2704],max_features_per_background_band=2000,max_headROI_features=600,max_other_person_features=1400),
        eligibility='Background center outside32pxpersonexclusion; require distance to original person silhouette >32+ceil(5.304*SIFT_size+2), and descriptor support wholly insideimage. Radius conservatively encloses OpenCV SIFT4x4 descriptor. Retain raw detector locations/reasons. Human headROI is upper25percent of existing person detectorbox intersectedwithpersonsilhouette, not verifiedfaceanatomy. Person features neverfitcameras.',
        matching=dict(method='RootSIFT,L2BF,mutualratio',ratio=.75,pair_steps=[1,2],require_three_view_cycles=True,family_radius_px=8),
        grouping='Connected cycle tracks with duplicateframe conflicts preserved andisolated. Join wholetracks transitively bysameframe<=8px before frozenSHA(video|N12family|root)%5 heldout, otherwiseFIT. Mixedbackground/personfamilies also isolated fromcameras. Never move heldout toFIT.',
        review='For every3x3backgroundspatialbin andsplit, selectup to6FIT/3heldoutwholefamilyrepresentatives, longesttrack first thenSHA; inspect ALL native128px observations in up to7columns. Selectup to8FIT/8heldoutheadROI representatives and8FIT/8heldoutotherperson representatives separately. No residual exists duringselection. Optional atmost16 ambiguous static-background rows receive256pxcontext withoutredetection.',
        geometry_gate='Before anynewcamera: at least20visuallyidentifiable staticFITfamilyrepresentatives and8heldout, at least3imagebins, at least6FIT+3heldout onrigidnon-groundobjects. Selectedinitialadjacentpair musthave12sharedFIT; every subsequentcamera8reviewedFIT2D3D. This onlyallows coarsefixedK geometry; fullvideo/facephysicalsubpixelacceptance remainsseparate. Onlyreviewedstaticfamilyrepresentatives mayfit; unknown/backgrounddynamic excludedwithreasons beforegeometry, observationsretained.',
        budget=dict(extraction_runs=1,frames=7,CPU_seconds=360,background_review_rows_max=81,person_review_rows_max=32,context_rows_max=16,new_cameras=0,GPU=0,training_steps=0),
        stop='Hash/domain/identityfailure or360s -> preservefailure. No threshold/detector/ROI sweep. Lowcoarsesupport means no camerafit, notproofcaptureunreconstructable.',
        input_hashes={str(f):sha(f) for f in files}))
    print('N12B_PLAN_REGISTERED',flush=True)

def run():
    assert not O.exists();p=read(P);assert sha(Path(__file__))==p['source_sha256']
    for f,h in p['input_hashes'].items():assert sha(Path(f))==h,f
    O.mkdir();start=time.monotonic();cv2.setNumThreads(4);cv2.setRNGSeed(12212)
    def progress(state,**kw):
        assert time.monotonic()-start<360,'CPU budget exceeded'
        save(O/'progress.json',dict(state=state,pid=os.getpid(),updated=now(),elapsed_seconds=time.monotonic()-start,**kw))
    progress('extracting_native')
    detector=cv2.SIFT_create(nfeatures=0,contrastThreshold=.02,edgeThreshold=10,sigma=1.6)
    masks={r['frame']:r for r in read(M/'result.json')['rows']}
    uv=[];des=[];roles=[];offset=[0];nodes=[];images={};counts=[]
    limits=[2000,2000,2000,600,1400]
    for i,f in enumerate(FRAMES):
        bgr=cv2.imread(str(M/f'frame_{f:06d}.png')); gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
        person=cv2.imread(str(M/f'person_{f:06d}.png'),0)>0;exclusion=cv2.imread(str(M/f'exclusion_{f:06d}.png'),0)>0
        assert gray.shape==person.shape==(2704,1520)
        distance=cv2.distanceTransform((~person).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        box=masks[f]['people'][0]['detected_box'];head_bottom=int(np.ceil(box[1]+.25*(box[3]-box[1])))
        kp=detector.detect(gray,None)
        candidate_roles=np.full(len(kp),-1,np.int16);reason=np.zeros(len(kp),np.int8)
        for j,k in enumerate(kp):
            x,y=np.rint(k.pt).astype(int);radius=int(np.ceil(5.304*k.size+2))
            if min(x,y,1519-x,2703-y)<radius:reason[j]=1;continue
            if person[y,x]:candidate_roles[j]=3 if y<head_bottom else 4;continue
            if exclusion[y,x] or distance[y,x]<=32+radius:reason[j]=2;continue
            candidate_roles[j]=0 if y<901 else(1 if y<1802 else 2)
        ids=[]
        for region,limit in enumerate(limits):
            pool=np.flatnonzero(candidate_roles==region).tolist()
            pool.sort(key=lambda j:(-kp[j].response,kp[j].pt[1],kp[j].pt[0],kp[j].size,kp[j].angle))
            ids.extend(pool[:limit])
        picked=[kp[j] for j in ids]; selected,descriptor=detector.compute(gray,picked)
        assert len(selected)==len(ids) and descriptor is not None
        xy=np.array([k.pt for k in selected],np.float32)
        assert np.array_equal(xy,np.array([kp[j].pt for j in ids],np.float32))
        rootsift=np.sqrt(descriptor/(descriptor.sum(axis=1,keepdims=True)+1e-12)).astype(np.float32)
        selected_roles=candidate_roles[ids]
        np.savez_compressed(O/f'features_{f:06d}.npz',uv=xy,descriptor=rootsift,role=selected_roles,
            detector_uv=np.array([k.pt for k in kp],np.float32),detector_size=np.array([k.size for k in kp],np.float32),
            detector_response=np.array([k.response for k in kp],np.float32),detector_role=candidate_roles,exclusion_reason=reason,selected_detector_indices=np.array(ids,np.int32))
        uv.append(xy);des.append(rootsift);roles.append(selected_roles);nodes.extend((i,j) for j in range(len(xy)));offset.append(len(nodes))
        images[f]=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        counts.append(dict(frame=f,detected=len(kp),descriptor_border_excluded=int((reason==1).sum()),person_support_excluded=int((reason==2).sum()),selected=len(ids),by_role=np.bincount(selected_roles,minlength=5).tolist(),gray_sha256=hashlib.sha256(gray.tobytes()).hexdigest()))
        progress('extracting_native',done=i+1,total=7);print('N12B_FEATURES',f,len(ids),flush=True)
    pairs={};bf=cv2.BFMatcher(cv2.NORM_L2)
    for i in range(7):
        for step in [1,2]:
            j=i+step
            if j>=7:continue
            forward=bf.knnMatch(des[i],des[j],k=2);back=bf.knnMatch(des[j],des[i],k=2)
            a={v[0].queryIdx:v[0].trainIdx for v in forward if len(v)==2 and v[0].distance<.75*v[1].distance}
            b={v[0].queryIdx:v[0].trainIdx for v in back if len(v)==2 and v[0].distance<.75*v[1].distance}
            pairs[i,j]={x:y for x,y in a.items() if b.get(y)==x}
            progress('matching_native',pairs=len(pairs));print('N12B_PAIR',i,j,len(pairs[i,j]),flush=True)
    tracks_union=Union(len(nodes));accepted=set();cycles=0
    for i in range(5):
        for a,b in pairs[i,i+1].items():
            c=pairs[i+1,i+2].get(b)
            if c is not None and pairs[i,i+2].get(a)==c:
                keys=[offset[i]+a,offset[i+1]+b,offset[i+2]+c];accepted.update(keys);cycles+=1
                tracks_union.join(keys[0],keys[1]);tracks_union.join(keys[1],keys[2])
    raw={}
    for node in sorted(accepted):raw.setdefault(tracks_union.find(node),[]).append(nodes[node])
    components=list(raw.values());family_union=Union(len(components));buckets={};near_pairs=[]
    for ti,track in enumerate(components):
        for i,j in track:
            point=uv[i][j];cell=np.floor(point/8).astype(int)
            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    for tj,other in buckets.get((i,cell[0]+dx,cell[1]+dy),[]):
                        if np.linalg.norm(point-other)<=8:family_union.join(ti,tj);near_pairs.append((ti,tj))
            buckets.setdefault((i,*cell),[]).append((ti,point))
    for a,b in near_pairs:assert family_union.find(a)==family_union.find(b)
    rows=[];families={}
    for ti,track in enumerate(components):
        role_list=[int(roles[i][j]) for i,j in track]
        category='background' if all(r<3 for r in role_list) else('headROI' if all(r==3 for r in role_list) else('person' if all(r>=3 for r in role_list) else'mixed'))
        fam=family_union.find(ti);split='heldout' if int(rank(p['input_video_sha256']+'|N12family|'+str(fam))[:8],16)%5==0 else'fit'
        conflict=len(set(i for i,j in track))!=len(track)
        row=dict(track=ti,family=fam,split=split,category=category,duplicate_frame_conflict=conflict,
                 observations=[dict(frame=FRAMES[i],feature=j,uv=uv[i][j].tolist(),role=int(roles[i][j])) for i,j in track])
        rows.append(row);families.setdefault(fam,dict(family=fam,split=split,tracks=[],categories=[],duplicate_frame_conflict=False))
        families[fam]['tracks'].append(ti);families[fam]['categories'].append(category);families[fam]['duplicate_frame_conflict']|=conflict
    for fam,data in families.items():
        data['categories']=sorted(set(data['categories']));data['isolated_from_camera']=data['duplicate_frame_conflict'] or data['categories']!=['background']
    for r in rows:r['camera_candidate']=not families[r['family']]['isolated_from_camera']
    save(O/'observations.json',dict(source_sha256=p['input_video_sha256'],frames=FRAMES,pixel_convention='integer_pixel_centers',role_names=['background_upper','background_middle','background_lower','headROI_unverified','other_person_unverified'],tracks=rows,
         raw_pairs=[dict(frames=[FRAMES[i],FRAMES[j]],matches=list(d.items())) for(i,j),d in pairs.items()]))
    save(O/'families.json',dict(radius_px=8,frozen_before_geometry=True,rows=list(families.values())))
    reps=[]
    for fam,data in families.items():
        candidates=[rows[j] for j in data['tracks'] if not rows[j]['duplicate_frame_conflict']]
        if not candidates:continue
        candidates.sort(key=lambda t:(-len(t['observations']),rank(f'{t["track"]}|N12representative')))
        t=candidates[0];mid=t['observations'][len(t['observations'])//2]['uv'];binxy=[min(2,int(mid[0]/(1520/3))),min(2,int(mid[1]/(2704/3)))]
        reps.append(dict(**t,spatial_bin=binxy))
    chosen=[]
    for split,limit in [('fit',6),('heldout',3)]:
        for by in range(3):
            for bx in range(3):
                pool=[t for t in reps if t['camera_candidate'] and t['split']==split and t['spatial_bin']==[bx,by]]
                pool.sort(key=lambda t:(-len(t['observations']),rank(f'{t["family"]}|N12review')));chosen.extend(pool[:limit])
    for category in ['headROI','person']:
        for split in ['fit','heldout']:
            pool=[t for t in reps if not t['duplicate_frame_conflict'] and t['category']==category and t['split']==split]
            pool.sort(key=lambda t:(-len(t['observations']),rank(f'{t["family"]}|N12review')));chosen.extend(pool[:8])
    assert len(chosen)==len(set(t['family'] for t in chosen))
    review=[];pages=[];page=None
    for rowid,t in enumerate(chosen):
        strip=Image.new('RGB',(128*7,128),(80,80,80))
        observations=t['observations']
        for o in observations:
            col=FRAMES.index(o['frame']);x,y=np.rint(o['uv']).astype(int)
            strip.paste(images[o['frame']].crop((x-64,y-64,x+64,y+64)),(128*col,0))
        name=f'review_{rowid:03d}_family_{t["family"]}.png';strip.save(O/name)
        review.append(dict(row=rowid,file=name,track=t['track'],family=t['family'],split=t['split'],category=t['category'],spatial_bin=t['spatial_bin'],observations=observations,camera_candidate=t['camera_candidate']))
        if rowid%6==0:page=Image.new('RGB',(896,6*132),(235,235,235))
        page.paste(strip,(0,(rowid%6)*132))
        if rowid%6==5 or rowid==len(chosen)-1:
            name=f'review_page_{rowid//6:02d}.png';page.save(O/name);pages.append(name)
    save(O/'visual-index.json',dict(rows=review,pages=pages,columns=FRAMES,cell_size=128,grey='unobserved slot, not sourcepixel',selection='pregeometry spatial/split strata longesttrack thenfixedSHA; wholefamilies',annotations_in_image=False))
    progress('frozen_pending_actual_material_review',tracks=len(rows),review_rows=len(chosen))
    for f,h in p['input_hashes'].items():assert sha(Path(f))==h
    result=dict(stage='N12b',state='complete_pending_actual_material_review',updated=now(),elapsed_seconds=time.monotonic()-start,features=counts,three_view_cycles=cycles,tracks=len(rows),families=len(families),
        by_category={cat:sum(t['category']==cat for t in rows) for cat in ['background','headROI','person','mixed']},
        eligible_camera_families=sum(not t['isolated_from_camera'] for t in families.values()),
        camera_family_splits={split:sum(not t['isolated_from_camera'] and t['split']==split for t in families.values()) for split in ['fit','heldout']},
        review_rows=len(chosen),review_by_category={cat:sum(t['category']==cat for t in chosen) for cat in ['background','headROI','person']},
        duplicate_frame_conflict_tracks=sum(t['duplicate_frame_conflict'] for t in rows),
        hashes={name:sha(O/name) for name in ['observations.json','families.json','visual-index.json']},
        K_changed=False,geometry_started=False,physical_identity_certified=False,training_started=False)
    save(O/'result.json',result);save(O/'progress.json',dict(state='complete_pending_actual_material_review',pid=None,updated=now()))
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException as exc:
        if O.exists():
            save(O/'failure.json',dict(updated=now(),error=repr(exc),traceback=traceback.format_exc()))
            save(O/'progress.json',dict(state='failed',pid=None,updated=now()))
        raise
