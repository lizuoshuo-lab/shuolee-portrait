"""N11: one native background-support supplement, no geometry or image warp."""
from pathlib import Path
from datetime import datetime, timezone
import os, sys, time, json, hashlib, traceback
import numpy as np
import cv2
from PIL import Image, ImageDraw

R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
I=R/'video-update-20261027';B=I/'background-geometry';M=B/'scene-support-masks'
P=B/'scene-support-supplement-plan.json';O=B/'scene-support-supplement'
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

class Union:
    def __init__(self,n):self.p=list(range(n))
    def find(self,i):
        while self.p[i]!=i:self.p[i]=self.p[self.p[i]];i=self.p[i]
        return i
    def join(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.p[max(a,b)]=min(a,b)

def register():
    assert not P.exists() and not O.exists(),'Never repeat extraction'
    old=read(I/'background-geometry-plan.json');review=read(M/'visual-review.json')
    assert review['state']=='all7_overlay_images_reviewed'
    files=[B/'observations.json',I/'background-geometry-plan.json',B/'material-quarantine/visual-review.json',M/'result.json',M/'visual-review.json']
    for f in old['frames']:
        files += [B/f'frame_{f:06d}.png',B/f'features_{f:06d}.npz',M/f'recovered_{f:06d}.png',M/f'exclusion_{f:06d}.png']
    save(P,dict(stage='N11_observations',registered=now(),source_sha256=sha(Path(__file__)),
        hypothesis='The coarse person exclusion rectangle removed nearby static scene support. Add only recovered original-pixel observations to test this evidence gap, without changing K or camera model.',
        evidence='N9 physically confirmed family399 mismatch, but N10 family-only quarantine does not improve face. All7 independently segmented exclusion masks and32px guard have now been visually reviewed.',
        frames=old['frames'],input_video_sha256=old['source']['sha256'],K_integer_centers=old['K_integer_centers'],K_changed=False,
        native_domain='Existing undistorted/downsampled/cropped1520x2704 MP4 frames. Detect directly on unchanged fullsize grayscale using recoveredmask, not resized/rectified crops. No second undistortion.',
        detector=old['detector'],matching=old['matching'],
        scope='Exactly7 new SIFT calls, one per recoveredmask. Reuse existing original features/descriptors; do not redetect previous outer scene. Match combined descriptors for step1/2 and require three-view cycles containing a new feature.',
        isolation='Original1125 tracks andfit/heldout identities remain immutable. Group all newtracks and oldtracks via identical nodes or sameframe<=8px proximity, transitively. Any newfamily touching ANY oldtrack is isolated from both newfit andnewheldout, including oldFIT/heldout/399. Never move heldout tofit; do not addoldtrackobservations orremoveoldtracks. New within-frame conflicting tracks are retained separately but not eligible.',
        split='Freeze whole surviving newfamilies before geometry by SHA256(videoSHA|N11family|id)%5==0 heldout, elsefit. Rawisolatedtracks retained withreason. Originalheldout remainsunchanged.',
        review='For each frozenfit/heldout split and3x3 spatialbin of original rectangle, select up to3 wholefamilyrepresentatives by SHA256(id|N11review). One representative perfamily, first/middle/last original160px crops, up to54rows. Review all selectedrows beforeanycamera; ambiguous material can request native320pxcontext without rematching. No geometricerror exists duringselection. Only explicitly physically-reviewed static newfamilies may later beadded toanexploratorycamera, not uncheckednewfamilies.',
        support_gate='Stop beforecamera if fewerthan20 reviewedstaticFITfamilies or8 reviewedheldoutfamilies, fewerthan3 spatialbins, or anyframe hasfewerthan5 reviewednewFITobservations. This is a coarsecoveragegate, not subpixelcalibration acceptance; reviewbias andstatic uncertainty remainreported.',
        cause_scope='Any future before/after effect concerns newly admitted reviewed native background support. Mask selection and observation admission are part ofthis defined change; do not attribute it exclusively toK orsegmentation.',
        budget=dict(extraction_runs=1,frames=7,CPU_seconds=240,review_rows_max=54,GPU=0,new_camera_runs=0,training_steps=0),
        stop='Input/hash/domain failure or240seconds stops the run with evidence retained. No detector/ratio/radius sweep. If samples lack trusted static physical correspondence, do not fitnewcamera ortrain.',
        input_hashes={str(f):sha(f) for f in files}))
    print('N11_SUPPLEMENT_REGISTERED',flush=True)

def run():
    assert not O.exists(),'No duplicate supplement'
    p=read(P);assert p['source_sha256']==sha(Path(__file__))
    for f,h in p['input_hashes'].items():assert sha(Path(f))==h,f
    O.mkdir();start=time.monotonic();cv2.setNumThreads(4);cv2.setRNGSeed(1225)
    def progress(state,**kw):
        if time.monotonic()-start>p['budget']['CPU_seconds']:raise TimeoutError('Registered240s budget exhausted')
        save(O/'progress.json',dict(state=state,pid=os.getpid(),updated=now(),elapsed_seconds=time.monotonic()-start,**kw))
    progress('extracting_recovered_background')
    frames=p['frames'];old=read(B/'observations.json')['tracks'];original_snapshot=json.dumps(old,sort_keys=True)
    uv,desc,oldcounts,offsets,images={},{},{},[0],{}
    detector=p['detector'];sift=cv2.SIFT_create(nfeatures=detector['features_per_frame'],contrastThreshold=detector['contrast'],edgeThreshold=detector['edge_threshold'],sigma=detector['sigma'])
    feature_report=[];nodes=[]
    for i,f in enumerate(frames):
        image=cv2.imread(str(B/f'frame_{f:06d}.png'));mask=cv2.imread(str(M/f'recovered_{f:06d}.png'),0)
        assert image.shape[:2]==mask.shape==(2704,1520)
        kp,d=sift.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),mask)
        newuv=np.array([k.pt for k in kp],dtype=np.float32).reshape(-1,2)
        d=np.zeros((0,128),np.float32) if d is None else np.sqrt(d/(d.sum(axis=1,keepdims=True)+1e-12))
        previous=np.load(B/f'features_{f:06d}.npz');n=len(previous['uv']);oldcounts[f]=n
        assert len(newuv)==len(d)
        xy=np.rint(newuv).astype(int)
        assert np.all(mask[xy[:,1],xy[:,0]]>0)
        uv[f]=np.concatenate((previous['uv'],newuv));desc[f]=np.concatenate((previous['descriptor'],d))
        assert np.array_equal(uv[f][:n],previous['uv']) and np.array_equal(desc[f][:n],previous['descriptor'])
        np.savez_compressed(O/f'features_{f:06d}.npz',uv=uv[f],descriptor=desc[f],old_feature_count=np.int64(n),new_native_uv=newuv)
        nodes.extend((f,j) for j in range(len(uv[f])));offsets.append(len(nodes));images[f]=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        feature_report.append(dict(frame=f,old=n,new=len(newuv),combined=len(uv[f]),native_width=1520,native_height=2704,mask_sha256=sha(M/f'recovered_{f:06d}.png')))
        progress('extracting_recovered_background',frames_done=i+1)
    bf=cv2.BFMatcher(cv2.NORM_L2);pairs={};pair_report=[]
    for i,f in enumerate(frames):
        for step in p['matching']['pair_steps']:
            j=i+step
            if j>=len(frames):continue
            g=frames[j];ab=bf.knnMatch(desc[f],desc[g],k=2);ba=bf.knnMatch(desc[g],desc[f],k=2)
            def ratio(rows):return {row[0].queryIdx:row[0].trainIdx for row in rows if len(row)==2 and row[0].distance<p['matching']['ratio']*row[1].distance}
            a,b=ratio(ab),ratio(ba);pairs[i,j]={x:y for x,y in a.items() if b.get(y)==x}
            pair_report.append(dict(frames=[f,g],matches=len(pairs[i,j]),with_new=sum(x>=oldcounts[f] or y>=oldcounts[g] for x,y in pairs[i,j].items())))
            progress('matching_combined_frozen_plus_new_features',pairs_done=len(pairs))
    nodeunion=Union(len(nodes));accepted=set();cycles=0
    for i in range(len(frames)-2):
        for a,b in pairs[i,i+1].items():
            c=pairs[i+1,i+2].get(b)
            if c is None or pairs[i,i+2].get(a)!=c:continue
            if not any(v>=oldcounts[frames[ii]] for ii,v in [(i,a),(i+1,b),(i+2,c)]):continue
            ids=[offsets[i]+a,offsets[i+1]+b,offsets[i+2]+c];accepted.update(ids);nodeunion.join(ids[0],ids[1]);nodeunion.join(ids[1],ids[2]);cycles+=1
    components={}
    for n in sorted(accepted):components.setdefault(nodeunion.find(n),[]).append(nodes[n])
    new=[];conflicts=[]
    for index,ns in enumerate(components.values()):
        observations=[dict(frame=f,feature=j,uv=uv[f][j].tolist(),new_feature=j>=oldcounts[f]) for f,j in sorted(ns)]
        row=dict(track=1125+index,observations=observations)
        (new if len(set(f for f,j in ns))==len(ns) else conflicts).append(row)
    # Include every original family as an immutable anchor in transitive proximity grouping.
    combined=old+new+conflicts;families=Union(len(combined));old_family_first={}
    for i,t in enumerate(old):
        if t['family'] in old_family_first:families.join(i,old_family_first[t['family']])
        else:old_family_first[t['family']]=i
    buckets={};radius=p['matching']['family_merge_radius_px']
    for ti,t in enumerate(combined):
        for ob in t['observations']:
            v=np.array(ob['uv']);cell=np.floor(v/radius).astype(int);f=ob['frame']
            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    for tj,other in buckets.get((f,int(cell[0])+dx,int(cell[1])+dy),[]):
                        if np.linalg.norm(v-other)<=radius:families.join(ti,tj)
            buckets.setdefault((f,int(cell[0]),int(cell[1])),[]).append((ti,v))
    oldlinks={}
    for ti,t in enumerate(old):oldlinks.setdefault(families.find(ti),set()).add(t['family'])
    conflict_roots={families.find(ti) for ti in range(len(old)+len(new),len(combined))}
    for ti,t in enumerate(new,len(old)):
        fam=families.find(ti);links=sorted(oldlinks.get(fam,set()))
        t['family']=fam;t['old_family_links']=links;t['touches_conflicting_component']=fam in conflict_roots
        if links or fam in conflict_roots:t['split']='isolated';t['reason']='Touches original family or ambiguous within-frame component; preserve frozen identity andprevent validation leakage'
        else:
            digest=hashlib.sha256(f'{p["input_video_sha256"]}|N11family|{fam}'.encode()).hexdigest()
            t['split']='heldout' if int(digest[:8],16)%5==0 else 'fit'
    assert json.dumps(old,sort_keys=True)==original_snapshot
    eligible=[t for t in new if t['split']!='isolated'];representatives={}
    for t in sorted(eligible,key=lambda z:z['track']):representatives.setdefault(t['family'],t)
    rectangle=read(I/'background-geometry-plan.json')['excluded_rect'];x0,y0,x1,y1=rectangle
    def spatial(t):
        candidates=[o for o in t['observations'] if o['new_feature']];ob=candidates[len(candidates)//2]
        x,y=ob['uv'];col=int(np.clip((x-x0)*3/(x1-x0),0,2));row=int(np.clip((y-y0)*3/(y1-y0),0,2))
        return row,col
    selected=[]
    for split in ['fit','heldout']:
        for rr in range(3):
            for cc in range(3):
                pool=[t for t in representatives.values() if t['split']==split and spatial(t)==(rr,cc)]
                ordered=sorted(pool,key=lambda t:hashlib.sha256(f'{t["family"]}|N11review'.encode()).hexdigest())
                selected.extend(ordered[:3])
    assert len(selected)<=54
    allfamilyindex=[]
    for fam,t in representatives.items():
        allfamilyindex.append(dict(family=fam,representative=t['track'],split=t['split'],spatial_bin=list(spatial(t)),frames=[o['frame'] for o in t['observations']],tracks=[z['track'] for z in eligible if z['family']==fam]))
    rows=[]
    for i,t in enumerate(selected):
        obs=t['observations'];pick=[obs[0],obs[len(obs)//2],obs[-1]];canvas=Image.new('RGB',(480,160),(80,80,80))
        bounds=[]
        for j,ob in enumerate(pick):
            x,y=np.rint(ob['uv']).astype(int);box=[int(x-80),int(y-80),int(x+80),int(y+80)];canvas.paste(images[ob['frame']].crop(tuple(box)),(j*160,0));bounds.append(box)
        name=f'review_{i:02d}_family_{t["family"]}.png';canvas.save(O/name)
        rows.append(dict(row=i,file=name,family=t['family'],representative_track=t['track'],split=t['split'],spatial_bin=list(spatial(t)),observations=pick,crop_bounds=bounds,point_is_crop_center=True))
    for page in range((len(rows)+5)//6):
        subset=rows[page*6:(page+1)*6];montage=Image.new('RGB',(480,160*len(subset)),(80,80,80))
        for j,row in enumerate(subset):montage.paste(Image.open(O/row['file']),(0,160*j))
        montage.save(O/f'review_page_{page:02d}.png')
    for f in frames:
        overlay=images[f].copy();draw=ImageDraw.Draw(overlay)
        for t in representatives.values():
            for ob in t['observations']:
                if ob['frame']!=f:continue
                x,y=ob['uv'];color=(60,220,255) if t['split']=='fit' else (255,185,40);draw.ellipse((x-5,y-5,x+5,y+5),outline=color,width=2)
        overlay.save(O/f'coverage_{f:06d}.png')
    save(O/'observations.json',dict(stage='N11',old_observations_sha256=sha(B/'observations.json'),source_sha256=p['input_video_sha256'],pixel_convention='integer_pixel_centers',new_tracks=new,conflicting_tracks=conflicts,original_tracks_modified=False,all_materials_certified=False))
    save(O/'families.json',dict(eligible_families=allfamilyindex,old_family_links=[dict(track=t['track'],family=t['family'],links=t['old_family_links']) for t in new if t['split']=='isolated']))
    save(O/'visual-index.json',dict(rows=rows,pages=[f'review_page_{p:02d}.png' for p in range((len(rows)+5)//6)],selection='Preregistered 3x3region/split familyhash, no geometry',all_native_images_unchanged=True))
    assert sha(B/'observations.json')==p['input_hashes'][str(B/'observations.json')]
    result=dict(stage='N11_observations',updated=now(),state='complete_pending_physical_material_review',elapsed_seconds=time.monotonic()-start,
        features=feature_report,pairs=pair_report,cycles_with_new=cycles,new_candidate_tracks=len(new),conflicting_components=len(conflicts),
        isolated_tracks=sum(t['split']=='isolated' for t in new),eligible_tracks=len(eligible),eligible_fit_families=sum(t['split']=='fit' for t in representatives.values()),
        eligible_heldout_families=sum(t['split']=='heldout' for t in representatives.values()),review_rows=len(rows),review_fit=sum(t['split']=='fit' for t in selected),review_heldout=sum(t['split']=='heldout' for t in selected),
        isolation_radius_px=radius,old_observations_unchanged=True,old_split_unchanged=True,oldoptimized_geometry_loaded=False,K_changed=False,geometry_started=False,training_started=False,
        observation_sha256=sha(O/'observations.json'),family_sha256=sha(O/'families.json'),review_index_sha256=sha(O/'visual-index.json'))
    progress('complete_pending_physical_material_review');save(O/'result.json',result)
    save(O/'progress.json',dict(state='complete_pending_physical_material_review',pid=None,updated=now()))
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:
        assert len(sys.argv)==2 and sys.argv[1] in ['register','run'];globals()[sys.argv[1]]()
    except Exception as exc:
        if O.exists():save(O/'failure.json',dict(updated=now(),error=repr(exc),traceback=traceback.format_exc()))
        raise
