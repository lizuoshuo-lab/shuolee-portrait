"""Frozen native appearance candidates for new video10; never solve cameras."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,sys,time,traceback,itertools
import numpy as np
from PIL import Image,ImageDraw

W=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001') if os.name=='nt' else Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=W/'shuolee/current/robust-portrait-122'; I=R/'video9-10-native-intake'; O=R/'video10-static-material'; P=O/'plan.json'
FRAMES=[0,107,215,322,430,537,644,752,859]
BODY=[(1080,280,1630,1520),(1040,270,1610,1520),(1020,170,1610,1520),(1040,90,1720,1520),(930,90,1810,1520),(1010,100,1820,1520),(1020,100,1760,1520),(1020,130,1710,1520),(1080,180,1650,1520)]
HEAD=[(1210,310,1480,640),(1220,300,1540,700),(1160,170,1540,650),(1220,90,1590,570),(1190,100,1590,650),(1210,100,1610,650),(1180,100,1560,640),(1170,130,1540,640),(1190,180,1500,650)]
# These are conservative background exclusions, not an anatomical segmentation.
EXCLUDE=[[(0,180,280,940),(400,900,1080,1520)],[(390,420,1040,1520),(2130,540,2704,1440)],[(1780,430,2704,1200),(1790,1200,2704,1520)],[(900,430,1640,930),(1600,950,2704,1520),(1480,560,1820,1000)],[(0,420,840,1220),(1730,700,2704,1520)],[(200,800,900,1520),(1550,650,1990,1330)],[(0,860,1010,1520),(1650,900,1970,1520)],[(400,850,1040,1520),(0,370,440,1150)],[(0,220,310,940),(400,910,1080,1520)]]
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8'); tmp.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rank(s):return hashlib.sha256(str(s).encode()).hexdigest()
def image_path(f):return I/'video10'/f'native_frame_{f:06d}.png'
def inside(pt,b):return b[0]<=pt[0]<b[2] and b[1]<=pt[1]<b[3]
class Union:
    def __init__(self,n):self.p=list(range(n))
    def find(self,i):
        while self.p[i]!=i:self.p[i]=self.p[self.p[i]];i=self.p[i]
        return i
    def join(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.p[max(a,b)]=min(a,b)

def register():
    assert not O.exists(),'Already registered: inspect existing state'
    intake=read(I/'content-review.json'); assert intake['actual_saved_visual_review']
    source=next(v for v in read(I/'result.json')['videos'] if v['name']=='video10.mp4')
    assert [s['index_zero_based'] for s in source['samples']]==FRAMES
    O.mkdir();files={str(f):sha(image_path(f)) for f in FRAMES}
    save(P,dict(registered=now(),state='registered_regions_pending_review',source_script_sha256=sha(Path(__file__)),
      input_video_sha256=source['sha256'],frames=FRAMES,native_size=[2704,1520],image_sha256=files,
      hypothesis='This newly received indoor orbit supplies nearby opaque architectural material at different heights/depths. Test physical support before one fixed-conditional-K camera candidate; not a single-factor attribution for grayT.',
      pixel_contract='Existing native PNGs, RGB decoded to OpenCV gray only. No resize, second undistortion, lens fitting, camera or fitted-body input.',
      regions=dict(body_boxes=BODY,head_boxes=HEAD,background_exclusions=EXCLUDE,meaning='Conservative hand-reviewed ROI only; head/body membership is not physical identity. Remaining background also needs material review.'),
      detector=dict(method='SIFT',nfeatures=0,contrast=.02,edge_threshold=10,sigma=1.6,background_grid=[8,5],per_cell=100,head_quota=600,body_quota=1000,support_radius='ceil(5.304*size+2)',person_guard_px=16),
      matching=dict(descriptor='RootSIFT',method='mutual BF L2 ratio',ratio=.75,pairs='all36pairs among9registeredimages',cycles='all84three-viewtriplets',family_radius_native_px=8),
      split='Freeze20percent whole-family hash HELD before geometry. Same-frame8px-neighbor tracks joined transitively. Duplicate-frame or background/subject mixed families excluded from camera fitting, raw kept.',
      review='One longest track per family, tie SHA. Spatial3x3 background strata up to6FIT/3HELD each; head and other body up to8FIT/8HELD each. At most113rows plus16 ambiguous native contexts. Inspect all observations; no residual-based selection.',
      geometry_gate='No geometry in this job. Coarse next-stage eligibility at least20individually reviewed fixed FITfamilies and8HELD, at least3bins, at least6FIT/3HELD nonground rigid surfaces and multi-depth support. Initialpair12sharedFIT, subsequentviews8reviewedFIT2D3D. Fullframes/subpixel face/body/background acceptance separate.',
      budget=dict(extraction_runs=1,frames=9,CPU_wall_seconds=600,max_review_rows=113,max_context_rows=16,GPU_tasks=0,cameras=0,training_steps=0),
      stop='Input mismatch/time exceeded/support missing: preserve evidence; no parameter sweep, repeated extraction, interpolation or heldout reassignment. This sparse check is not fullvideo reconstruction.'))
    for j in range(3):
        page=Image.new('RGB',(1352,2280),'white')
        for k in range(3):
            n=j*3+k;im=Image.open(image_path(FRAMES[n])).convert('RGBA'); layer=Image.new('RGBA',im.size,(0,0,0,0));dr=ImageDraw.Draw(layer)
            for b in EXCLUDE[n]:dr.rectangle(b,fill=(240,40,40,95),outline=(240,40,40,255),width=5)
            dr.rectangle(BODY[n],fill=(130,50,240,50),outline=(130,50,240,255),width=6)
            dr.rectangle(HEAD[n],outline=(255,180,0,255),width=6)
            page.paste(Image.alpha_composite(im,layer).convert('RGB').resize((1352,760)),(0,760*k))
        page.save(O/f'regions_{j:02d}.jpg',quality=93)
    save(O/'progress.json',dict(updated=now(),state='registered_regions_pending_review',pid=None))
    print('Plan registered; inspect regions before run.')

def run():
    import cv2
    p=read(P);assert sha(Path(__file__))==p['source_script_sha256']
    assert read(O/'region-review.json')['safe_for_candidate_extraction']
    for f in FRAMES:assert sha(image_path(f))==p['image_sha256'][str(f)]
    out=O/'run'; out.mkdir(); start=time.monotonic();cv2.setNumThreads(4);cv2.setRNGSeed(12310)
    def state(status,**kw):
        assert time.monotonic()-start<600,'Frozen CPU wall budget exhausted'
        save(O/'progress.json',dict(updated=now(),state=status,pid=os.getpid(),elapsed=time.monotonic()-start,**kw))
    state('extracting_native_candidates'); imgs={};xy=[];desc=[];roles=[];offset=[0];nodes=[];counts=[]
    detector=cv2.SIFT_create(nfeatures=0,contrastThreshold=.02,edgeThreshold=10,sigma=1.6)
    for i,f in enumerate(FRAMES):
        rgb=np.array(Image.open(image_path(f)).convert('RGB'));assert rgb.shape==(1520,2704,3)
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);imgs[f]=Image.fromarray(rgb)
        forbidden=np.zeros(gray.shape,np.uint8);cv2.rectangle(forbidden,BODY[i][:2],BODY[i][2:],1,-1)
        for b in EXCLUDE[i]:cv2.rectangle(forbidden,b[:2],b[2:],1,-1)
        dist=cv2.distanceTransform(1-forbidden,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        keys=detector.detect(gray,None);rawrole=np.full(len(keys),-1,np.int16);reason=np.zeros(len(keys),np.uint8)
        for z,k in enumerate(keys):
            x,y=np.rint(k.pt).astype(int);rad=int(np.ceil(5.304*k.size+2))
            if min(x,y,2703-x,1519-y)<rad:reason[z]=1;continue
            if inside(k.pt,HEAD[i]):rawrole[z]=40;continue
            if inside(k.pt,BODY[i]):rawrole[z]=41;continue
            if dist[y,x]<=rad+16:reason[z]=2;continue
            rawrole[z]=min(4,int(y/304))*8+min(7,int(x/338))
        selected=[]
        for region in range(42):
            pool=np.flatnonzero(rawrole==region).tolist();pool.sort(key=lambda z:(-keys[z].response,keys[z].pt,keys[z].angle,keys[z].size))
            selected.extend(pool[:(100 if region<40 else 600 if region==40 else 1000)])
        picked,ds=detector.compute(gray,[keys[z] for z in selected]); assert len(picked)==len(selected) and ds is not None
        uv=np.array([k.pt for k in picked],np.float32);assert np.array_equal(uv,np.array([keys[z].pt for z in selected],np.float32))
        ds=np.sqrt(ds/(ds.sum(axis=1,keepdims=True)+1e-12)).astype(np.float32);rr=rawrole[selected]
        np.savez_compressed(out/f'features_{f:06d}.npz',uv=uv,descriptor=ds,role=rr,selected_indices=np.array(selected),raw_uv=np.array([k.pt for k in keys]),raw_size=np.array([k.size for k in keys]),raw_response=np.array([k.response for k in keys]),raw_angle=np.array([k.angle for k in keys]),raw_role=rawrole,exclusion_reason=reason)
        xy.append(uv);desc.append(ds);roles.append(rr);nodes.extend((i,z) for z in range(len(uv)));offset.append(len(nodes))
        counts.append(dict(frame=f,detected=len(keys),selected=len(uv),background=int((rr<40).sum()),headROI=int((rr==40).sum()),bodyROI=int((rr==41).sum())))
        state('extracting_native_candidates',frames_done=i+1);print('FEATURES',f,len(uv),flush=True)
    bf=cv2.BFMatcher(cv2.NORM_L2);pairs={}
    for a,b in itertools.combinations(range(9),2):
        def direction(s,t):return {m[0].queryIdx:m[0].trainIdx for m in bf.knnMatch(s,t,k=2) if len(m)==2 and m[1].distance>1e-12 and m[0].distance<.75*m[1].distance}
        ab=direction(desc[a],desc[b]);ba=direction(desc[b],desc[a]);pairs[a,b]={x:y for x,y in ab.items() if ba.get(y)==x}
        np.savez_compressed(out/f'pair_{a}_{b}.npz',matches=np.array(list(pairs[a,b].items()),np.int32).reshape(-1,2))
        state('matching_candidates',pairs_done=len(pairs));print('PAIR',a,b,len(pairs[a,b]),flush=True)
    u=Union(len(nodes));accepted=set();cycles=0
    for a,b,c in itertools.combinations(range(9),3):
        for x,y in pairs[a,b].items():
            z=pairs[b,c].get(y)
            if z is not None and pairs[a,c].get(x)==z:
                ns=[offset[a]+x,offset[b]+y,offset[c]+z];accepted.update(ns);u.join(ns[0],ns[1]);u.join(ns[1],ns[2]);cycles+=1
    components={}
    for node in sorted(accepted):components.setdefault(u.find(node),[]).append(nodes[node])
    tracks=list(components.values());fu=Union(len(tracks));buckets={};near=[]
    for ti,tr in enumerate(tracks):
        for a,z in tr:
            pt=xy[a][z];cell=np.floor(pt/8).astype(int)
            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    for other,q in buckets.get((a,cell[0]+dx,cell[1]+dy),[]):
                        if np.linalg.norm(pt-q)<=8:fu.join(ti,other);near.append((ti,other))
            buckets.setdefault((a,*cell),[]).append((ti,pt))
    rows=[];families={}
    for ti,tr in enumerate(tracks):
        rr=[int(roles[a][z]) for a,z in tr];cat='background' if all(v<40 for v in rr) else 'headROI' if all(v==40 for v in rr) else 'bodyROI' if all(v>=40 for v in rr) else 'mixed'
        fam=fu.find(ti);split='heldout' if int(rank(p['input_video_sha256']+'|video10-native-family|'+str(fam))[:8],16)%5==0 else 'fit'
        conflict=len(set(a for a,z in tr))!=len(tr)
        row=dict(track=ti,family=fam,split=split,category=cat,duplicate_frame_conflict=conflict,observations=[dict(frame=FRAMES[a],feature=z,uv=xy[a][z].tolist(),role=int(roles[a][z])) for a,z in tr]);rows.append(row)
        fa=families.setdefault(fam,dict(family=fam,split=split,tracks=[],categories=[],conflict=False));fa['tracks'].append(ti);fa['categories'].append(cat);fa['conflict']|=conflict
    for a,b in near:assert fu.find(a)==fu.find(b)
    for fam,fa in families.items():fa['categories']=sorted(set(fa['categories']));fa['candidate_background']=not fa['conflict'] and fa['categories']==['background']
    reps=[]
    for fam,fa in families.items():
        pool=[rows[t] for t in fa['tracks'] if not rows[t]['duplicate_frame_conflict']]
        if not pool:continue
        t=sorted(pool,key=lambda t:(-len(t['observations']),rank(f'{t["track"]}|representative')))[0];pt=t['observations'][len(t['observations'])//2]['uv'];binxy=[min(2,int(pt[0]/(2704/3))),min(2,int(pt[1]/(1520/3)))]
        reps.append(dict(**t,spatial_bin=binxy,candidate_background=fa['candidate_background']))
    chosen=[]
    for split,limit in [('fit',6),('heldout',3)]:
        for by in range(3):
            for bx in range(3):
                pool=[t for t in reps if t['candidate_background'] and t['split']==split and t['spatial_bin']==[bx,by]]
                chosen.extend(sorted(pool,key=lambda t:(-len(t['observations']),rank(f'{t["family"]}|review')))[:limit])
    for category in ['headROI','bodyROI']:
        for split in ['fit','heldout']:
            pool=[t for t in reps if t['category']==category and t['split']==split and not families[t['family']]['conflict']]
            chosen.extend(sorted(pool,key=lambda t:(-len(t['observations']),rank(f'{t["family"]}|review')))[:8])
    assert len(chosen)<=113 and len(chosen)==len(set(t['family'] for t in chosen))
    review=[];pages=[]
    for i,t in enumerate(chosen):
        strip=Image.new('RGB',(1152,128),(80,80,80))
        for ob in t['observations']:
            x,y=np.rint(ob['uv']).astype(int);strip.paste(imgs[ob['frame']].crop((x-64,y-64,x+64,y+64)),(128*FRAMES.index(ob['frame']),0))
        name=f'row_{i:03d}_family_{t["family"]}.png';strip.save(out/name);review.append(dict(**t,row=i,file=name))
        if i%6==0:page=Image.new('RGB',(1152,792),(235,235,235))
        page.paste(strip,(0,(i%6)*132))
        if i%6==5 or i==len(chosen)-1:
            name=f'review_page_{i//6:02d}.png';page.save(out/name);pages.append(name)
    save(out/'observations.json',dict(frames=FRAMES,source_sha256=p['input_video_sha256'],integer_pixel_centers=True,tracks=rows))
    save(out/'families.json',dict(radius=8,split_frozen_pregeometry=True,rows=list(families.values())))
    save(out/'visual-index.json',dict(columns=FRAMES,rows=review,pages=pages,cell_px=128,unobserved='gray placeholder',burned_text=False))
    for f in FRAMES:assert sha(image_path(f))==p['image_sha256'][str(f)]
    result=dict(updated=now(),state='complete_pending_material_review',wall_seconds=time.monotonic()-start,features=counts,pairs={f'{a}-{b}':len(m) for(a,b),m in pairs.items()},cycles=cycles,tracks=len(rows),families=len(families),review_rows=len(chosen),review_pages=len(pages),K_changed=False,cameras_estimated=False,physical_identity_verified=False,training_started=False,GPU_tasks=0)
    save(out/'result.json',result);save(O/'progress.json',dict(updated=now(),state=result['state'],pid=None,result='run/result.json'));print(json.dumps(result),flush=True)

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException as exc:
        if O.exists():save(O/'failure.json',dict(updated=now(),error=repr(exc),traceback=traceback.format_exc()));save(O/'progress.json',dict(updated=now(),state='failed',pid=None))
        raise
