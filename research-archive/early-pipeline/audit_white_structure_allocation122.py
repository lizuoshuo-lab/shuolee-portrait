"""N12c: read-only semantic allocation audit of frozen N12b observations."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,sys,time,os
import numpy as np
from PIL import Image
I=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video-update-20261027')
W=I/'clip03-native-matching';M=I/'clip03-scene-support';P=I/'clip03-structure-allocation-plan.json';O=I/'clip03-structure-allocation'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def rank(x):return hashlib.sha256(str(x).encode()).hexdigest()
def box(x0,y0,x1,y1):return [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]

def regions():
    # Regions annotated from the actually viewed full frames, displayed1151x2048.
    # Converted once to native1520x2704. They are search regions, NOT certified materials.
    raw=[
      (184,'lamp_fixture','upper',box(465,198,512,255)),
      (184,'lamp_fixture','lower',box(507,304,559,358)),
      (184,'pole','visible',box(491,360,510,566)),
      (184,'hydrant','1',box(25,850,78,928)),
      (184,'hydrant','2',box(101,850,148,928)),
      (184,'hydrant','3',box(174,852,219,930)),
      (184,'hydrant','4',box(244,852,289,930)),
      (184,'hydrant','5',box(314,848,360,930)),
      (184,'steps','visible',[[800,837],[992,822],[995,847],[1031,847],[1032,876],[1078,878],[1078,910],[830,935]]),
      (184,'near_building','column',box(565,0,603,567)),
      (184,'distant_facade','right',box(1101,400,1151,595)),
      (232,'lamp_fixture','upper',box(120,179,168,229)),
      (232,'lamp_fixture','lower',box(169,296,230,353)),
      (232,'pole','visible',[[170,357],[184,357],[214,961],[197,961]]),
      (232,'steps','visible',[[180,866],[418,846],[417,891],[388,939],[357,985],[185,951]]),
      (232,'near_building','column',box(174,0,213,276)),
      (232,'distant_roof','red_roof',[[746,284],[1151,147],[1151,259],[812,341]]),
      (232,'distant_facade','main',[[811,342],[1122,255],[1093,696],[808,700]]),
      (280,'steps','visible',[[0,869],[341,845],[342,871],[381,869],[381,926],[0,1009]]),
      (280,'distant_roof','red_roof',[[378,298],[709,224],[726,294],[416,348]]),
      (280,'distant_facade','main',[[416,348],[699,291],[701,604],[425,625]]),
      (280,'distant_facade','red_vertical',box(701,296,726,685)),
      (280,'fence_post','middle',box(775,695,799,921)),
      (280,'fence_post','right',box(1062,683,1087,943))]
    scale=np.array([1520/1151,2704/2048])
    return [dict(frame=f,category=c,object_hint=o,display_polygon=xy,native_polygon=(np.asarray(xy)*scale).tolist()) for f,c,o,xy in raw]

def inside(uv,poly):
    x,y=uv[:,0],uv[:,1];p=np.asarray(poly);ans=np.zeros(len(uv),bool)
    for a,b in zip(p,np.roll(p,-1,axis=0)):
        if a[1]==b[1]:continue
        ans^=((a[1]>y)!=(b[1]>y))&(x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])
    return ans

def register():
    assert not P.exists() and not O.exists()
    assert read(W/'visual-review.json')['gate']['coarse_camera_passed'] is False
    frames=read(W/'observations.json')['frames']
    files=[W/n for n in ['result.json','observations.json','families.json','visual-index.json','visual-review.json']]
    files += [W/f'features_{f:06d}.npz' for f in frames]+[M/f'frame_{f:06d}.png' for f in frames]
    save(P,dict(stage='N12c',registered=now(),source_sha256=sha(Path(__file__)),
      hypothesis='N12b spatial/longest-track selection may miss rigid built-structure observations already in the frozen pool. Audit detector->eligibility->quota->pair->cycle->review attrition in visually grounded static-object search regions, before anynew camera solve.',
      cause_variable='Semantic allocation/review selection on existing observations only; K,detector,descriptors,matching and frozenfamily split unchanged.',
      actual_full_frames_reviewed_before_query=[184,232,280],regions=regions(),
      region_uncertainty='Coarse manual search polygons only. Boundaries mayinclude foliage, glass or occlusion. Inclusion nevercertifies staticity or physicalpoint identity; all selected families require actualnativecontext review. No use ofresiduals.',
      selection='Unreviewed background-candidate wholefamilies touching anyregion. Chooseone representative inside a region bymostobservations thenfixedSHA; assign one category bylargest numberofanchorobservations, fixedSHA tie. Per8categories chooseup to4FIT/2heldout, longesttrack thenfixedSHA. Retainallfamilies andexclusions. No changingFIT/heldout.',
      budget=dict(cpu_seconds=90,new_detection=0,new_descriptors=0,new_matching=0,camera_solve=0,training=0,additional_family_review_max=48,optional256px_context_max=16),
      stop='Low numbers distinguish detection/selection/matching/semantic gap; do not relax this budget or rerunfeatures. Any freshgeometry needs separatelyregistered supportbasedplan; no implicit permission from counts.',
      input_hashes={str(f):sha(f) for f in files}))
    print('N12C_PLAN_REGISTERED')

def run():
    assert not O.exists();p=read(P);assert sha(Path(__file__))==p['source_sha256']
    for f,h in p['input_hashes'].items():assert sha(Path(f))==h
    O.mkdir();started=time.monotonic();obs=read(W/'observations.json');idx=read(W/'visual-index.json');frames=obs['frames']
    save(O/'progress.json',dict(state='readonly_allocation_audit',pid=os.getpid(),updated=now()))
    previous={t['family'] for t in idx['rows']};tracks=obs['tracks'];features={f:np.load(W/f'features_{f:06d}.npz') for f in frames}
    node_tracks={}
    for t in tracks:
        for o in t['observations']:node_tracks.setdefault((o['frame'],o['feature']),set()).add(t['track'])
    pairs=set()
    for pair in obs['raw_pairs']:
        for a,b in pair['matches']:pairs.update([(pair['frames'][0],a),(pair['frames'][1],b)])
    allocation=[];touch={};region_panels=[]
    for ri,region in enumerate(p['regions']):
        f=region['frame'];d=features[f];poly=region['native_polygon']
        raw=np.flatnonzero(inside(d['detector_uv'],poly));sel=np.flatnonzero(inside(d['uv'],poly));ts=set()
        for j in sel:ts.update(node_tracks.get((f,int(j)),set()))
        eligible=[t for t in ts if tracks[t]['camera_candidate']]
        fams={tracks[t]['family'] for t in eligible}
        for t in eligible:touch.setdefault(t,[]).append(ri)
        allocation.append(dict(region=ri,frame=f,category=region['category'],object_hint=region['object_hint'],
         detected_centers=len(raw),eligible_before_quota=int(np.sum((d['detector_role'][raw]>=0)&(d['detector_role'][raw]<3))),
         excluded_descriptor_border=int(np.sum(d['exclusion_reason'][raw]==1)),excluded_person_support=int(np.sum(d['exclusion_reason'][raw]==2)),
         selected_background_centers=int(np.sum(d['role'][sel]<3)),selected_person_centers=int(np.sum(d['role'][sel]>=3)),
         paired_selected_centers=sum((f,int(j)) in pairs for j in sel),cycled_tracks=len(eligible),families=len(fams),
         N12b_previously_reviewed_families=len(fams&previous),unreviewed_families=sorted(fams-previous)))
        # Region context is native and rectangular, so ambiguity at polygon edges remains visible.
        poly=np.array(poly);lo=np.floor(poly.min(0)).astype(int);hi=np.ceil(poly.max(0)).astype(int)
        name=f'region_{ri:02d}_{f}_{region["category"]}.png'
        Image.open(M/f'frame_{f:06d}.png').convert('RGB').crop((*lo,*hi)).save(O/name)
        region_panels.append(dict(region=ri,file=name,native_crop=[*lo.tolist(),*hi.tolist()]))
    families={}
    for ti,rids in touch.items():
        t=tracks[ti]
        if t['family'] in previous:continue
        families.setdefault(t['family'],[]).append(ti)
    candidates=[]
    for family,tis in families.items():
        tis.sort(key=lambda ti:(-len(tracks[ti]['observations']),rank(f'{ti}|N12c_rep')));t=tracks[tis[0]]
        categories={p['regions'][ri]['category'] for ti in tis for ri in touch[ti]}
        category=sorted(categories,key=lambda c:(-sum(p['regions'][ri]['category']==c for ti in tis for ri in touch[ti]),rank(c)))[0]
        candidates.append(dict(**t,semantic_category=category,region_ids=sorted({ri for ti in tis for ri in touch[ti]}),all_touching_categories=sorted(categories)))
    chosen=[]
    categories=['lamp_fixture','pole','hydrant','steps','near_building','distant_roof','distant_facade','fence_post']
    for cat in categories:
        for split,limit in [('fit',4),('heldout',2)]:
            pool=[t for t in candidates if t['semantic_category']==cat and t['split']==split]
            pool.sort(key=lambda t:(-len(t['observations']),rank(f'{t["family"]}|N12c_review')));chosen.extend(pool[:limit])
    assert len(chosen)<=48 and len(chosen)==len({t['family'] for t in chosen})
    images={f:Image.open(M/f'frame_{f:06d}.png').convert('RGB') for f in frames};review=[];pages=[]
    for i,t in enumerate(chosen):
        strip=Image.new('RGB',(128*7,128),(80,80,80))
        for o in t['observations']:
            x,y=np.rint(o['uv']).astype(int);col=frames.index(o['frame']);strip.paste(images[o['frame']].crop((x-64,y-64,x+64,y+64)),(128*col,0))
        name=f'review_{i:03d}_family_{t["family"]}.png';strip.save(O/name)
        review.append(dict(**t,row=i,file=name))
        if i%6==0:page=Image.new('RGB',(896,792),(235,235,235))
        page.paste(strip,(0,(i%6)*132))
        if i%6==5 or i==len(chosen)-1:
            name=f'review_page_{i//6:02d}.png';page.save(O/name);pages.append(name)
    save(O/'allocation.json',dict(regions=allocation,region_images=region_panels,candidate_families=candidates,prior_family_review_count=len(previous),
       note='Overlappingregions maycountsameobservationmorethanonce; no crossregion sum presented asunique count. Region labels are searchlabels only.'))
    save(O/'visual-index.json',dict(rows=review,pages=pages,columns=frames,cell_size=128,grey='unobserved slot',annotations_in_image=False))
    for f,h in p['input_hashes'].items():assert sha(Path(f))==h
    elapsed=time.monotonic()-started;assert elapsed<90
    r=dict(stage='N12c',state='readonly_allocation_complete_pending_material_review',updated=now(),elapsed_seconds=elapsed,
      unreviewed_touched_families=len(candidates),selected_review_families=len(chosen),by_category={cat:{split:sum(t['semantic_category']==cat and t['split']==split for t in candidates) for split in ['fit','heldout']} for cat in categories},
      K_changed=False,features_recomputed=False,matches_recomputed=False,new_cameras=0,training_started=False,
      hashes={n:sha(O/n) for n in ['allocation.json','visual-index.json']})
    save(O/'result.json',r);save(O/'progress.json',dict(state=r['state'],pid=None,updated=now()));print(json.dumps(r,ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
