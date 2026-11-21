"""One frozen 24-frame native addition. Cache anchors; no camera estimation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,itertools,time,sys,signal,traceback
import numpy as np
from PIL import Image
import match_video10_material123 as base

R=base.R; B=base.O; O=R/'video10-temporal-bridge'
read=base.read;save=base.save;sha=base.sha;rank=base.rank;Union=base.Union;now=base.now
def imgpath(f):return base.image_path(f) if f in base.FRAMES else O/'native'/f'native_frame_{f:06d}.png'
def choose_split(old_labels,signature):
    labels=set(old_labels)
    if len(labels)>1:return 'quarantine'
    if labels:return next(iter(labels))
    return 'heldout' if int(rank(signature)[:8],16)%5==0 else 'fit'
def code_checks():
    assert choose_split(['fit','heldout'],'x')=='quarantine'
    assert choose_split(['heldout','heldout'],'x')=='heldout'
    assert choose_split(['fit'],'x')=='fit'
    assert choose_split([],'x')==choose_split([],'x')
    u=Union(4);u.join(0,1);u.join(1,2);assert u.find(0)==u.find(2)!=u.find(3)

def run():
    import cv2
    code_checks();p=read(O/'plan.json');ad=read(O/'execution-admission.json')
    assert sha(Path(__file__))==ad['matcher_sha256']
    assert sha(O/'regions.json')==ad['regions_sha256']
    assert read(O/'region-review.json')['safe_for_candidate_extraction']
    frames=p['frames'];lookup={f:i for i,f in enumerate(frames)};anchors=set(p['anchor_frames'])
    region={x['frame']:x for x in read(O/'regions.json')['rows']}
    bp=read(B/'plan.json')
    for name,h in p['cached_npz_sha256'].items():assert sha(B/'run'/name)==h,name
    for f in frames:assert sha(imgpath(f))==(bp['image_sha256'][str(f)] if f in anchors else region[f]['source_sha256'])
    out=O/'run';out.mkdir();start=time.monotonic()
    def deadline(*_):raise TimeoutError('Frozen600second CPU wall budget exhausted')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(600)
    def state(s,**kw):save(O/'progress.json',dict(updated=now(),state=s,pid=os.getpid(),elapsed=time.monotonic()-start,**kw))
    cv2.setNumThreads(4);cv2.setRNGSeed(12310)
    detector=cv2.SIFT_create(nfeatures=0,contrastThreshold=.02,edgeThreshold=10,sigma=1.6)
    xy=[];desc=[];roles=[];nodes=[];offset=[0];counts=[];imgs={}
    for i,f in enumerate(frames):
        rgb=np.array(Image.open(imgpath(f)).convert('RGB'));assert rgb.shape==(1520,2704,3);imgs[f]=Image.fromarray(rgb)
        if f in anchors:
            with np.load(B/'run'/f'features_{f:06d}.npz') as d:
                uv=d['uv'].copy();ds=d['descriptor'].copy();rr=d['role'].copy();detected=len(d['raw_uv'])
        else:
            r=region[f];gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
            forbidden=np.zeros(gray.shape,np.uint8);cv2.rectangle(forbidden,r['body'][:2],r['body'][2:],1,-1)
            for b in r['exclude_background']:cv2.rectangle(forbidden,b[:2],b[2:],1,-1)
            dist=cv2.distanceTransform(1-forbidden,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            keys=detector.detect(gray,None);rawrole=np.full(len(keys),-1,np.int16);reason=np.zeros(len(keys),np.uint8)
            for z,k in enumerate(keys):
                x,y=np.rint(k.pt).astype(int);rad=int(np.ceil(5.304*k.size+2))
                if min(x,y,2703-x,1519-y)<rad:reason[z]=1;continue
                if base.inside(k.pt,r['head']):rawrole[z]=40;continue
                if base.inside(k.pt,r['body']):rawrole[z]=41;continue
                if dist[y,x]<=rad+16:reason[z]=2;continue
                rawrole[z]=min(4,int(y/304))*8+min(7,int(x/338))
            selected=[]
            for role in range(42):
                pool=np.flatnonzero(rawrole==role).tolist();pool.sort(key=lambda z:(-keys[z].response,keys[z].pt,keys[z].angle,keys[z].size))
                selected.extend(pool[:(100 if role<40 else 600 if role==40 else 1000)])
            picked,ds=detector.compute(gray,[keys[z] for z in selected]);assert ds is not None and len(picked)==len(selected)
            uv=np.array([k.pt for k in picked],np.float32);assert np.array_equal(uv,np.array([keys[z].pt for z in selected],np.float32))
            ds=np.sqrt(ds/(ds.sum(axis=1,keepdims=True)+1e-12)).astype(np.float32);rr=rawrole[selected];detected=len(keys)
            np.savez_compressed(out/f'features_{f:06d}.npz',uv=uv,descriptor=ds,role=rr,selected_indices=np.array(selected),raw_uv=np.array([k.pt for k in keys]),raw_size=np.array([k.size for k in keys]),raw_response=np.array([k.response for k in keys]),raw_angle=np.array([k.angle for k in keys]),raw_role=rawrole,exclusion_reason=reason)
        xy.append(uv);desc.append(ds);roles.append(rr);nodes.extend((i,z) for z in range(len(uv)));offset.append(len(nodes))
        counts.append(dict(frame=f,cached=f in anchors,detected=detected,selected=len(uv),background=int((rr<40).sum()),headROI=int((rr==40).sum()),bodyROI=int((rr==41).sum())))
        state('extracting_native_temporal',frames_done=i+1);print('FEATURE',f,'cached' if f in anchors else 'new',len(uv),flush=True)
    pairs={};pair_origin={};bf=cv2.BFMatcher(cv2.NORM_L2)
    for a,b in itertools.combinations(range(9),2):
        ai,bi=lookup[base.FRAMES[a]],lookup[base.FRAMES[b]]
        with np.load(B/'run'/f'pair_{a}_{b}.npz') as d:pairs[ai,bi]={int(x):int(y) for x,y in d['matches']}
        pair_origin[ai,bi]='cached_anchor'
    def direction(s,t):return {m[0].queryIdx:m[0].trainIdx for m in bf.knnMatch(s,t,k=2) if len(m)==2 and m[1].distance>1e-12 and m[0].distance<.75*m[1].distance}
    for a,b in itertools.combinations(range(len(frames)),2):
        if b-a>2:continue
        assert (a,b) not in pairs
        ab=direction(desc[a],desc[b]);ba=direction(desc[b],desc[a]);pairs[a,b]={x:y for x,y in ab.items() if ba.get(y)==x};pair_origin[a,b]='new_local'
        np.savez_compressed(out/f'pair_{a}_{b}.npz',matches=np.array(list(pairs[a,b].items()),np.int32).reshape(-1,2))
        state('matching_native_temporal',pairs_done=len(pairs));print('PAIR',a,b,len(pairs[a,b]),flush=True)
    assert len(pairs)==99 and sum(v=='new_local' for v in pair_origin.values())==63
    u=Union(len(nodes));accepted=set();cycles=0;triplets=0
    for a,b,c in itertools.combinations(range(len(frames)),3):
        if not all(k in pairs for k in [(a,b),(b,c),(a,c)]):continue
        triplets+=1
        for x,y in pairs[a,b].items():
            z=pairs[b,c].get(y)
            if z is not None and pairs[a,c].get(x)==z:
                ns=[offset[a]+x,offset[b]+y,offset[c]+z];accepted.update(ns);u.join(ns[0],ns[1]);u.join(ns[1],ns[2]);cycles+=1
    components={}
    for node in sorted(accepted):components.setdefault(u.find(node),[]).append(nodes[node])
    tracks=list(components.values());fu=Union(len(tracks));buckets={};node_track={}
    for ti,tr in enumerate(tracks):
        for a,z in tr:
            node_track[frames[a],z]=ti;pt=xy[a][z];cell=np.floor(pt/8).astype(int)
            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    for other,q in buckets.get((a,cell[0]+dx,cell[1]+dy),[]):
                        if np.linalg.norm(pt-q)<=8:fu.join(ti,other)
            buckets.setdefault((a,*cell),[]).append((ti,pt))
    oldtracks=read(B/'run/observations.json')['tracks'];oldfamilies={f['family']:f for f in read(B/'run/families.json')['rows']}
    oldreview={r['family']:r for r in read(B/'visual-review.json')['rows']};old_to_t={}
    for t in oldtracks:
        members=[node_track[(ob['frame'],ob['feature'])] for ob in t['observations']]
        old_to_t.setdefault(t['family'],set()).update(members)
    for members in old_to_t.values():
        members=sorted(members)
        for j in members[1:]:fu.join(members[0],j)
    old_to_new={old:fu.find(next(iter(ts))) for old,ts in old_to_t.items()}
    family_old={}
    for old,new in old_to_new.items():family_old.setdefault(new,[]).append(old)
    families={};rows=[]
    for ti,tr in enumerate(tracks):
        fam=fu.find(ti);rr=[int(roles[a][z]) for a,z in tr]
        cat='background' if all(v<40 for v in rr) else 'headROI' if all(v==40 for v in rr) else 'bodyROI' if all(v>=40 for v in rr) else 'mixed'
        conflict=len(set(a for a,z in tr))!=len(tr)
        row=dict(track=ti,family=fam,category=cat,duplicate_frame_conflict=conflict,observations=[dict(frame=frames[a],feature=z,uv=xy[a][z].tolist(),role=int(roles[a][z])) for a,z in tr]);rows.append(row)
        fa=families.setdefault(fam,dict(family=fam,tracks=[],categories=[],conflict=False));fa['tracks'].append(ti);fa['categories'].append(cat);fa['conflict']|=conflict
    for fam,fa in families.items():
        fa['categories']=sorted(set(fa['categories']));old=sorted(family_old.get(fam,[]));fa['old_families']=old
        signature=p['source_sha256']+'|temporal-native-family|'+','.join(f'{ob["frame"]}:{ob["feature"]}' for t in fa['tracks'] for ob in rows[t]['observations'])
        fa['split']=choose_split([oldfamilies[x]['split'] for x in old],signature)
        fa['inherited_reviews']=[dict(old_family=x,verdict=oldreview[x]['verdict'],note=oldreview[x]['note']) for x in old if x in oldreview]
        fa['inherited_contradiction']=any(x['verdict']=='contradicted' for x in fa['inherited_reviews'])
        fa['inherited_ambiguity']=any(x['verdict']=='ambiguous' for x in fa['inherited_reviews'])
        fa['candidate_background']=not fa['conflict'] and fa['categories']==['background'] and fa['split']!='quarantine' and not fa['inherited_contradiction']
        for t in fa['tracks']:rows[t]['split']=fa['split']
    for old,new in old_to_new.items():
        assert families[new]['split'] in [oldfamilies[old]['split'],'quarantine']
        if old in oldreview and oldreview[old]['verdict']=='contradicted':assert families[new]['inherited_contradiction']
    reps=[]
    for fam,fa in families.items():
        pool=[rows[t] for t in fa['tracks'] if not rows[t]['duplicate_frame_conflict']]
        if not pool:continue
        t=sorted(pool,key=lambda t:(-len(t['observations']),rank(f'{t["track"]}|representative')))[0];pt=t['observations'][len(t['observations'])//2]['uv']
        reps.append(dict(**t,spatial_bin=[min(2,int(pt[0]/(2704/3))),min(2,int(pt[1]/(1520/3)))],candidate_background=fa['candidate_background'],old_families=fa['old_families'],inherited_reviews=fa['inherited_reviews']))
    chosen=[]
    for split,limit in [('fit',6),('heldout',3)]:
        for by in range(3):
            for bx in range(3):
                pool=[t for t in reps if t['candidate_background'] and t['split']==split and t['spatial_bin']==[bx,by]]
                chosen.extend(sorted(pool,key=lambda t:(-len(t['observations']),rank(f'{t["family"]}|review')))[:limit])
    for cat in ['headROI','bodyROI']:
        for split in ['fit','heldout']:
            pool=[t for t in reps if t['category']==cat and t['split']==split and not families[t['family']]['conflict'] and not families[t['family']]['inherited_contradiction']]
            chosen.extend(sorted(pool,key=lambda t:(-len(t['observations']),rank(f'{t["family"]}|review')))[:8])
    assert len(chosen)<=113 and len(chosen)==len(set(t['family'] for t in chosen))
    review=[];pages=[];page=Image.new('RGB',(1152,1584),(235,235,235));cursor=0;page_rows=[]
    def finish_page():
        nonlocal page,cursor,page_rows
        name=f'review_page_{len(pages):02d}.png';page.crop((0,0,1152,cursor)).save(out/name)
        pages.append(dict(file=name,rows=page_rows,height=cursor));page=Image.new('RGB',(1152,1584),(235,235,235));cursor=0;page_rows=[]
    for i,t in enumerate(chosen):
        obs=t['observations'];height=((len(obs)+8)//9)*132;strip=Image.new('RGB',(1152,height),(80,80,80))
        for j,ob in enumerate(obs):
            x,y=np.rint(ob['uv']).astype(int);strip.paste(imgs[ob['frame']].crop((x-64,y-64,x+64,y+64)),(128*(j%9),132*(j//9)))
        name=f'row_{i:03d}_family_{t["family"]}.png';strip.save(out/name);review.append(dict(**t,row=i,file=name,cell_frames=[ob['frame'] for ob in obs],cell_center=[64,64]))
        if cursor+height+4>1584:finish_page()
        page.paste(strip,(0,cursor));page_rows.append(dict(row=i,y=cursor,height=height));cursor+=height+4
    if cursor:finish_page()
    save(out/'observations.json',dict(frames=frames,source_sha256=p['source_sha256'],integer_pixel_centers=True,tracks=rows))
    save(out/'families.json',dict(radius=8,split_frozen_pregeometry=True,rows=list(families.values()),old_family_map=old_to_new))
    save(out/'visual-index.json',dict(rows=review,pages=pages,cell_px=128,columns_per_wrap=9,layout='Consecutive native crops for every observation; frame order in each row metadata, no spatial warp.',burned_text=False))
    common=[]
    for t in rows:
        obs=[ob for ob in t['observations'] if ob['frame'] in anchors]
        if len(obs)>=2:common.append(dict(track=t['track'],family=t['family'],split=t['split'],anchor_observations=obs,inherited_old_families=families[t['family']]['old_families']))
    save(out/'common-anchor-endpoints.json',dict(tracks=common,interpretation='Endpoint keypoints unchanged. Identity improvements require visual evidence; all old erroneous links preserved, not silently healed.'))
    for name,h in p['cached_npz_sha256'].items():assert sha(B/'run'/name)==h
    signal.alarm(0)
    result=dict(updated=now(),state='complete_pending_physical_review',wall_seconds=time.monotonic()-start,features=counts,cached_feature_frames=9,new_feature_frames=24,cached_pairs=36,new_pairs=63,available_triplets=triplets,cycles=cycles,tracks=len(rows),families=len(families),quarantined_families=sum(f['split']=='quarantine' for f in families.values()),contradicted_families=sum(f['inherited_contradiction'] for f in families.values()),common_anchor_tracks=len(common),review_rows=len(review),review_pages=len(pages),K_changed=False,cameras_estimated=False,physical_identity_verified=False,training_started=False,GPU_tasks=0,causal_limit='Added observations plus pre-extraction ROI allocation. Not a pure temporal-density or K effect.')
    save(out/'result.json',result);save(O/'progress.json',dict(updated=now(),state=result['state'],pid=None,result='run/result.json'));print(json.dumps(result),flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='check':code_checks();print('FAMILY_SPLIT_CONTROLS_PASS')
    else:
        try:run()
        except BaseException as exc:
            save(O/'failure.json',dict(updated=now(),error=repr(exc),traceback=traceback.format_exc()));save(O/'progress.json',dict(updated=now(),state='failed',pid=None));raise
