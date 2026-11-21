"""N12d: one selection-only native RootSIFT coverage comparison."""
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
from match_white_scene122 import Union
from audit_white_structure_allocation122 import inside
I=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video-update-20261027')
W=I/'clip03-native-matching';A=I/'clip03-structure-allocation';M=I/'clip03-scene-support'
O=I/'clip03-grid-coverage/array-cache-repair';P=I/'clip03-grid-coverage/array-cache-repair-plan.json'
FRAMES=[184,200,216,232,248,264,280]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rank(s):return hashlib.sha256(str(s).encode()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()

def fair_select(xy,response,size,angle,roles,old_ids):
    chosen=[]; bands=[(0,901),(901,1802),(1802,2704)]
    for role,(lo,hi) in enumerate(bands):
        cells=[[] for _ in range(32)]
        for j in np.flatnonzero(roles==role):
            x,y=xy[j];cx=min(7,int(x*8/1520));cy=min(3,int((y-lo)*4/(hi-lo)))
            cells[cy*8+cx].append(int(j))
        for c in cells:c.sort(key=lambda j:(-float(response[j]),float(xy[j,1]),float(xy[j,0]),float(size[j]),float(angle[j])))
        selected=[];depth=0
        while len(selected)<2000:
            added=0
            for c in cells:
                if depth<len(c) and len(selected)<2000:selected.append(c[depth]);added+=1
            if not added:break
            depth+=1
        assert len(selected)==min(2000,int(np.sum(roles==role)))
        chosen.extend(selected)
    chosen.extend(int(j) for j in old_ids if roles[j]>=3)
    assert len(chosen)==len(set(chosen))
    return np.array(chosen,np.int32)

def split_policy(old_splits,old_isolated,categories,key):
    split=next(iter(old_splits)) if len(old_splits)==1 else('isolated' if len(old_splits)>1 else('heldout' if int(rank(key)[:8],16)%5==0 else 'fit'))
    return split, bool(old_isolated or len(old_splits)>1 or categories!={'background'})

def controls():
    # Highest-response dominance must not consume every cell's budget.
    xy=np.array([[20+(j%2)*800,10+(j//2)%800] for j in range(5000)],np.float32)
    roles=np.zeros(5000,np.int16);response=np.where(xy[:,0]<100,10.,1.)
    ids=fair_select(xy,response,np.ones(5000),np.zeros(5000),roles,np.arange(5000))
    assert len(ids)==2000 and np.sum(xy[ids,0]>100)>=900
    assert split_policy({'fit','heldout'},False,{'background'},'x')==('isolated',True)
    assert split_policy({'heldout'},False,{'background'},'x')==('heldout',False)
    assert split_policy({'fit'},True,{'background'},'x')==('fit',True)
    assert split_policy({'fit'},False,{'background','person'},'x')==('fit',True)
    # A new bridge joins two old families even if endpoints never touch each other.
    u=Union(3);u.join(0,2);u.join(1,2);assert u.find(0)==u.find(1)
    assert split_policy({'fit','heldout'},False,{'background'},'bridge')[1]
    return dict(checks=6,passed=True,scope='Quota coverage, inheritedheldout, mixedsplit bridge and material isolation; notphysicalcalibration')

def register():
    assert not O.exists() and not P.exists()
    assert read(A/'visual-review.json')['combined_with_N12b']==dict(fit=3,heldout=0)
    files=[W/n for n in ['observations.json','families.json','result.json','visual-review.json']]
    files += [I/'clip03-native-matching-plan.json',I/'clip03-structure-allocation-plan.json',A/'allocation.json',A/'visual-review.json']
    files += [I/'clip03-grid-coverage-plan.json',I/'clip03-grid-coverage/failure.json',Path(__file__).parent/'compare_white_coverage122.py']
    files += [W/f'features_{f:06d}.npz' for f in FRAMES]+[M/f'frame_{f:06d}.png' for f in FRAMES]
    save(P,dict(stage='N12d',registered=now(),source_sha256=sha(Path(__file__)),controls=controls(),implementation_repair='First attempt was interrupted at repeatedNPZdecompression in equalitycheck;3.364GBread,84.687active seconds. Cacheeacharrayonce. Samefixedselection/matching/familypolicy;no firstattemptmatching result existed. Remaining95seconds staysbelow original180total. Originalscript/plan/failurepreserved.',
      hypothesis='N12c verified coarse-band top-response quota loses built-structure coverage. A finer uniform quota should retain more native static-object candidates at equal descriptor budget. It may fail to improve actual physical correspondences; countincrease alone is notsuccess.',
      causal_variable='Only background allocation: same three2000point bands subdivided8columns x4rows each; round-robin percell sorted bysameSIFTresponse/tiekeys until2000. Same 600headROI/1400otherperson exact selected IDs. No manuallytargeted region weighting.',
      fixed='Same7nativeimages, pixelgraybytes, allrawSIFTpositions/sizes/responses, contrast.02 edge10 sigma1.6, eligibility/personsupport/border rules,8000descriptors,RootSIFTmutualratio.75,pair steps1/2,threeviewcycle,8pxfamily radius,conditional4000K. NoK/pose/geometry training.',
      detector_recovery='N12b omitted angle/octave forunselecteddetections. Recreate same detector once perframe solely to recover metadata, require bit-exact allstoredrawxy/size/response and bit-exact8000baselineRootSIFTdescriptors. Stopon mismatch; do not retune. Saveallkeypoint metadata. Baseline selection/matching results are not rerun or overwritten.',
      split='Combine newcycletracks and originaltracks by sameframe<=8px, seed unions byoldwholefamily. Inherit every oldfamily split. A newcomponent joiningFITandheldout oroldisolatedfamily isquarantined, nevertransferheldouttoFIT. Newonlycomponents getfixedSHA80/20 beforegeometry. Preserveallobservations.',
      review='Same24semanticsearchregions onlyforevaluation, notselection. Per8objectcategories up to4FIT/2heldout representatives, longesttrackthenfixedSHA. Groundlowerband per3xbin up to5FIT/3heldout excludingselectedfamilies. Separately up to8FIT/8heldout headROI andotherperson. Allnative128pxobservations; optional16native256pxambiguouscontexts. Maximum104rows.',
      geometry_gate='Inherited20staticFIT/8heldout,3imagebins,6FIT/3heldoutnon-ground,12sharedFITinitialadjacentpair and8reviewedFIT2D3D later. Materialreviewmustpass beforea separatelyregistered freshfixedK geometry run. No thresholdorlabel relaxation.',
      pass_evidence='More independentlysplit, visuallycredible built-surface correspondences and coverage; reportallattrition/mixedsplit quarantines. Wholevideo/allface<1px acceptance remainsseparate.',
      budget=dict(execution_runs=1,frames=7,CPU_seconds=95,original_total_budget_seconds=180,prior_active_seconds=84.687,GPU=0,camera_solves=0,training=0,context_rows_max=16),
      stop='Inputidentity/control/budget failure stops. Insufficientreviewedstatic support means no camera ortraining; no secondgridsetting,thresholdsweep ornewweights.',
      K_integer_centers=read(I/'clip03-native-matching-plan.json')['K_integer_centers'],input_hashes={str(f):sha(f) for f in files}))
    print('N12D_REGISTERED',flush=True)

def run():
    assert not O.exists();p=read(P);assert p['source_sha256']==sha(Path(__file__))
    assert controls()['passed']
    for f,h in p['input_hashes'].items():assert sha(Path(f))==h
    O.mkdir();started=time.monotonic();cv2.setNumThreads(4);cv2.setRNGSeed(12212)
    def progress(state,**kw):
        assert time.monotonic()-started<95,'N12d CPU wall budget exceeded'
        save(O/'progress.json',dict(state=state,pid=os.getpid(),updated=now(),elapsed_seconds=time.monotonic()-started,**kw))
    base=read(W/'observations.json');br=read(W/'result.json');bfams={x['family']:x for x in read(W/'families.json')['rows']}
    sift=cv2.SIFT_create(nfeatures=0,contrastThreshold=.02,edgeThreshold=10,sigma=1.6)
    xy=[];desc=[];role=[];dids=[];nodes=[];offset=[0];counts=[]
    for fi,f in enumerate(FRAMES):
        progress('recovering_identical_detector_metadata',frame=f)
        gray=cv2.cvtColor(cv2.imread(str(M/f'frame_{f:06d}.png')),cv2.COLOR_BGR2GRAY)
        assert hashlib.sha256(gray.tobytes()).hexdigest()==br['features'][fi]['gray_sha256']
        archive=np.load(W/f'features_{f:06d}.npz');d={name:archive[name] for name in archive.files};archive.close();kp=sift.detect(gray,None)
        rawxy=np.array([k.pt for k in kp],np.float32);size=np.array([k.size for k in kp],np.float32);response=np.array([k.response for k in kp],np.float32)
        for a,b in [(rawxy,d['detector_uv']),(size,d['detector_size']),(response,d['detector_response'])]:assert np.array_equal(a,b),'Raw detectoridentity changed'
        angle=np.array([k.angle for k in kp],np.float32);octave=np.array([k.octave for k in kp],np.int32)
        oldids=d['selected_detector_indices'];_,old_desc=sift.compute(gray,[kp[j] for j in oldids]);old_desc=np.sqrt(old_desc/(old_desc.sum(axis=1,keepdims=True)+1e-12)).astype(np.float32)
        assert np.array_equal(old_desc,d['descriptor']),'Baseline descriptoridentity changed'
        ids=fair_select(rawxy,response,size,angle,d['detector_role'],oldids);assert len(ids)==len(oldids)==8000
        selected,newdesc=sift.compute(gray,[kp[j] for j in ids]);pts=np.array([k.pt for k in selected],np.float32)
        assert np.array_equal(pts,rawxy[ids]);newdesc=np.sqrt(newdesc/(newdesc.sum(axis=1,keepdims=True)+1e-12)).astype(np.float32)
        rr=d['detector_role'][ids];assert np.array_equal(ids[rr>=3],oldids[d['role']>=3])
        oldmap={int(j):n for n,j in enumerate(oldids)};common=[(n,oldmap[int(j)]) for n,j in enumerate(ids) if int(j) in oldmap]
        assert all(np.array_equal(newdesc[n],d['descriptor'][m]) for n,m in common)
        np.savez_compressed(O/f'features_{f:06d}.npz',uv=pts,descriptor=newdesc,role=rr,selected_detector_indices=ids,
          detector_uv=rawxy,detector_size=size,detector_response=response,detector_angle=angle,detector_octave=octave,
          detector_class_id=np.array([k.class_id for k in kp],np.int32),detector_role=d['detector_role'],exclusion_reason=d['exclusion_reason'])
        xy.append(pts);desc.append(newdesc);role.append(rr);dids.append(ids);nodes.extend((fi,j) for j in range(len(ids)));offset.append(len(nodes))
        counts.append(dict(frame=f,all_detector_identity_exact=True,all_baseline_descriptors_exact=True,shared_descriptors_exact=True,person_IDs_unchanged=True,selected=8000,retained_old_ids=len(common),new_ids=8000-len(common)))
        print('N12D_SELECTED',f,counts[-1]['new_ids'],flush=True)
    progress('matching_same_parameters');pairs={};matcher=cv2.BFMatcher(cv2.NORM_L2)
    for i in range(7):
        for step in [1,2]:
            j=i+step
            if j>=7:continue
            forward=matcher.knnMatch(desc[i],desc[j],k=2);back=matcher.knnMatch(desc[j],desc[i],k=2)
            a={v[0].queryIdx:v[0].trainIdx for v in forward if len(v)==2 and v[0].distance<.75*v[1].distance}
            b={v[0].queryIdx:v[0].trainIdx for v in back if len(v)==2 and v[0].distance<.75*v[1].distance}
            pairs[i,j]={x:y for x,y in a.items() if b.get(y)==x};progress('matching_same_parameters',pairs=len(pairs))
    u=Union(len(nodes));accepted=set();cycles=0
    for i in range(5):
        for a,b in pairs[i,i+1].items():
            c=pairs[i+1,i+2].get(b)
            if c is not None and pairs[i,i+2].get(a)==c:
                ns=[offset[i]+a,offset[i+1]+b,offset[i+2]+c];accepted.update(ns);u.join(ns[0],ns[1]);u.join(ns[1],ns[2]);cycles+=1
    raw={}
    for node in sorted(accepted):raw.setdefault(u.find(node),[]).append(nodes[node])
    tracks=[]
    for ti,ns in enumerate(raw.values()):
        roles=[int(role[i][j]) for i,j in ns];cat='background' if all(r<3 for r in roles) else('headROI' if all(r==3 for r in roles) else('person' if all(r>=3 for r in roles) else'mixed'))
        tracks.append(dict(track=ti,category=cat,duplicate_frame_conflict=len(set(i for i,j in ns))!=len(ns),observations=[dict(frame=FRAMES[i],feature=j,detector_id=int(dids[i][j]),uv=xy[i][j].tolist(),role=int(role[i][j])) for i,j in ns]))
    progress('frozen_cross_variant_family_split')
    old=base['tracks'];joint=old+tracks;fu=Union(len(joint));buckets={};oldroots={}
    for ti,t in enumerate(old):
        if t['family'] in oldroots:fu.join(ti,oldroots[t['family']])
        else:oldroots[t['family']]=ti
    for ti,t in enumerate(joint):
        for o in t['observations']:
            uv=np.array(o['uv']);cell=np.floor(uv/8).astype(int);f=o['frame']
            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    for tj,other in buckets.get((f,cell[0]+dx,cell[1]+dy),[]):
                        if np.linalg.norm(uv-other)<=8:fu.join(ti,tj)
            buckets.setdefault((f,*cell),[]).append((ti,uv))
    groups={}
    for ti,t in enumerate(joint):groups.setdefault(fu.find(ti),[]).append(ti)
    family_records=[]
    for root,tis in groups.items():
        newids=[ti-len(old) for ti in tis if ti>=len(old)]
        if not newids:continue
        olds={old[ti]['family'] for ti in tis if ti<len(old)};splits={bfams[x]['split'] for x in olds}
        categories={joint[ti]['category'] for ti in tis};old_iso=any(bfams[x]['isolated_from_camera'] for x in olds)
        key=base['source_sha256']+'|N12d|'+str(min((o['frame'],o['detector_id']) for j in newids for o in tracks[j]['observations']))
        split,iso=split_policy(splits,old_iso,categories,key);conflict=any(joint[ti]['duplicate_frame_conflict'] for ti in tis);iso|=conflict
        rec=dict(family=root,tracks=newids,split=split,old_families=sorted(olds),old_splits=sorted(splits),old_isolated=old_iso,isolated_from_camera=iso,mixed_old_splits=len(splits)>1,categories=sorted(categories),duplicate_frame_conflict=conflict)
        family_records.append(rec)
        for ti in newids:tracks[ti].update(family=root,split=split,camera_candidate=not iso)
    assert all(t['split']!='fit' for t in tracks if len(next(x for x in family_records if x['family']==t['family'])['old_splits'])>1)
    save(O/'observations.json',dict(source_sha256=base['source_sha256'],frames=FRAMES,pixel_convention='integer_pixel_centers',tracks=tracks,raw_pairs=[dict(frames=[FRAMES[i],FRAMES[j]],matches=list(v.items())) for(i,j),v in pairs.items()]))
    save(O/'families.json',dict(radius_px=8,baseline_labels_preserved=True,cross_variant_union=True,rows=family_records))
    regions=read(I/'clip03-structure-allocation-plan.json')['regions'];node_tracks={};pairnodes=set();allocation=[];touch={}
    for t in tracks:
        for o in t['observations']:node_tracks.setdefault((o['frame'],o['feature']),set()).add(t['track'])
    for (i,j),pairs_ in pairs.items():
        for a,b in pairs_.items():pairnodes.update([(FRAMES[i],a),(FRAMES[j],b)])
    for ri,region in enumerate(regions):
        f=region['frame'];fi=FRAMES.index(f);ids=np.flatnonzero(inside(xy[fi],region['native_polygon']));ts=set()
        for j in ids:ts.update(node_tracks.get((f,int(j)),set()))
        eligible=[j for j in ts if tracks[j]['camera_candidate']]
        for ti in eligible:touch.setdefault(ti,[]).append(ri)
        allocation.append(dict(region=ri,frame=f,category=region['category'],selected_background=int(np.sum(role[fi][ids]<3)),paired_selected=sum((f,int(j)) in pairnodes for j in ids),cycles_before_isolation=len(ts),camera_tracks=len(eligible),camera_families=len({tracks[j]['family'] for j in eligible})))
    candidates=[]
    for fam in family_records:
        pool=[tracks[j] for j in fam['tracks'] if tracks[j]['camera_candidate'] and j in touch]
        if not pool:continue
        pool.sort(key=lambda t:(-len(t['observations']),rank(f'{t["track"]}|N12d_rep')));t=pool[0]
        cats={regions[ri]['category'] for t0 in pool for ri in touch[t0['track']]}
        cat=sorted(cats,key=lambda c:(-sum(regions[ri]['category']==c for t0 in pool for ri in touch[t0['track']]),rank(c)))[0]
        candidates.append(dict(**t,semantic_category=cat,region_ids=sorted({ri for t0 in pool for ri in touch[t0['track']]})))
    chosen=[];categories=['lamp_fixture','pole','hydrant','steps','near_building','distant_roof','distant_facade','fence_post']
    for cat in categories:
        for split,limit in [('fit',4),('heldout',2)]:
            pool=[t for t in candidates if t['semantic_category']==cat and t['split']==split];pool.sort(key=lambda t:(-len(t['observations']),rank(f'{t["family"]}|N12d_review')));chosen.extend(pool[:limit])
    reps=[]
    for fam in family_records:
        if fam['split']=='isolated' or fam['duplicate_frame_conflict']:continue
        pool=[tracks[j] for j in fam['tracks']];pool.sort(key=lambda t:(-len(t['observations']),rank(f'{t["track"]}|N12d_rep')));t=pool[0]
        mid=t['observations'][len(t['observations'])//2]['uv'];reps.append(dict(**t,spatial_bin=[min(2,int(mid[0]/(1520/3))),min(2,int(mid[1]/(2704/3)))]))
    for bx in range(3):
        for split,limit in [('fit',5),('heldout',3)]:
            used={t['family'] for t in chosen};pool=[t for t in reps if t['camera_candidate'] and t['split']==split and t['spatial_bin']==[bx,2] and t['family'] not in used]
            pool.sort(key=lambda t:(-len(t['observations']),rank(f'{t["family"]}|N12d_ground')));chosen.extend([dict(**t,semantic_category='ground_search') for t in pool[:limit]])
    for cat in ['headROI','person']:
        for split in ['fit','heldout']:
            used={t['family'] for t in chosen};pool=[t for t in reps if t['category']==cat and t['split']==split and t['family'] not in used]
            pool.sort(key=lambda t:(-len(t['observations']),rank(f'{t["family"]}|N12d_person')));chosen.extend([dict(**t,semantic_category=cat) for t in pool[:8]])
    assert len(chosen)<=104 and len(chosen)==len({t['family'] for t in chosen})
    images={f:Image.open(M/f'frame_{f:06d}.png').convert('RGB') for f in FRAMES};review=[];pages=[]
    for i,t in enumerate(chosen):
        strip=Image.new('RGB',(128*7,128),(80,80,80))
        for o in t['observations']:
            x,y=np.rint(o['uv']).astype(int);strip.paste(images[o['frame']].crop((x-64,y-64,x+64,y+64)),(128*FRAMES.index(o['frame']),0))
        name=f'review_{i:03d}_family_{t["family"]}.png';strip.save(O/name);review.append(dict(**t,row=i,file=name))
        if i%6==0:page=Image.new('RGB',(896,792),(235,235,235))
        page.paste(strip,(0,(i%6)*132))
        if i%6==5 or i==len(chosen)-1:name=f'review_page_{i//6:02d}.png';page.save(O/name);pages.append(name)
    save(O/'visual-index.json',dict(rows=review,pages=pages,columns=FRAMES,cell_size=128,annotations_in_image=False,selection='preregistered structure/ground/person and splitstrata; noresiduals'))
    save(O/'allocation.json',dict(regions=allocation,candidate_families=candidates,baseline_region_allocation=str(A/'allocation.json')))
    for f,h in p['input_hashes'].items():assert sha(Path(f))==h
    progress('complete_pending_actual_material_review')
    result=dict(stage='N12d',state='complete_pending_actual_material_review',updated=now(),elapsed_seconds=time.monotonic()-started,controls=controls(),features=counts,cycles=cycles,tracks=len(tracks),families=len(family_records),
      old_split_mixed_families=sum(f['mixed_old_splits'] for f in family_records),isolated_camera_families=sum(f['isolated_from_camera'] for f in family_records),
      camera_families_by_split={s:sum(not f['isolated_from_camera'] and f['split']==s for f in family_records) for s in ['fit','heldout']},
      review_rows=len(review),review_by_search={c:sum(t['semantic_category']==c for t in chosen) for c in categories+['ground_search','headROI','person']},
      K_changed=False,new_cameras=0,training_started=False,physical_accuracy_passed=False,
      hashes={n:sha(O/n) for n in ['observations.json','families.json','visual-index.json','allocation.json']})
    save(O/'result.json',result);save(O/'progress.json',dict(state=result['state'],pid=None,updated=now()));print(json.dumps(result),flush=True)

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException as exc:
        if O.exists():save(O/'failure.json',dict(updated=now(),error=repr(exc),traceback=traceback.format_exc()));save(O/'progress.json',dict(state='failed',pid=None,updated=now()))
        raise
