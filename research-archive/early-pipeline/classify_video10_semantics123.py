"""Classify cached observations once after actual mask review. No matching or geometry."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,collections,json
import numpy as np
from PIL import Image
import match_video10_material123 as base
import match_video10_temporal_bridge123 as bridge
import cv2
R=base.R;B=base.O;O=bridge.O;N=R/'video10-semantic-role'
read=base.read;save=base.save;sha=base.sha;now=base.now

def run():
    assert not (N/'classification-result.json').exists(),'Already completed: do not repeat'
    a=read(N/'execution-admission.json');r=read(N/'result.json');review=read(N/'mask-visual-review.json')
    assert review['actual_saved_visual_review'] and review['frames_reviewed']==33
    assert all(sha(Path(f))==h for f,h in a['frozen_hashes'].items())
    rejected=set(review['rejected_frames']);assert rejected=={617}
    semantics={};details=[];start=time.monotonic();cv2.setNumThreads(4)
    # Independent pixel-distance check establishes the support guard convention.
    toy=np.zeros((31,31),np.uint8);toy[10:15,13:18]=1
    dt=cv2.distanceTransform(1-toy,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    fy,fx=np.where(toy)
    for x,y in [(0,0),(13,9),(20,20),(14,13)]:
        expected=np.sqrt(((fx-x)**2+(fy-y)**2).min());assert abs(float(dt[y,x])-expected)<1e-5
    for rr in r['records']:
        f=rr['frame'];region=a['regions'][str(f)];mask=np.asarray(Image.open(rr['mask']).convert('L'))>127
        assert sha(Path(rr['mask']))==rr['mask_sha256']
        path=(B if f in base.FRAMES else O)/'run'/f'features_{f:06d}.npz'
        with np.load(path) as d:
            xy=d['uv'].copy();roles=d['role'].copy();size=d['raw_size'][d['selected_indices']].copy()
            assert np.array_equal(xy,d['raw_uv'][d['selected_indices']].astype(np.float32))
        forbid=mask.astype(np.uint8)
        for box in region['exclude_background']:cv2.rectangle(forbid,box[:2],box[2:],1,-1)
        dist=cv2.distanceTransform(1-forbid,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        pts=np.rint(xy).astype(int);guard=np.ceil(5.304*size+2).astype(int)+16
        sem=[];clearance=[]
        for z,(x,y) in enumerate(pts):
            if f in rejected:value='unreliable_mask'
            elif mask[y,x]:value='person_mask'
            elif dist[y,x]<=guard[z]:value='uncertain_support'
            else:value='background_candidate'
            sem.append(value);clearance.append(float(dist[y,x]))
        semantics[f]=sem
        details.append(dict(frame=f,feature_file=str(path),feature_sha256=sha(path),mask_sha256=rr['mask_sha256'],
            labels=sem,support_radius_plus16=guard.tolist(),clearance_native_px=clearance,counts=dict(collections.Counter(sem)),
            old_body_head_now_background=int(sum(roles[z]>=40 and sem[z]=='background_candidate' for z in range(len(sem))))))
        print('SEMANTICS',f,dict(collections.Counter(sem)),flush=True)
    obs=read(O/'run/observations.json');oldfam=read(O/'run/families.json');tracks=[]
    for t in obs['tracks']:
        labels=[semantics[o['frame']][o['feature']] for o in t['observations']]
        tracks.append(dict(track=t['track'],family=t['family'],split=t['split'],original_category=t['category'],
          observations=[dict(**o,secondary_semantic_label=v) for o,v in zip(t['observations'],labels)],
          secondary_category='background_candidate' if all(v=='background_candidate' for v in labels) else 'person_mask' if all(v=='person_mask' for v in labels) else 'mixed_or_uncertain'))
    families=[]
    for fa in oldfam['rows']:
        ts=[tracks[i] for i in fa['tracks']];allbg=all(t['secondary_category']=='background_candidate' for t in ts)
        eligible=allbg and not fa['conflict'] and not fa['inherited_contradiction'] and fa['split']!='quarantine'
        families.append(dict(**fa,secondary_candidate_background=eligible,secondary_categories=sorted(set(t['secondary_category'] for t in ts))))
    assert len(tracks)==len(obs['tracks']) and len(families)==len(oldfam['rows'])
    for old,new in zip(obs['tracks'],tracks):
        assert old['track']==new['track'] and old['family']==new['family'] and old['split']==new['split']
        assert old['observations']==[{k:v for k,v in o.items() if k!='secondary_semantic_label'} for o in new['observations']]
    assert all(sha(Path(f))==h for f,h in a['frozen_hashes'].items())
    save(N/'secondary-feature-labels.json',dict(updated=now(),records=details,original_bytes_unchanged=True))
    save(N/'secondary-observations.json',dict(frames=obs['frames'],tracks=tracks,coordinates_unchanged=True))
    save(N/'secondary-families.json',dict(rows=families,original_membership_and_splits_unchanged=True))
    eligible=[f for f in families if f['secondary_candidate_background']]
    recovered=[f for f in eligible if not f['candidate_background']]
    lost=[f for f in families if f['candidate_background'] and not f['secondary_candidate_background']]
    coverage=[]
    for fr in obs['frames']:
        fams=[f for f in eligible if any(o['frame']==fr for t in f['tracks'] for o in tracks[t]['observations'])]
        coverage.append(dict(frame=fr,candidate_families=len(fams),fit=sum(f['split']=='fit' for f in fams),heldout=sum(f['split']=='heldout' for f in fams)))
    mixed_cases=[]
    for c in read(O/'mixed-region-cases/index.json')['rows']:
        fa=next(x for x in families if x['family']==c['family']);t=tracks[c['track']]
        mixed_cases.append(dict(case=c['case'],family=c['family'],track=c['track'],secondary_candidate_background=fa['secondary_candidate_background'],labels=[o['secondary_semantic_label'] for o in t['observations']]))
    result=dict(updated=now(),state='secondary_labels_complete_physical_review_pending',seconds=time.monotonic()-start,frames=33,rejected_mask_frames=[617],
        tracks=len(tracks),families=len(families),original_candidate_families=sum(f['candidate_background'] for f in families),
        secondary_candidate_families=len(eligible),recovered_families=len(recovered),lost_families=len(lost),
        recovered_ids=[f['family'] for f in recovered],lost_ids=[f['family'] for f in lost],coverage=coverage,mixed_cases=mixed_cases,
        input_bytes_and_coordinates_unchanged=True,family_splits_unchanged=True,quarantined_families=sum(f['split']=='quarantine' for f in families),
        original_contradictions_retained=sum(f['inherited_contradiction'] for f in families),camera_operations=0,training_started=False,physical_identity_verified=False,
        interpretation='Secondary eligibility is only a spatial support screen. Masks are coarse proposals; reflection, repeating textures and actual material identity still require review. Failed mask617 makes all observations there unknown; no raw frame removed or interpolated.')
    save(N/'classification-result.json',result);save(N/'progress.json',dict(updated=now(),state=result['state'],pid=None,training_started=False))
    print(json.dumps({k:result[k] for k in ['state','secondary_candidate_families','recovered_families','lost_families','seconds']}),flush=True)

if __name__=='__main__':run()
