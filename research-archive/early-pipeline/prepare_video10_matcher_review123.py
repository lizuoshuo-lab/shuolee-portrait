"""Freeze all review queries before any new matcher output exists."""
import numpy as np
import match_video10_temporal_bridge123 as bridge
b=bridge.base;R=b.R;O=bridge.O;B=bridge.B;N=R/'video10-semantic-role';Q=R/'video10-correspondence-contrast'
assert not (Q/'review-allocation.json').exists() and not (Q/'run').exists()
p=b.read(Q/'plan.json'); old_plan_sha=b.sha(Q/'plan.json')
tracks=b.read(N/'secondary-observations.json')['tracks']; fs={x['family']:x for x in b.read(N/'secondary-families.json')['rows']}
support=b.read(N/'material-support.json')['combined_rows']; graph=b.read(N/'graph-audit-result.json')
byframe={f:[] for f in p['frames']}
for t in tracks:
    fa=fs[t['family']]
    if fa['conflict'] or fa['inherited_contradiction'] or fa['split']=='quarantine':continue
    for o in t['observations']:
        if o['frame'] not in byframe:continue
        byframe[o['frame']].append(dict(family=t['family'],track=t['track'],split=t['split'],**o))
chosen=[]
for pair_index,(a,c) in enumerate(p['pairs']):
    selected=[];used=set()
    if pair_index==6:
        for fam in graph['best_closure']['seed_families']:
            r=next(x for x in support if x['family']==fam)
            t=tracks[r['track']];o=next(x for x in t['observations'] if x['frame']==a)
            selected.append(dict(family=fam,track=t['track'],split=t['split'],**o,review_group='coarse_positive_control'))
    else:
        # Source role eligibility is frozen, never based on the new prediction or its confidence.
        for split in ['fit','heldout']:
            got=0
            for cycle in range(4):
                for cell in range(9):
                    if got>=4:break
                    pool=[x for x in byframe[a] if x['family'] not in used and x['split']==split and x['secondary_semantic_label']=='background_candidate' and min(2,int(x['uv'][1]*3/1520))*3+min(2,int(x['uv'][0]*3/2704))==cell]
                    if not pool:continue
                    x=min(pool,key=lambda x:b.rank(f'{pair_index}|{x["family"]}|{x["feature"]}|review-background'))
                    selected.append(dict(**x,review_group='background_candidate'));used.add(x['family']);got+=1
        for split in ['fit','heldout']:
            pool=[x for x in byframe[a] if x['family'] not in used and x['split']==split and x['role']==40]
            if pool:
                x=min(pool,key=lambda x:b.rank(f'{pair_index}|{x["family"]}|{x["feature"]}|review-head'))
                selected.append(dict(**x,review_group='headROI_unknown_material'));used.add(x['family'])
    fp=(B if a in b.FRAMES else O)/'run'/f'features_{a:06d}.npz'
    with np.load(fp) as d:
        for x in selected:
            assert np.array_equal(d['uv'][x['feature']],np.array(x['uv'],np.float32))
            oldtargets=[o for o in tracks[x['track']]['observations'] if o['frame']==c]
            chosen.append(dict(row=len(chosen),pair_index=pair_index,source=a,target=c,query_feature=x['feature'],source_uv=x['uv'],family=x['family'],split=x['split'],review_group=x['review_group'],
                frozen_same_track_target=oldtargets[0] if oldtargets else None,
                physical_point_certified=False,source_role=x['role'],source_secondary_label=x['secondary_semantic_label']))
assert len(chosen)<=72
b.save(Q/'review-allocation.json',dict(registered=b.now(),before_inference=True,plan_sha256_before_clarification=old_plan_sha,rows=chosen,
    selection='Each of6diagnostic pairs: up to4FIT+4HELD background families spatial3x3 roundrobin, SHA tie; up to1FIT+1HELD headROI. Last control pair all12 prior reviewed FITseedfamilies. Duplicate sourcefamilies within eachpair avoided; crosspair repetition declared, not extraindependentGT.',
    max_contexts=16,actual_rows=len(chosen)))
p['visual_review']['selection']='Frozen review-allocation.json: six pairs up to10queries each plus12 prior positive coarse control families, at most72rows. Native3x3 spatial roundrobin andSHA before inference; both accepted and rejected outputs inspected. HeadROI may contain background; not a faceGT label.'
p['review_allocation_sha256']=b.sha(Q/'review-allocation.json');b.save(Q/'plan.json',p)
print(dict(review_queries=len(chosen),inference_started=False,old_query_coordinates_unchanged=True))
