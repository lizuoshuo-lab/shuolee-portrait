"""Correct the ceiling's omission of already visually recovered background families."""
import itertools, json, time
from pathlib import Path
import match_video10_temporal_bridge123 as b

U=b.R/'video10-remaining-material-review'; N=b.R/'video10-semantic-role'
assert not (U/'ceiling-corrected-plan.json').exists()
inputs=[U/'ceiling-result.json',U/'material-support.json',U/'identity-contradictions.json',
        N/'secondary-families.json',N/'secondary-observations.json',N/'material-visual-review.json']
support=b.read(U/'material-support.json'); families=b.read(N/'secondary-families.json')['rows']
tracks=b.read(N/'secondary-observations.json')['tracks']; frames=b.read(b.O/'plan.json')['frames']
reviewed={r['family']:set(r['frames']) for r in support['combined_rows'] if r['split']=='fit'}
reject={r['family'] for r in b.read(U/'identity-contradictions.json')['rows']}
reject.update(r['family'] for r in b.read(N/'material-visual-review.json')['rows'] if r['verdict']=='dynamic_display_excluded')
eligible_old={f['family'] for f in families if f['secondary_candidate_background'] and f['split']=='fit' and not f['conflict'] and not f['inherited_contradiction'] and f['family'] not in reject}
missing=sorted(set(reviewed)-eligible_old)
assert missing==[945,946,1015,1029,1032,1043,1120,1124]
for f in families:
    if f['family'] in missing:
        assert f['split']=='fit' and not f['conflict'] and not f['inherited_contradiction'] and f['family'] not in reject
plan=dict(registered=b.now(),reason='The initial ceiling filtered secondary_candidate_background and omitted8 already visually recovered background FIT families. Its broad variant was not a superset of the reviewed baseline.',
    correction='Union the original eligible set with the already reviewed coarse FIT families. No new material is certified. Preserve original result as a flawed-ceiling record.',
    recovered_families=missing,source_sha256={str(p):b.sha(p) for p in inputs},
    unchanged='Same saved pixels, family identities, FIT/HELD, 12 seed and8 extension gates; no threshold change, matching, camera or training.',
    union_caveat='Merging near-duplicate branch frame sets is only an optimistic count bound, never a valid shared 3D track.',
    budget=dict(CPU_seconds=10,variants=3,inference=0,camera_estimates=0,training=0))
b.save(U/'ceiling-corrected-plan.json',plan)
start=time.monotonic()

def ceiling(fit):
    byframe={f:{k for k,v in fit.items() if f in v} for f in frames}
    seeds=[(a,c) for a,c in itertools.combinations(frames,2) if len(byframe[a]&byframe[c])>=12]
    records=[]
    for seed in seeds:
        reached=set(seed); seq=[]
        while True:
            active={k for k,v in fit.items() if len(v&reached)>=2}
            ready=sorted([(len(byframe[f]&active),f) for f in frames if f not in reached and len(byframe[f]&active)>=8],key=lambda x:(-x[0],x[1]))
            if not ready: break
            count,f=ready[0]; reached.add(f); seq.append(dict(frame=f,potential=count))
        records.append(dict(seed=list(seed),seed_count=len(byframe[seed[0]]&byframe[seed[1]]),reached=sorted(reached),extension=seq,
                            unreached_counts={str(f):len(byframe[f]&active) for f in frames if f not in reached}))
    best=max(records,key=lambda r:len(r['reached'])) if records else None
    return dict(fit_families=len(fit),seed_count=len(seeds),best=best,reached_count=len(best['reached']) if best else 0,
                all33_possible=bool(best and len(best['reached'])==33),all_seeds=records)

eligible=[f for f in families if f['family'] in eligible_old or f['family'] in reviewed]
allbranches={f['family']:{o['frame'] for t in f['tracks'] for o in tracks[t]['observations']} for f in eligible}
assert set(reviewed)<=set(allbranches)
assert all(v<=allbranches[k] for k,v in reviewed.items())
variants={'reviewed_representative':ceiling(reviewed),
          'reviewed_family_branches_optimistic':ceiling({k:v for k,v in allbranches.items() if k in reviewed}),
          'all_eligible_fit_optimistic':ceiling(allbranches)}
old=b.read(U/'ceiling-result.json')
assert variants['reviewed_representative']==old['variants']['reviewed_representative']
assert variants['reviewed_family_branches_optimistic']['fit_families']==162
assert variants['all_eligible_fit_optimistic']['fit_families']==527
for key in variants:
    assert variants[key]['reached_count']>=old['variants'][key]['reached_count']
elapsed=time.monotonic()-start; assert elapsed<10
for p,h in plan['source_sha256'].items(): assert b.sha(Path(p))==h
result=dict(updated=b.now(),elapsed_seconds=elapsed,plan_sha256=b.sha(U/'ceiling-corrected-plan.json'),variants=variants,
            corrected_initial_omission=missing,baseline_reproduced=True,all_variants_contain_baseline=True,
            new_observations=0,new_cameras=0,training_admitted=False,ceiling_is_not_actual_correspondence_or_calibration=True,
            scope='This is only the current sequential count admission using saved graph candidates. Failure does not prove every reconstruction method or the whole original video impossible.')
b.save(U/'ceiling-corrected-result.json',result)
print(json.dumps({name:{k:v for k,v in value.items() if k not in ['best','all_seeds']} for name,value in variants.items()}))
