"""Once-only optimistic count ceilings of saved tracks, never camera admission."""
import itertools,time,json
from pathlib import Path
import match_video10_temporal_bridge123 as b
U=b.R/'video10-remaining-material-review';N=b.R/'video10-semantic-role'
assert not (U/'ceiling-plan.json').exists()
inputs=[U/'material-support.json',U/'identity-contradictions.json',U/'graph-audit-result.json',N/'secondary-families.json',N/'secondary-observations.json',N/'material-visual-review.json']
plan=dict(registered=b.now(),hypothesis='The reviewed subset may fail while unreviewed cached tracks could still connect views. Check an intentionally optimistic ceiling before inferring extraction failure or starting another inference pass.',
    source_sha256={str(p):b.sha(p) for p in inputs},
    variants=['reviewed family exact representative baseline','all old branches of reviewed coarse FIT families','all original eligible FIT families including ambiguous/unreviewed as optimistic potential'],
    exclusions='Original FIT/HELD unchanged; no HELD used. Original conflict/quarantine/inherited contradiction and newly confirmed dynamic display or material contradiction excluded.',
    union_caveat='Union across near-duplicate branches intentionally treats them as if one common material point for a ceiling only; it is not a valid observation track and must never be used in a solver.',
    budget=dict(CPU_seconds=10,geometry_solves=0,new_observations=0,inference_passes=0),
    stopping='Three fixed ceilings once with unchanged >=12 seed/>=8 extension. No threshold scan, camera estimation or promotion of unreviewed observations.')
b.save(U/'ceiling-plan.json',plan)
start=time.monotonic();frames=b.read(b.O/'plan.json')['frames']
support=b.read(U/'material-support.json');families=b.read(N/'secondary-families.json')['rows'];tracks=b.read(N/'secondary-observations.json')['tracks']
reject={r['family'] for r in b.read(U/'identity-contradictions.json')['rows']}
reject.update(r['family'] for r in b.read(N/'material-visual-review.json')['rows'] if r['verdict']=='dynamic_display_excluded')
def ceiling(fit):
    byframe={f:{k for k,v in fit.items() if f in v} for f in frames}
    seeds=[(a,c) for a,c in itertools.combinations(frames,2) if len(byframe[a]&byframe[c])>=12]
    records=[]
    for seed in seeds:
        reached=set(seed);seq=[]
        while True:
            active={k for k,v in fit.items() if len(v&reached)>=2}
            ready=sorted([(len(byframe[f]&active),f) for f in frames if f not in reached and len(byframe[f]&active)>=8],key=lambda x:(-x[0],x[1]))
            if not ready:break
            n,f=ready[0];reached.add(f);seq.append(dict(frame=f,potential=n))
        records.append(dict(seed=list(seed),seed_count=len(byframe[seed[0]]&byframe[seed[1]]),reached=sorted(reached),extension=seq,unreached_counts={str(f):len(byframe[f]&active) for f in frames if f not in reached}))
    best=max(records,key=lambda r:len(r['reached'])) if records else None
    return dict(fit_families=len(fit),seed_count=len(seeds),best=best,reached_count=len(best['reached']) if best else 0,all33_possible=bool(best and len(best['reached'])==33),all_seeds=records)
reviewed={r['family']:set(r['frames']) for r in support['combined_rows'] if r['split']=='fit'}
eligible=[f for f in families if f['secondary_candidate_background'] and f['split']=='fit' and not f['conflict'] and not f['inherited_contradiction'] and f['family'] not in reject]
allbranches={f['family']:{o['frame'] for t in f['tracks'] for o in tracks[t]['observations']} for f in eligible}
variants={
    'reviewed_representative':ceiling(reviewed),
    'reviewed_family_branches_optimistic':ceiling({k:v for k,v in allbranches.items() if k in reviewed}),
    'all_eligible_fit_optimistic':ceiling(allbranches)}
old=b.read(U/'graph-audit-result.json')['best_closure'];assert variants['reviewed_representative']['best']['reached']==old['reached']
elapsed=time.monotonic()-start;assert elapsed<10
for p,h in plan['source_sha256'].items():assert b.sha(Path(p))==h
result=dict(updated=b.now(),elapsed_seconds=elapsed,plan_sha256=b.sha(U/'ceiling-plan.json'),variants=variants,new_observations=0,new_cameras=0,training_admitted=False,
    known_excluded_families=sorted(reject),ceiling_is_not_actual_correspondence_or_calibration=True)
b.save(U/'ceiling-result.json',result)
print(json.dumps({name:{k:v for k,v in val.items() if k not in ['all_seeds','best']} for name,val in variants.items()},ensure_ascii=False))
