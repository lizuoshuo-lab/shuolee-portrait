"""Unchanged necessary support graph after the complete frozen222 review."""
from pathlib import Path
import json,itertools,time
import match_video10_temporal_bridge123 as b
U=b.R/'video10-remaining-material-review';p=U/'material-support.json';s=b.read(p)
assert b.read(U/'visual-review.json')['reviewed_rows']==222
assert not (U/'graph-audit-plan.json').exists()
plan=dict(registered=b.now(),input_sha256=b.sha(p),previous_algorithm_sha256=b.sha(Path(__file__).with_name('audit_video10_semantic_graph123.py')),
    operation='Same prior count-only closure: >=12FIT shared seed, >=8 potentially triangulatable FIT families for each new view, all possible seeds once. Whole original families and splits fixed.',
    hypothesis='Adding previously unreviewed existing physical candidates may restore graph support; no new detector/matcher/K/pose variable.',
    budget=dict(CPU_seconds=10,new_observations=0,new_inference=0,camera_optimizations=0),
    interpretation='Coarse counts overestimate independent material/3D support. Passing is only a necessary condition; it does not establish actual camera estimates, all-video registration or face<1px.')
b.save(U/'graph-audit-plan.json',plan)
start=time.monotonic();frames=b.read(b.O/'plan.json')['frames']
fit={r['family']:set(r['frames']) for r in s['combined_rows'] if r['split']=='fit'}
byframe={f:{k for k,v in fit.items() if f in v} for f in frames}
seeds=[(a,c) for a,c in itertools.combinations(frames,2) if len(byframe[a]&byframe[c])>=12]
result=[]
for seed in seeds:
    reached=set(seed);sequence=[]
    while True:
        active={k for k,v in fit.items() if len(v&reached)>=2}
        ready=sorted([(len(byframe[f]&active),f) for f in frames if f not in reached and len(byframe[f]&active)>=8],key=lambda v:(-v[0],v[1]))
        if not ready:break
        count,f=ready[0];reached.add(f);sequence.append(dict(frame=f,potential_2d3d_count=count))
    result.append(dict(seed=list(seed),seed_families=sorted(byframe[seed[0]]&byframe[seed[1]]),reached=sorted(reached),addition_sequence=sequence,
        unreached=[f for f in frames if f not in reached],unreached_shared={str(f):len(byframe[f]&active) for f in frames if f not in reached}))
best=max(result,key=lambda r:len(r['reached'])) if result else None
neighbors={f:{g for g in frames if f!=g and len(byframe[f]&byframe[g])>=8} for f in frames};remaining=set(frames);components=[]
while remaining:
    todo=[min(remaining)];comp=set()
    while todo:
        f=todo.pop()
        if f in comp:continue
        comp.add(f);todo.extend(neighbors[f]-comp)
    remaining-=comp;components.append(sorted(comp))
elapsed=time.monotonic()-start;assert elapsed<10 and b.sha(p)==plan['input_sha256']
ok=bool(best and len(best['reached'])==len(frames))
out=dict(updated=b.now(),elapsed_seconds=elapsed,plan_sha256=b.sha(U/'graph-audit-plan.json'),input_sha256=b.sha(p),all_seed_closures=result,best_closure=best,
    reviewed_fixed_fit=len(fit),frame_count=len(frames),weak_shared8_components=components,counts_are_upper_bounds=True,
    every_registered_frame_has8_fit=all(len(byframe[f])>=8 for f in frames),whole_graph_count_gate=ok,cameras_estimated=False,training_admitted=False,
    conclusion='Count-only necessary coverage '+('passes; actual geometry, physical points and calibration unverified.' if ok else 'still incomplete; do not infer all-frame camera support from per-frame totals.'))
b.save(U/'graph-audit-result.json',out)
print(json.dumps({k:v for k,v in out.items() if k!='all_seed_closures'},ensure_ascii=False))
