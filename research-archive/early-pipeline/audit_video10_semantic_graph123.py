"""Read-only necessary support closure. No F/E/PnP/BA or camera interpolation."""
import itertools
import match_video10_temporal_bridge123 as bridge
b=bridge.base; N=b.R/'video10-semantic-role'; O=bridge.O
assert not (N/'graph-audit-plan.json').exists()
src=N/'material-support.json'; support=b.read(src); rows=support['combined_rows']
frames=b.read(O/'plan.json')['frames']; threshold_pair=12; threshold_pnp=8
b.save(N/'graph-audit-plan.json',dict(registered=b.now(),input_sha256=b.sha(src),
    hypothesis='Restoring background appearance may increase totals without connecting the whole camera graph. Count-only closure is a necessary upper bound, not successful triangulation or physical accuracy.',
    frozen='All reviewed family identities, coordinates, splits and exclusions; no new review selection or parameters.',
    budget=dict(seed_pairs='all pairs with at least12 reviewed FIT families, each once',CPU_seconds=10,geometry_optimizations=0,new_observations=0),
    operation='Seed families seen in both views; add a view only if >=8 current potentially triangulatable FIT families observe it; extend potential families seen by >=2 reached views. Report each closure and missing support. Counts overestimate real usable3D support.',
    stop='No fitting. Insufficient connected support cannot be repaired with interpolation, lowering thresholds or moving HELD to FIT.'))
fit={r['family']:set(r['frames']) for r in rows if r['split']=='fit'}
byframe={f:{fam for fam,ff in fit.items() if f in ff} for f in frames}
seeds=[(a,c) for a,c in itertools.combinations(frames,2) if len(byframe[a]&byframe[c])>=threshold_pair]
results=[]
for seed in seeds:
    reached=set(seed); sequence=[]
    while True:
        active={fam for fam,ff in fit.items() if len(ff&reached)>=2}
        choices=[(len(byframe[f]&active),f) for f in frames if f not in reached]
        ready=sorted([x for x in choices if x[0]>=threshold_pnp],key=lambda x:(-x[0],x[1]))
        if not ready:break
        c,f=ready[0]; reached.add(f);sequence.append(dict(frame=f,potential_2d3d_count=c))
    results.append(dict(seed=list(seed),seed_families=sorted(byframe[seed[0]]&byframe[seed[1]]),
        reached=sorted(reached),addition_sequence=sequence,unreached=[f for f in frames if f not in reached],
        unreached_shared={str(f):len(byframe[f]&active) for f in frames if f not in reached}))
best=max(results,key=lambda r:len(r['reached'])) if results else None
# Weak shared-observation connectivity is separately reported, never promoted to PnP support.
neighbors={f:{g for g in frames if f!=g and len(byframe[f]&byframe[g])>=threshold_pnp} for f in frames}
remaining=set(frames); components=[]
while remaining:
    todo=[min(remaining)]; comp=set()
    while todo:
        f=todo.pop()
        if f in comp:continue
        comp.add(f);todo.extend(neighbors[f]-comp)
    remaining-=comp;components.append(sorted(comp))
result=dict(updated=b.now(),plan_sha256=b.sha(N/'graph-audit-plan.json'),input_sha256=b.sha(src),
    all_seed_closures=results,best_closure=best,reviewed_fixed_fit=60,frame_count=len(frames),
    weak_shared8_components=components,counts_are_upper_bounds=True,
    every_registered_frame_has8_fit=all(len(byframe[f])>=8 for f in frames),
    whole_graph_count_gate=bool(best and len(best['reached'])==len(frames)),
    cameras_estimated=False,training_admitted=False,
    conclusion='Even coarse correspondence counts cannot connect every registered view. No physical calibration or SMPL-X alignment is established.')
assert b.sha(src)==b.read(N/'graph-audit-plan.json')['input_sha256']
b.save(N/'graph-audit-result.json',result)
print(result)
