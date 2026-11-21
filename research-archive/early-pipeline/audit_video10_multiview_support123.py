"""Single read-only support audit. Counts/cycle rank are not camera accuracy."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,itertools,time,collections
import numpy as np
from global_track_initializer123 import connected_components
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
G=R/'video10-global-initializer';O=G/'multiview-support';P=G/'multiview-support-plan.json'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=lambda:datetime.now(timezone.utc).isoformat()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
O.mkdir(exist_ok=True);assert not (O/'execution-admission.json').exists()
plan=read(P)
for p,h in plan['source_sha256'].items():assert sha(Path(p))==h
save(O/'execution-admission.json',dict(updated=now(),pid=os.getpid(),plan_sha256=sha(P),worker_sha256=sha(Path(__file__)),GPU=False,training=False))
start=time.monotonic()
fit=read(R/'video10-remaining-material-review/global-input-preflight/fit-observations.json');frames=fit['frames'];fi={f:i for i,f in enumerate(frames)}
vis={t['family']:set(o['frame'] for o in t['observations']) for t in fit['tracks']}
assert len(vis)==len(fit['tracks'])==162
perframe={f:{k for k,v in vis.items() if f in v} for f in frames}
pairs=read(G/'relative-graph/repair-invalid-pose/pairs.json');valid=[p for p in pairs if 'R_b_from_a' in p]
edgeindex={(p['a'],p['b']):i for i,p in enumerate(valid)}
cycles=read(G/'relative-graph/repair-invalid-pose/rotation-cycles.json')
tri=[];incidence=np.zeros((len(cycles),len(valid)))
for i,c in enumerate(cycles):
    a,b,d=c['frames'];S={a,b,d};shared={f:perframe[f] for f in S}
    repeated={fid for fid,v in vis.items() if len(v&S)>=2}
    allthree={fid for fid,v in vis.items() if S<=v}
    counts={str(f):len(repeated&perframe[f]) for f in c['frames']}
    union=set.union(*shared.values());edges=[(a,b),(b,d),(a,d)]
    for key,sgn in zip(edges,[1.,1.,-1.]):incidence[i,edgeindex[key]]=sgn
    tri.append(dict(id=i,frames=c['frames'],cycle_deg=c['rotation_closure_deg'],
        distinct_FIT_in_at_least_two_views=len(repeated),FIT_repeated_families=sorted(repeated),
        per_view_repeated_FIT=counts,all_three_FIT=len(allthree),all_three_families=sorted(allthree),
        pairs=[dict(frames=[a,b],shared=valid[edgeindex[(a,b)]]['count'],robust=valid[edgeindex[(a,b)]]['robust_admitted']) for a,b in edges],
        descriptive_12_total_8_per_view=len(repeated)>=12 and min(counts.values())>=8,
        admission=False,reason='Union-of-triple counts describe a different joint factor; do not relax old per-pair12robust rule.'))
assert len(tri)==32
links=[]
for i,j in itertools.combinations(range(len(tri)),2):
    cams=set(tri[i]['frames'])&set(tri[j]['frames'])
    common=set(tri[i]['FIT_repeated_families'])&set(tri[j]['FIT_repeated_families'])
    # Three shared XYZ points is only a necessary algebraic similarity support count;
    # collinearity, identity and actual coordinates are not verified here.
    if len(cams)>=2 or len(common)>=3:
        links.append(dict(a=i,b=j,shared_cameras=sorted(cams),shared_families=sorted(common),
            justification='>=2camera identities or>=3material representatives; geometric degeneracy and independence unknown'))
group2=connected_components(len(tri),[(r['a'],r['b']) for r in links if len(r['shared_cameras'])>=2])
groupboth=connected_components(len(tri),[(r['a'],r['b']) for r in links])
def groupinfo(groups):return [dict(triples=g,frames=sorted(set(f for i in g for f in tri[i]['frames'])),families=sorted(set(f for i in g for f in tri[i]['FIT_repeated_families']))) for g in groups]
components=connected_components(len(frames),[(fi[a],fi[b]) for a,b in edgeindex]);cycle_dim=len(valid)-len(frames)+len(components)
rank=int(np.linalg.matrix_rank(incidence));_,sv,vt=np.linalg.svd(incidence,full_matrices=True)
participation=np.count_nonzero(incidence,axis=0)
untri=[dict(a=p['a'],b=p['b'],shared=p['count'],inliers=p['inlier_count'],robust=p['robust_admitted']) for p,n in zip(valid,participation) if n==0]
covered=set(f for t in tri for f in t['frames']);missing=sorted(set(frames)-covered)
unknownbridges=[]
for f in missing:
    touching=[t for t,v in vis.items() if f in v]
    unknownbridges.append(dict(frame=f,fit_families=sorted(touching),tracks=[dict(family=t,frames=sorted(vis[t])) for t in touching],
        incident_pairs=[dict(a=p['a'],b=p['b'],shared=p['count'],state=p['state'],robust=p['robust_admitted']) for p in pairs if f in [p['a'],p['b']]]))
# List material evidence outside the diagnosed pair triangles, without creating/solving new pairs.
materialtriples=collections.defaultdict(set)
for fid,v in vis.items():
    for tr in itertools.combinations(sorted(v),3):materialtriples[tr].add(fid)
outside=[dict(frames=list(tr),shared_three_FIT=len(ids),families=sorted(ids),
    missing_pose_edges=[list(e) for e in itertools.combinations(tr,2) if e not in edgeindex])
    for tr,ids in sorted(materialtriples.items()) if any(e not in edgeindex for e in itertools.combinations(tr,2))]
save(O/'closed-triples.json',tri);save(O/'triple-links.json',links);save(O/'material-triples-outside-pair-graph.json',outside)
save(O/'uncovered-frames.json',unknownbridges)
np.savez_compressed(O/'cycle-incidence.npz',matrix=incidence,singular_values=sv,edges=np.asarray(list(edgeindex)))
for p,h in plan['source_sha256'].items():assert sha(Path(p))==h
elapsed=time.monotonic()-start;assert elapsed<plan['budget']['CPU_seconds']
result=dict(updated=now(),state='read_only_multiview_support_audited',plan_sha256=sha(P),elapsed_seconds=elapsed,
    frames=len(frames),fit_families=len(vis),pose_edges=len(valid),pose_graph_components=[[frames[i] for i in g] for g in components],
    closed_triples=len(tri),triangle_covered_frames=sorted(covered),triangle_missing_frames=missing,
    cycle_space_dimension=cycle_dim,triangle_cycle_rank=rank,unrepresented_long_cycle_dimensions=cycle_dim-rank,
    edges_without_triangle=untri,
    descriptive_12total_8each_triples=sum(t['descriptive_12_total_8_per_view'] for t in tri),
    two_camera_overlap_components=groupinfo(group2),camera_or_material_overlap_components=groupinfo(groupboth),
    material_only_extra_triples=len(outside),
    material_extra_max_three_shared=max((t['shared_three_FIT'] for t in outside),default=0),
    physical_families_independence_certified=False,geometric_scale_identifiability_tested=False,
    quality_accepted=False,new_camera_estimates=0,heldout_used=False,training_started=False,
    limitation='Topology/counts only. Coarse repeated material families can still be collinear, wrong, blurred or nonrigid. Triangle closure is fit consistency, not physical accuracy. Raw three-view observations outside the estimated pair graph are explicitly retained, not treated as absent.')
save(O/'result.json',result);save(O/'progress.json',dict(updated=now(),pid=None,state='complete',training_started=False))
brief={k:v for k,v in result.items() if k not in ['two_camera_overlap_components','camera_or_material_overlap_components','edges_without_triangle']}
brief.update(two_camera_components=[x['frames'] for x in result['two_camera_overlap_components']],camera_or_material_components=[x['frames'] for x in result['camera_or_material_overlap_components']],untriangulated_pose_edges=untri)
print(json.dumps(brief,ensure_ascii=False))
