"""Bounded training-only frame-index hypothesis; no image or dataset mutation."""
from pathlib import Path
import json, hashlib, shutil
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint

LAB=Path(__file__).resolve().parents[2]
SRC=LAB/'outputs/dynamic-human-enerf-actor56-time-049'
OUT=LAB/'outputs/dynamic-human-enerf-time-graph-051'
OUT.mkdir(exist_ok=True)
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
if (OUT/'summary.json').exists():raise FileExistsError('Completed graph diagnostic')
save(OUT/'protocol.json',dict(fit_frames=[740,750],check_frame=760,minimum_moving_tracks_per_fit_anchor=12,
    offsets=list(range(-2,3)),edge_differences=list(range(-3,4)),gauge_camera='01',gauge_offset=0,
    objective='Equal edge weight, average of two fit-anchor median absolute Sampson distances in native pixels.',
    solver='Binary one-hot node and edge variables, exact discrete difference constraints; scipy HiGHS MILP.',
    limits=['Constant relative-index hypothesis, not physical camera-clock calibration.',
    'An edge table holds source time fixed, so shifting both cameras requires further direct validation.',
    'Only training cameras; no heldout image or mask access. No retiming or training in this diagnostic.'],
    check_data_used_in_fit=False))
raw=json.loads((SRC/'timing-diagnostic.json').read_text())['pairs']
lookup={(r['source'],r['target'],r['anchor']):r for r in raw}
allpairs=sorted({(r['source'],r['target']) for r in raw})
edges=[]
for c,d in allpairs:
    rows=[lookup.get((c,d,t)) for t in [740,750]]
    if all(r and r['moving_count']>=12 for r in rows):
        cost=np.mean([r['moving']['scores_px'] for r in rows],axis=0)
        edges.append((c,d,cost))
cams=sorted({c for c,d,_ in edges}|{d for c,d,_ in edges});assert '01' in cams
adj={c:set() for c in cams}
for c,d,_ in edges:adj[c].add(d);adj[d].add(c)
seen={'01'}
while True:
    after=seen|set.union(*(adj[c] for c in seen))
    if after==seen:break
    seen=after
assert seen==set(cams),'Graph disconnected: needs component-specific gauges'
labels=np.arange(-2,3);lags=np.arange(-3,4);nc=len(cams);ne=len(edges);nv=5*nc+7*ne
cost=np.zeros(nv);matrix=[];rhs=[]
for i,c in enumerate(cams):
    r=np.zeros(nv);r[5*i:5*i+5]=1;matrix.append(r);rhs.append(1)
for j,(c,d,values) in enumerate(edges):
    k=5*nc+7*j;cost[k:k+7]=values
    r=np.zeros(nv);r[k:k+7]=1;matrix.append(r);rhs.append(1)
    r=np.zeros(nv);r[k:k+7]=lags
    r[5*cams.index(c):5*cams.index(c)+5]=labels
    r[5*cams.index(d):5*cams.index(d)+5]=-labels
    matrix.append(r);rhs.append(0)
r=np.zeros(nv);r[5*cams.index('01')+2]=1;matrix.append(r);rhs.append(1)
mat=np.stack(matrix);rhs=np.array(rhs)
result=milp(cost,integrality=np.ones(nv),bounds=Bounds(np.zeros(nv),np.ones(nv)),
    constraints=LinearConstraint(mat,rhs,rhs),options=dict(time_limit=45,mip_rel_gap=0))
assert result.success and result.mip_gap==0,result.message
solution=np.rint(result.x);assert np.max(np.abs(mat@solution-rhs))<1e-9
offset={c:int(labels[np.argmax(solution[5*i:5*i+5])]) for i,c in enumerate(cams)}
records=[]
for c,d,values in edges:
    lag=offset[d]-offset[c];assert -3<=lag<=3
    check=lookup.get((c,d,760));covered=check is not None and check['moving_count']>=12
    records.append(dict(source=c,target=d,lag=lag,fit_zero_px=float(values[3]),fit_selected_px=float(values[lag+3]),
        check_count=check['moving_count'] if check else 0,check_covered=covered,
        check_zero_px=check['moving']['scores_px'][3] if covered else None,
        check_selected_px=check['moving']['scores_px'][lag+3] if covered else None))
covered=[r for r in records if r['check_covered']]
summary=dict(status='training_only_globally_consistent_relative_indices_fit_check_pending_direct_observation_validation',
    offsets=offset,fit_edges=ne,connected_cameras=len(cams),objective_zero=float(sum(v[3] for _,_,v in edges)),
    objective_selected=float(result.fun),mip_gap=float(result.mip_gap),records=records,
    check_edges=len(covered),check_zero_mean_px=float(np.mean([r['check_zero_px'] for r in covered])),
    check_selected_mean_px=float(np.mean([r['check_selected_px'] for r in covered])),
    changed_check_edges=sum(r['lag']!=0 for r in covered),
    changed_check_improved=sum(r['lag']!=0 and r['check_selected_px']<r['check_zero_px'] for r in covered),
    source_sha256=hashlib.sha256((SRC/'timing-diagnostic.json').read_bytes()).hexdigest(),goal_complete=False)
save(OUT/'summary.json',summary)
shutil.copy2(__file__,OUT/Path(__file__).name)
print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2))
