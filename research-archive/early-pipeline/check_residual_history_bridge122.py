"""Read-only transport of saved N6 camera parameters into the explicit new model.

No camera fitting, point changes, correspondence search, or candidate promotion.
This only checks formula direction and integer-center -> pixel-corner transport.
"""
from pathlib import Path
import json,hashlib
import numpy as np
from native_residual_camera122 import projection,with_projection,project_world,rays_world
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
O=R/'native-residual-camera';N=R/'video-update-20261027/background-geometry/residual-radial/serialization-repair'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if (O/'historical-n6-bridge.json').exists():raise FileExistsError('Once-only history bridge')
inputs={str(p):sha(p) for p in [N/'result.json',N/'fit-diagnostic.npz']}
plan=dict(scope='read-only saved N6 mapping regression',inputs=inputs,
    hypothesis='New Brown contract with onlyk1 reproduces historical N6 output mapping when K andpixels both shift+0.5.',
    budget='one CPU run, no solver or feature extraction',data_use='saved FIT points only; no residual-based material selection',
    acceptance='formula andorigin transport max<1e-9px, all existing files unchanged',physical_promotion=False)
(O/'historical-n6-bridge-plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
r=json.loads((N/'result.json').read_text(encoding='utf-8'));K=np.array(r['K_integer_centers']);k=r['k1']
with np.load(N/'fit-diagnostic.npz',allow_pickle=False) as z:points=z['points']
pr=projection([k,0,0,0,0,0],[0,0],[0,0,0,0],input_domain='processed_output_pixels')
rows=[]
for c in r['cameras']:
    e=np.array(c['w2c']);pc=points@e[:3,:3].T+e[:3,3];good=pc[:,2]>0
    # Visibility exclusion here is geometric, never based on measured residual.
    p=points[good];xy=pc[good,:2]/pc[good,2:];f=np.diag(K)[:2]
    old=K[:2,2]+f*xy*(1+k*np.sum(xy*xy,axis=1,keepdims=True))
    corner=K.copy();corner[:2,2]+=.5
    camera=with_projection(dict(id=str(c['frame']),image_name=str(c['frame'])+'.png',width=1520,height=2704,K=corner.tolist(),w2c=e.tolist()),pr)
    actual=project_world(p,camera)
    err=float(np.max(np.abs(actual-(old+.5))))
    origin,dirs=rays_world(actual,camera);expected=p-origin;expected/=np.linalg.norm(expected,axis=1,keepdims=True)
    rayerr=float(np.max(np.abs(dirs-expected)))
    if err>=1e-9 or rayerr>=1e-9:raise AssertionError('Historical model mapping mismatch')
    rows.append(dict(frame=c['frame'],positive_depth_points=int(good.sum()),behind_camera=int((~good).sum()),max_abs_px=err,max_ray_difference=rayerr))
assert all(sha(p)==h for p,h in inputs.items())
summary={g:d['summary'] for g,d in r['validation'].items()}
result=dict(passed=True,rows=rows,original_files_unchanged=True,old_k1=k,new_K_adopted=False,old_candidate_promoted=False,
    conclusion='HistoricalN6 forward direction andnew Brownk1 specialcase agree after explicit+0.5transport. The new renderer does not erase its failed validation; no prior formula inversion bug found in thischeck.',
    historical_validation=summary,physical_quality_accepted=False)
(O/'historical-n6-bridge.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(passed=True,cameras=len(rows),points=len(points),max_abs_px=max(x['max_abs_px'] for x in rows),old_candidate_promoted=False)))
