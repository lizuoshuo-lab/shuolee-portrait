"""Read-only diagnosis of N13 numerical parity failure; no fit or extraction."""
from pathlib import Path
import json, hashlib, numpy as np
from datetime import datetime,timezone
from shared_background122 import evaluate, project
I=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video-update-20261027')
D=I/'clip03-fresh-geometry';O=I/'material-geometry-consumer';G=I/'clip03-grid-coverage/array-cache-repair'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert (O/'failure.json').exists() and not (O/'parity-diagnosis.json').exists()
base=read(D/'result.json');K=np.array(base['K_integer_centers']);cams={c['frame']:np.array(c['w2c']) for c in base['cameras']}
support=read(I/'clip03-roof-review/reviewed-support.json')['tracks'];visual=read(G/'visual-index.json');review=read(G/'visual-review.json')
tracks=[t for t in support if t['split']=='heldout']+[visual['rows'][r['row']] for r in review['rows'] if r['reviewed_material']=='face_surface_coarse' and r['split']=='heldout']
reference=read(D/'validation.json')['final'];ref={(r['track'],r['target']):r for g in ['background','face_heldout'] for r in reference[g]['observations']}
old={f:E@np.linalg.inv(cams[216]) for f,E in cams.items()}
vals={label:evaluate(K,c,tracks) for label,c in [('rebased184',cams),('restored216',old)]}
rows=[]
for i,r in enumerate(vals['rebased184']['observations']):
    z=vals['restored216']['observations'][i];rr=ref[r['track'],r['target']]
    rows.append(dict(track=r['track'],frame=r['target'],family=r['family'],raw_observed=r['observed'],
        error184=r['error_px'],error216=z['error_px'],reference_error=rr['error_px'],
        delta184=r['error_px']-rr['error_px'],delta216=z['error_px']-rr['error_px'],
        image_shift184_vs216=float(np.linalg.norm(np.array(r['predicted'])-z['predicted']))))
# A fixed world point and its correctly transformed camera still project equally.
points=np.load(D/'diagnostic-structure.npz')['points'];E216=cams[216];point216=points@E216[:3,:3].T+E216[:3,3]
fixed_point_max=max(float(abs(project(K,cams[f],points)[0]-project(K,old[f],point216)[0]).max()) for f in cams)
out=dict(updated=datetime.now(timezone.utc).isoformat(),state='parity_failure_localized_to_retriangulation',targets=len(rows),
 max_error_difference184=max(abs(r['delta184']) for r in rows),max_error_difference_restored216=max(abs(r['delta216']) for r in rows),
 max_prediction_shift184_vs216=max(r['image_shift184_vs216'] for r in rows),fixed_point_projection_change_max_px=fixed_point_max,
 rows=rows,input_candidate_sha256=sha(D/'result.json'),no_parameter_update=True,no_training=True,
 interpretation='A rigid gauge change commutes with projection of a fixed point, but homogeneous algebraic DLT minimizes a coordinate-dependent normalized residual. Noisily retriangulated points therefore differ under a world-origin change. This is not evidence that outputK or physical correspondence is correct.')
(O/'parity-diagnosis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
