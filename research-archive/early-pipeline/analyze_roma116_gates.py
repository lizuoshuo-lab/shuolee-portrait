"""Decompose the unchanged preregistered gates; no threshold selection or rerun."""
from pathlib import Path
import datetime as dt
import json
import numpy as np
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/portrait-alignment-open-source-116/roma')
D=R/'smoke';reg=json.loads((R/'smoke-preregistration.json').read_text(encoding='utf8'));rows=[]
def quant(a):
    a=np.asarray(a);v=a[np.isfinite(a)]
    return dict(n=len(a),median=float(np.median(v)),p90=float(np.percentile(v,90))) if len(v) else dict(n=len(a))
for f in sorted(D.glob('triple-*.npz')):
    with np.load(f) as z:rows.append({k:z[k] for k in z.files})
keys=['probe_ids','groups','confidence','fb','cycle','valid','eligible','projection','points']
a={k:np.concatenate([r[k] for r in rows]) for k in keys}
valid=a['valid'];conf=a['confidence']>=reg['confidence_threshold'];fb=a['fb'].max(-1)<=reg['forward_backward_max_raw_px'];cycle=a['cycle']<=reg['three_view_cycle_max_raw_px']
assert np.array_equal(valid&conf&fb&cycle,a['eligible'])
error=np.linalg.norm(a['projection'][:,1]-a['points'][:,1],axis=-1)
result={}
for g,label in enumerate(['background','face_lower','face_upper','torso','legs_shoes']):
    m=a['groups']==g;geometric=m&valid&fb&cycle;approved=m&a['eligible']
    result[label]=dict(total=int(m.sum()),valid=int((m&valid).sum()),confidence_pass=int((m&conf).sum()),fb_pass=int((m&fb).sum()),cycle_pass=int((m&cycle).sum()),all_pass=int(approved.sum()),
        valid_fb_cycle_without_confidence=int(geometric.sum()),confidence_only_fail=int((geometric&~conf).sum()),confidence_distribution=quant(a['confidence'][m]),
        held_projection_descriptive_valid_fb_cycle_raw_px=quant(error[geometric]),
        caveat='The no-confidence subset is descriptive only. It is not a new acceptance threshold or a calibration candidate. No physical correspondence truth is implied.')
out=dict(updated=dt.datetime.now(dt.timezone.utc).isoformat(),thresholds_unchanged=True,new_gpu_inference=False,scope='Exact original473probes; decompose existing outputs only',regions=result,
    conclusion='Primary confidence threshold rejects many numerically consistent correspondences. Primary coverage gate fails; this is not evidence that all rejected matches are wrong.',
    root_cause_resolved=False)
(D/'gate-decomposition.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
