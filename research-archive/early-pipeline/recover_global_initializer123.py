"""Recover completed controls after the recorded progress-writer failure, without rerun."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
H=Path(__file__).parent
G=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video10-global-initializer')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
assert not (G/'result-completed.json').exists()
old=read(G/'result.json');plan=read(G/'plan.json');ad=read(G/'execution-admission.json')
assert old['state']=='failed' and "multiple values for keyword argument 'pid'" in old['error']
assert len(old['controls'])==4 and all(c['passed'] for c in old['controls'])
assert sha(H/'global_track_initializer123.py')==ad['implementation_sha256']
for p,h in plan['source_sha256'].items():assert sha(Path(p))==h
assert old['elapsed_seconds']<plan['budget']['CPU_seconds']
with np.load(G/'synthetic-reconstruction.npz',allow_pickle=False) as a:
    assert all(np.isfinite(a[k]).all() for k in a.files)
    pixel=float(np.max(np.linalg.norm(a['projection_uv']-a['observation_uv'],axis=1)))
    assert pixel==old['controls'][1]['reprojection_max_px']
    counts=dict(frames=len(a['frames']),fit_families=len(a['families']),observations=len(a['observation_ids']))
stamp=datetime.now(timezone.utc).isoformat()
result=dict(updated=stamp,state='four_synthetic_controls_passed_status_write_recovered',
    controls=old['controls'],elapsed_seconds=old['elapsed_seconds'],**counts,
    failure_preserved='result.json',failure_sha256=sha(G/'result.json'),
    recovery='All four controls and source hash recheck completed before final status serialization failed. Read persisted results; no estimator rerun.',
    plan_sha256=sha(G/'plan.json'),implementation_sha256=ad['implementation_sha256'],
    probe_original_sha256=ad['script_sha256'],probe_status_fix_sha256=sha(H/'probe_video10_global_initializer123.py'),
    reconstruction_sha256=sha(G/'synthetic-reconstruction.npz'),
    heldout_used=False,real_pixels_used=False,relative_rotations_supplied_from_synthetic_truth=True,
    real_camera_estimates=0,new_body_fit=False,training_started=False,quality_accepted=False,
    remaining='Real essential-matrix branch estimation, outlier robustness, physical parallax, complete real-frame registration and independent face/background/body validation remain unverified.')
save(G/'result-completed.json',result)
save(G/'progress.json',dict(updated=stamp,pid=None,state='complete_recovered',result='result-completed.json',original_failure='result.json',training_started=False,GPU=False))
print(json.dumps(dict(state=result['state'],**counts,controls=[{k:v for k,v in c.items() if k!='diagnostic'} for c in old['controls']]),ensure_ascii=False))
