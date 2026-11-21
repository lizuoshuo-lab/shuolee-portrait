from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
D=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video10-global-initializer/relative-graph')
assert not (D/'repair-readonly-pose/result.json').exists()
old=json.loads((D/'repair-readonly-pose/progress.json').read_text())
assert old['pairs_completed']==58
stamp=datetime.now(timezone.utc).isoformat()
failure=dict(updated=stamp,state='failed_final_serialization',previous_progress=old,
 error='PoseLib zero-translation failure pose was normalized; NaNs prevented final and failure serialization.',
 evidence='Saved tool result contained invalid value encountered in divide at line78, then ValueError: Out of range float values are not JSON compliant: nan at final pairs.json and failure writer. Process exited1; no result persisted.',
 computed_pairs=58,saved_pairs=0,estimator_settings_changed=False,training_started=False)
(D/'repair-readonly-pose/failure-record.json').write_text(json.dumps(failure,indent=2)+'\n')
(D/'repair-readonly-pose/progress.json').write_text(json.dumps(dict(updated=stamp,pid=None,state='failed_final_serialization',training_started=False,failure='failure-record.json'),indent=2)+'\n')
plan=dict(registered=stamp,previous_failure=failure['error'],
 fix='Reject finite/zero-baseline or fewer-than5inlier returned pose before normalization; retain raw returned R/t and inliers plus per-pair records. Failure status is written before potentially nonfinite results.',
 recomputation='All58same deterministic pair estimates once because the prior computations were not persisted. This is an implementation repair, not a new candidate; no seed,threshold,pair,K or feature changes.',
 script_sha256=hashlib.sha256((Path(__file__).parent/'diagnose_video10_relative_graph123.py').read_bytes()).hexdigest(),
 CPU_seconds=300,new_features=0,training=0)
(D/'repair-invalid-pose-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps(plan))
