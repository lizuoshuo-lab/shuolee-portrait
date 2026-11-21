"""Read-only full-roster arm observations and pose diagnostics; no rejection/filter."""
from datetime import datetime, timezone
import time
import numpy as np
import audit_gaussianavatars_params123 as a
from smplify_score_contract123 import relative_confidence

started = time.monotonic()
root = a.REC / 'cross-subject-sh0-124'
run = root / 'grass-temporal-refinement'
out = run / 'occluded-arm-audit'
assert a.read(run/'temporal-progress.json')['state'] == 'completed'
assert not a.lp(out/'plan.json').exists(), 'One read-only audit; do not rerun.'
inputs = root/'grass-input/data/inputs.json'
observations = root/'grass-input/data/observations.npz'
sources = {str(p): a.sha(p) for p in [inputs, observations]}
plan = dict(updated=datetime.now(timezone.utc).isoformat(), frames=94, optimizer_steps=0,
    hypothesis='Far-arm observations may change identity under occlusion; temporal smoothness alone cannot prove correct depth.',
    selection='All94 frames. Follow-up raw inspection of temporal transition is post-hoc explanatory evidence, not independent validation.',
    limits='No coordinates, scores, fits, masks or model selection changed. Wrist proximity alone does not prove duplicate identity; genuine hand contact is possible.',
    cpu_budget_seconds=60, sources=sources)
a.save(out/'plan.json',plan)
rows=a.read(inputs)['records']
with np.load(a.lp(observations),allow_pickle=False) as z:
    points=z['keypoints'];scores=z['scores'];boxes=z['boxes'];flip=z['flip_disagreement']
confidence,_=relative_confidence(scores,boxes,flip)
records=[]
for i,row in enumerate(rows):
    assert time.monotonic()-started < 60
    path=run/'official-fit/frames'/row['frame']/'winner-camera-space.npz'
    with np.load(a.lp(path),allow_pickle=False) as z:
        joints=z['joints_camera'][0]
        projected=z['joints_projected_native'][0]
    height=float(boxes[i,3]-boxes[i,1])
    records.append(dict(frame=row['frame'],time_seconds=row['time_seconds'],
        raw_image=row['image'],raw_image_sha256=row['source_sha256'],fit_sha256=a.sha(path),
        left_xy=points[i,[5,7,9]].tolist(),right_xy=points[i,[6,8,10]].tolist(),
        left_confidence=confidence[i,[5,7,9]].tolist(),right_confidence=confidence[i,[6,8,10]].tolist(),
        left_flip_delta_px=flip[i,[5,7,9]].tolist(),
        wrist_pair_distance_px=float(np.linalg.norm(points[i,9]-points[i,10])),
        wrist_pair_distance_per_box_height=float(np.linalg.norm(points[i,9]-points[i,10])/height),
        hand_centroid_distance_px=float(np.linalg.norm(points[i,91:112].mean(0)-points[i,112:133].mean(0))),
        left_hand_mean_confidence=float(confidence[i,91:112].mean()),
        right_hand_mean_confidence=float(confidence[i,112:133].mean()),
        left_mesh_shoulder_elbow_wrist_camera=joints[[5,6,7]].tolist(),
        right_mesh_shoulder_elbow_wrist_camera=joints[[2,3,4]].tolist(),
        left_mesh_projected_native=projected[[5,6,7]].tolist(),
        right_mesh_projected_native=projected[[2,3,4]].tolist()))
for path,digest in sources.items():assert a.sha(path)==digest
a.save(out/'result.json',dict(updated=datetime.now(timezone.utc).isoformat(),frames=94,
    elapsed_seconds=time.monotonic()-started,records=records,optimizer_steps=0,
    source_hashes_unchanged=True,observation_filter_applied=False,
    physical_3d_ground_truth=False,quality_accepted=False,
    caveat=plan['limits']))
print('READ_ONLY_ARM_OBSERVATIONS_READY94; NO_FILTER_OR_NEW_FIT')
