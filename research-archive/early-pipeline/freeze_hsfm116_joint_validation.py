from pathlib import Path
import datetime as dt,json,hashlib
import numpy as np
H=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/portrait-alignment-open-source-116/hsfm');J=H/'joint-diagnostic-reserved'
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(J/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
assert not (J/'frozen-validation.json').exists()
# All142 candidate triplets were inspected on the six original-image inventory sheets.
static=[3,4,5,6,7,8,9,10,12,13,14,31,32,34,35,36,37,38,39,40,41,43,44,45,46,47,48,50,62,63,64,65,66,67,68,69,70,73,74,75,76,77,78,79,81,82,85,86,105,106,112,117,118,119,124,126,129,130,131,133,134,135,137,138,139]
a=np.load(J/'validation-tracks.npz');b=np.load(J/'background-tracks.npz');fg=a['groups']!=0
frames=np.concatenate([a['frames'][fg],b['frames'][static]]);xy=np.concatenate([a['xy'][fg],b['xy'][static]]);groups=np.concatenate([a['groups'][fg],np.zeros(len(static),int)])
np.savez_compressed(J/'frozen-validation.npz',frames=frames,xy=xy,groups=groups)
labels=['background','face_lower','face_upper','torso','legs_shoes'];counts={n:dict(tracks=int((groups==i).sum()),blocks=sorted(set(int(x) for x in frames[groups==i,1]))) for i,n in enumerate(labels)}
save('validation-visual-review.json',dict(updated=now(),reviewed_images=[f'validation-review-{i}.jpg' for i in [1,2,3]]+[f'background-inventory-{i}.jpg' for i in range(1,7)],
    observations=['The original KLT bank retained35probes but background0; a pre-detection prior-bank exclusion repair increased new foreground coverage.',
                  'Repaired KLT background coverage remains6in1block; no camera fit was attempted from either inadequate bank.',
                  'One separately preregistered SIFTcycle bank has142matches, including unrelated people. Descriptor cycle consistency does not prove scene rigidity.',
                  'All142background candidates were visually inspected. The fixed listed65 static-object candidates exclude visible bystanders and uncertain reflection/foliage/monitor content.',
                  'Foreground probes include nose/cheek detail, ear region, jacket lettering, seams and shoes. Some image-defined torso bins contain trousers/hands; bins are fixed observation strata, not exact semantic part labels.',
                  'KLT and SIFT observations remain estimated correspondences. A small overview review cannot prove subpixel material identity or zero subject motion.'],
    static_background_ids=static,other_background_ids=[i for i in range(len(b['frames'])) if i not in static],
    selection_uses_camera_error=False,selected_before_any_new_fit=True,all_rejected_or_secondary_observations_retained=True))
save('frozen-validation.json',dict(updated=now(),state='frozen_before_camera_fit',coverage_passed=all(c['tracks']>=10 and len(c['blocks'])>=3 for c in counts.values()),
    counts=counts,sha256=sha(J/'frozen-validation.npz'),selection='Original-image semantic visibility only, no candidate residual. New background static subset is fixed before fit; all142also reported separately.',
    primary='New foreground tracks plus manually reviewed static-object background tracks',
    held='Middle frame held; endpoints triangulate one shared3Dpoint. Same bank for074/control/joint. Invalid depth/projection is a failure, never discarded.',
    dense_weight_mask_radius_processed_px=3,unseen_test=False,physical_GT=False,previous_bank_values_used_for_admission=False))
save('implementation-addendum.json',dict(updated=now(),reason='Source audit before optimizer implementation, not post-result tuning',
    upstream_main_shape_prior_weight=1.,function_default_shape_prior_weight=0.,chosen_shape_prior_weight=1.,
    shared_initial_body_registration_steps=200,scene_steps_each_arm=500,joint_body_pose_unfrozen_last100=True,
    dense_graph='Complete directed13-image156edges; reuse4native-smoke outputs after identity remap, infer remaining152once',
    pose_gauge='Transform both scene points and cameras into first raw100camera. Freeze firstcamera pose in both arms, record single initial scale from freshHMRcamera centers.',
    all_conditions_shared_except_human_term=True,body_is_shared_static=True,
    disclosure='Bounded local headless HSfM-SMPL diagnostic with disclosed initializer/detector/coordinate adaptations. No full-official-reproduction claim.'))
print(json.dumps(counts))
