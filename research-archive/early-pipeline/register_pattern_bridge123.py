"""Record actual native-crop review, then preregister one identity-graph control."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib

T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';REC=R.parent/'local-clothed-surface-123'
O=REC/'pattern-preflight/native-material';V=O/'visual-review'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
stamp=datetime.now(timezone.utc).isoformat()
assert not (O/'bridge-plan.json').exists(), 'Do not repeat this registered comparison.'
notes=[
'Head profile/hair and nose edge, coarse region consistent. A silhouette is not a fixed material point.',
'Head profile/hair and nose edge, coarse region consistent. Exact physical identity remains unknown.',
'Shirt leaf/sleeve print, coarse motif consistent; repeated print and folds remain ambiguous.',
'Shirt stem/print, coarse motif consistent; exact material identity unverified.',
'Repetitive carpet texture, physical identity uncertain.',
'Plant foliage, coarse structure consistent; rigidity, occlusion and exact identity unknown.',
'Face mouth/hair region, coarse position consistent; exact identity unverified.',
'First query is on shirt print; the other two queries are at the eye/eyelid. This is a physical correspondence contradiction.',
'Shirt branch print, coarse motif consistent; repeated pattern identity uncertain.',
'Arm/wrist/bracelet near cloth boundary, then print. Occlusion could explain appearance; no definite contradiction assigned.',
'Chair hardware/seat, coarse structure consistent; exact material point and occlusion unverified.',
'Chair hardware/seat, coarse structure consistent; person occlusion visible.',
'Face/nose/cheek region, coarse consistent; subpixel localization and material identity unverified.',
'Face/nose/cheek region, coarse consistent; exact physical point unverified.',
'Person ROI includes face/hair margin. ROI category is not a semantic material label.',
'Shirt/table boundary and low-texture fabric. Exact center identity uncertain.',
'Door/wall base corner, coarse structure consistent; last view has person occlusion.',
'Blinds/chair edge, repeated texture and partial occlusion leave identity uncertain.',
'Face/hair/nose region, coarse consistent; profile visibility changes.',
'Face/hair/nose region, coarse consistent; exact physical point unverified.',
'Hair crossing printed shirt, occlusion identity uncertain.',
'Hair crossing printed shirt, occlusion identity uncertain.',
'Chair arm against blinds, coarse structure consistent; exact identity unverified.',
'Floor/door rail near person. The query need not be on the person; no definite contradiction assigned.',
'Right-profile face/hair/nose edge, coarse consistent; contour material identity can change.',
'Right-profile face/hair/nose edge, coarse consistent; exact physical point unverified.',
'Printed sleeve motif, coarse consistent; exact identity unverified.',
'Wrist/watch/bracelet/skin, coarse consistent; exact identity unverified.',
'Repetitive carpet texture, physical identity uncertain.',
'Floor/metal door rail, coarse structure consistent; repeated texture identity uncertain.'
]
idx=read(V/'index.json');assert len(idx['rows'])==len(notes)==30
review=[]
for row,note in zip(idx['rows'],notes):
    review.append(dict(row=row['row'],family_id=row['family_id'],track_id=row['track_id'],split=row['split'],
        image=row['image'],observation=note,physical_contradiction=row['row']==7,
        physical_identity_certified=False,subpixel_certified=False,development_only=True))
save(V/'review.json',dict(updated=stamp,actual_saved_pixels_reviewed=True,
    reviewed_sheets=[f'materials_{i:02d}.png' for i in range(1,7)],
    individually_reviewed_rows=[7,9,15,23],rows=review,
    actual_all10_false_track_pixels_reviewed=True,all10_image='false_track_all10_native.png',
    review_selection='Frozen before completed matches. These30 reviewed families are development evidence from now on; none may count as independent validation of a rule informed by this review.',
    physical_face_HELD=0,face_quality_accepted=False,training_allowed=False))
tr=read(O/'false-track-trace.json')
for view in tr['views']:
    view['physical_label']='printed_shirt' if view['slot'] in [0,1] else 'eye_eyelid_region'
    view['actual_native_pixels_reviewed']=True
    view['exact_subpixel_identity_verified']=False
tr.update(physical_all10nodes_review_pending=False,actual_all10nodes_reviewed=stamp,
          finding='A single graph bridge merges shirt observations with eye-region observations. Whole-component triangle count does not protect this connection.',
          no_camera_or_fit_causal_claim='This confirmed bad track is HELD and has never entered camera or SMPL-X fitting. It proves an association weakness, not the unique cause of earlier fits.')
save(O/'false-track-trace-reviewed.json',tr)
plan=dict(registered=stamp,stage='one_topology_only_bridge_control',hypothesis='A single unsupported bridge can merge two physically different tracks despite mutual descriptors and cycles elsewhere in the component.',
    fixed=['All99 native images,396237 descriptors,464 pair results and original pixel coordinates',
           'Existing1.5px dedup,3px duplicate families and frozen FIT/HELD membership',
           'Existing quarantined families remain quarantined; no recovery of their nodes',
           'No intrinsics, extrinsics, pose solver, SMPL-X or training'],
    sole_variable='Within each previously unquarantined appearance component, quarantine every graph bridge from identity fusion. Record maximal components after removing bridges. Only components with at least3 distinct frames remain geometric candidates. This is a topological eligibility test, not physical truth.',
    preservation='All original nodes and edges remain in immutable input files. Every bridge and short component is written with its reason. Never reassign family or FIT/HELD when splitting a component.',
    controls=['Two cycles joined by one bridge separate','A true triangle remains intact','A chain has no eligible identity segment','Cycle edge order does not change partitions','Independent NetworkX bridge result matches custom Tarjan on every processed graph'],
    inputs={name:sha(O/name) for name in ['appearance-tracks.json','duplicate-families.json','component-index.npz','pair-hashes.json','feature-manifest.json']},
    review_input_sha256=sha(V/'index.json'),
    development_family_ids=sorted({x['family_id'] for x in idx['rows']}),
    leakage_policy='All30 visually inspected families are development-only for this change. Their original FIT/HELD labels remain fixed; HELD is never used to fit cameras. Report remaining unopened HELD separately without calling it physically verified.',
    evaluation='Re-evaluate the same30 queries, all99 frame ROI support and the known false bridge. Count support losses as well as preserved support. Do not select new attractive examples, tune a threshold, or delete by geometric error.',
    budget=dict(passes=1,CPU_seconds=600,GPUs=0,new_feature_or_matching_passes=0,new_downloads=0),
    stop='Preserve failure and stop on input hash change, split leakage, audit discrepancy, budget limit or absent support. No second trial or training admission based only on graph topology.')
save(O/'bridge-plan.json',plan)
for root in [R,REC]:
    q=read(root/'queue-state.json');q.update(updated=stamp,running=None,worker_pid=None,worker_command=None,
        current_stage='pattern_bridge_control_registered',training_running=False,local_GPU_task_active=False,
        next_plan=str(O/'bridge-plan.json'),latest_evidence=str(O/'false-track-trace-reviewed.json'),
        next='单次检查跨视角身份图中的桥接边，保留原始观测和整族划分。当前没有新相机、SMPL-X或高斯训练。')
    save(root/'queue-state.json',q)
print(json.dumps(dict(reviewed_rows=30,all10_false_track_reviewed=True,plan=str(O/'bridge-plan.json'),training=False),ensure_ascii=False))
