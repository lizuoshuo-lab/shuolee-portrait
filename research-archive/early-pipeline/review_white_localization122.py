"""Actual native-image review of frozen N12e local measurements."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
I=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video-update-20261027')
D=I/'clip03-localization'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (D/'visual-review.json').exists()
r=read(D/'result.json');m=read(D/'measurements.json')['rows'];v=read(D/'visual-index.json');p=read(I/'clip03-localization-plan.json')
assert sha(D/'measurements.json')==r['measurements_sha256'] and len(m)==83 and r['measured_pairs']==477
assert [x['row'] for x in v['rows']]==p['budget']['raw_review_rows']
face=[]
for t in m:
 if t['material_label']=='face_surface_coarse':
  q=list(t['pairs'].values());assert len(q)==6 and all(x['state']=='measured' for x in q)
  face.append(dict(row=t['row'],family=t['family'],split=t['split'],frames=[x['frame'] for x in t['observations']],
    max_SIFT_affine_disagreement_px=max(x['center_delta_norm_px'] for x in q),
    max_forward_backward_px=max(x['error_px'] for x in t['forward_backward']),max_cycle_px=max(x['error_px'] for x in t['cycle']),
    max_conditional_white_noise_sigma_px=max(x['conditional_one_sigma_major_px'] for x in q),
    localization_ground_truth_available=False))
review=dict(stage='N12e',updated=datetime.now(timezone.utc).isoformat(),state='localization_diagnosis_review_complete_no_camera',
 previous_goal_turn='progress: N12d changed selection evidence and completed actual UI/PPT publication',
 actual_visual_review=dict(rows=p['budget']['raw_review_rows'],pages=v['pages'],all_three_native_observations_reviewed=True,
  observations=['Rows52 and60 visibly change eyelid openness in184,200,216. Do not infer the tracked point itself moves by the reported amount, but local rigid brightness constancy is not exact.',
   'Rows55,56,57,59 contain actual nose/mouth/beard texture. Texture patches are not independent surveyed material landmarks.',
   'Rows18/19 show the same roof bay context. Row14 first observation reaches image boundary, as recorded by missing41px support.',
   'Row24 remains repeated glazing/reflected texture. Small affine closure cannot upgrade the prior uncertain material label.',
   'Rows9/27 are the previously reviewed stair/ground controls; native images and labels are unchanged.'],
  diagnostic_warp_images_displayed=False,annotations_in_images=False),
 complete_denominator=dict(rows=83,directed_pairs=498,measured=477,missing_border_support=10,ECC_nonconvergence=8,outside_local_support=3,all6_directions_measured_rows=78),
 face_per_family=face,
 findings=[
  'SIFT and fixed local affine measurements disagree on6face families: median0.780px over36directed pairs, max3.254px. This is a measured estimator disagreement, not an independently known SIFT error or proof the affine estimate is correct.',
  'Face forward/backward max0.998px and3viewclosure max0.252px under this local model. Allobservations are in just3frames184/200/216. This is self-consistency, not independent <1px 3D reprojection acceptance.',
  'Roof measured32of36directions: SIFT disagreement max0.806px, FBmax0.186px. Four directions lack native support. No failed direction is silently dropped from the denominator.',
  'Ambiguousglass24of24directions gives cyclemax0.152px despite unresolved surface identity. Therefore numerical consistency alone cannot certify physically fixed material.',
  'Under affine/white-noise assumptions the local face major-axis sigma is small, but excludes blink, reflectance, model mismatch and resampling correlation. It is not a calibrated physical uncertainty bound.',
  'This experiment fixes cameras and uses no 3D solution. It establishes a pixel-level measurement-model difference before camera solving. It cannot attribute oldhat3.1px camera residual to this whiteT data or uniquely separate motion from correspondence bias.'
 ],
 failure_accounting=[dict(row=t['row'],family=t['family'],split=t['split'],states={k:x['state'] for k,x in t['pairs'].items()}) for t in m if not t['all6_measured']],
 admission=dict(camera=False,training=False,quality=False,prior_nonground_material_gate='N12d7FIT/2heldout remains below6/3; localization results do not relax this.',
  independent_face_requirement='Not passed: no independent physical point/error ground truth and only6facefamilies across3diagnosticframes.'),
 next='Freeze N12e; no second window/threshold/ECC attempt. Before camera fitting, preregister one exhaustive review of the already-detected roof structural families in N12d, retaining their old splits and every negative/ambiguous case, to establish whether material-support failure is due to the representative review cap or actual missing independent points. This is review of existing observations only, not new detection/matching or residual-selected supplementation. Geometry remains gated. Keep photometric measurements diagnostic until physical identity and uncertainty can be supported.',
 hashes={n:sha(D/n) for n in ['result.json','measurements.json','visual-index.json']},plan_sha256=sha(I/'clip03-localization-plan.json'),
 original_observations_unchanged=True,K_changed=False,training_images_warped=False,new_cameras=0,training_started=False,physical_subpixel_accepted=False)
save(D/'visual-review.json',review)
save(D/'progress.json',dict(state=review['state'],pid=None,updated=review['updated'],next=review['next']))
print(json.dumps(dict(complete=True,face_families=len(face),measured_pairs=477,attempted_pairs=498,next=review['next']),ensure_ascii=False))
