"""Record the18 actual native boundary reviews without claiming visibility truth."""
from datetime import datetime,timezone
import audit_gaussianavatars_params123 as a

out=a.REC/'cross-subject-sh0-124/hand-identity-guard124'
assert not a.lp(out/'visual-review.json').exists()
data=a.read(out/'boundary-review-images.json');assert len(data['images'])==18
notes={
 'grass':{
  '000073':'Unchanged predecessor: left fingers are already scattered on garment/occluded arm. No claim that all hidden-hand errors are caught.',
  '000074':'Ambiguous orange left chain and red left body wrist withheld; visible right chest hand preserved. Entry has native evidence.',
  '000084':'Midpoint: duplicates on visible right hand removed, right remains.',
  '000094':'Last frame: hidden left targets still duplicate visible right hand; quarantine remains appropriate as uncertainty, no3Dlocation inferred.'},
 'winterhat':{
  '000096':'Missed collision precedes entry: orange left fingers already duplicate visible right hand. This remains an explicit false negative, not fixed by this rule.',
  '000097':'Entry removes hidden left chain while preserving visible right chest hand.',
  '000104':'Midpoint removes duplicate left chain and low stray point; visible right hand remains.',
  '000112':'Last frame still lacks visible far left hand; retained quarantine is uncertainty rather than inferred pose.'},
 'indoor':{
  '000019':'Missed collision before first entry: orange left fingers on visible right hip hand, red wrist on shirt. False negative retained.',
  '000020':'Entry removes unsupported far left chain, visible right hand preserved.',
  '000026':'Midpoint still shows only right near hand. Duplicate orange chain removed.',
  '000032':'Last quarantined frame: orange points spread over forearm/clothing, not a visible left hand.',
  '000033':'Geometric recovery test clears, but orange left fingers remain on near forearm/garment. Separation plus wrist consistency does not establish visibility. Recovery is NOT quality-accepted.',
  '000077':'Before second entry, cyan right hand duplicates near left watch hand and blue wrist is on forearm. Missed ambiguous predecessor.',
  '000078':'Right-chain entry removes unsupported cyan targets; visible left watch hand preserved.',
  '000088':'Midpoint: both detections share near left hand; quarantine retains visible left only.',
  '000098':'Last: right chain duplicates near left hand; removed cyan targets remain unsupported.',
  '000099':'Recovery shows a small distinct far-hand region at torso edge; finger accuracy/occlusion remains uncertain. Not independent visibility truth.'}}
records=[]
for item in data['images']:
    subject=item['subject'];key=item['frame'][-6:];path=item['path']
    win=path[5].upper()+':/'+path[7:];assert a.sha(win)==item['sha256']
    records.append(dict(**item,actual_viewed=True,note=notes[subject][key]))
stamp=datetime.now(timezone.utc).isoformat()
a.save(out/'visual-review.json',dict(updated=stamp,state='partial_identity_guard_reviewed',
    actual_selected_native_images_reviewed=18,all_suppressed_observations_individually_reviewed=False,
    original_contact_negatives_reviewed_in='../cross-hand-identity-audit/visual-review.json',
    no_new_observation_coordinates=True,default_adopted=False,quality_accepted=False,
    limitations=['False negatives at winterhat96 and indoor19/77.',
      'Indoor33 is a false visibility recovery: separate internally consistent detections can still be on clothing.',
      'Guard identifies only one failure mode and is not a complete visibility/tracking solution.'],
    next_decision='Allow one separately registered grass causal control only: same subject-independent rule, no threshold change. All21grass flagged frames form the final74-94 interval; no unsupported recovery occurs there. Other subjects are not admitted by this review. Preserve the guard as experimental; do not adopt as a production visibility classifier.',
    grass_control_proposal_reviewable=True,other_subject_fitting_admitted=False,records=records))
print('HAND_IDENTITY18_REVIEWED_PARTIAL_NOT_DEFAULT')
