"""Record actual all-eight-view inspection; release only the matching live gate."""
from datetime import datetime, timezone
import os
import audit_gaussianavatars_params123 as a

out=a.REC/'cross-subject-sh0-124/winterhat-clock18k-cache124'
snap=out/'gaussian-run/step-001000'
now=datetime.now(timezone.utc).isoformat()
assert not (out/'review-1000-decision.json').exists()
request=a.read(out/'review-1000-request.json')
assert a.sha(snap/'world.ply')==request['world_ply_sha256']
geometry=a.read(out/'gaussian-run/geometry-step-001000.json')
assert geometry['new_relative_normal_reversals']==0
assert geometry['protected_face_motion_m']==0
assert geometry['minimum_area_ratio']>.1
assert geometry['max_added_offset_m']<.02
review=dict(updated=now,step=1000,actual_visual_review=True,
    overview_body_face_feet_reviewed=[13,16,41,56,57,71,96,109],
    native_crops_reviewed=['frame_000071 body','frame_000071 face','frame_000071 feet'],
    baseline_reviewed='winterhat-selected-sh0-pipeline124/gaussian-run/step-001000/review/body-01.png, face-01.png, feet-01.png',
    finding='All eight same-camera views inspected. Relative to the baseline at1000, skirt edge bubbles are reduced and hat boundaries appear less scattered. The jacket, bag, boots and face remain soft; red bag fragments and small isolated skirt splats remain. No new gross limb distortion or mesh fold was observed. The source-fit arm depth and feet alignment remain uncertain. This is a whole-clock-profile comparison, not attribution to one schedule field or a quality pass.',
    points=26618,geometry=geometry,quality_accepted=False,
    source_pose_quality_accepted=False,novel_views_reviewed=False,
    cold_video_timing_accepted=False,
    decision='continue_same_live_optimizer_to_registered18000')
a.save(snap/'visual-review.json',review)
decision=dict(updated=now,world_ply_sha256=request['world_ply_sha256'],
    decision='continue',review=str(snap/'visual-review.json'),
    same_live_optimizer=True,maximum_step=18000)
temp=out/'review-1000-decision.json.tmp'
a.save(temp,decision)
os.replace(temp,out/'review-1000-decision.json')
print('WINTERHAT_CLOCK1000_ACTUAL_REVIEW_CONTINUE')
