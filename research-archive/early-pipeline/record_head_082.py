import json, hashlib
from pathlib import Path
from datetime import datetime
import numpy as np

B=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001\shuolee')
O=B/'current/human-head-surface-initialization-082'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=read(O/'rigid_neutral-result.json');b=read(O/'shared_shape100-result.json')
gain=1-b['held_nose_mouth']['median']/a['held_nose_mouth']['median']
gates=dict(nose_mouth_gain_at_least_10pct=gain>=.1,
    upper_p90_no_more_than_2px_worse=b['held_upper']['p90']<=a['held_upper']['p90']+2,
    positive_depth=b['positive_all'],visibility_retained_95pct=b['held_visibility_retention']>=.95,
    identity_not_saturated=b['max_shape']<2.95)
images=[]
for arm in ['rigid_neutral','shared_shape100']:
    for item in read(O/(arm+'-review/manifest.json'))['records']:
        images.append(dict(file=arm+'-review/'+item['file'],sha256=item['sha256'],actually_viewed=True))
decision=dict(updated=datetime.now().astimezone().isoformat(),round='082',type='initialization geometry diagnostic',
    user_pipeline='Full original images determine cameras; a fresh human prior initializes the person Gaussians.',
    facts=['175 complete whole-image cameras remain fixed; no interpolation.',
      'The previous initializer used the fresh body only for bounding box and scale. Its points were sampled from the multi-mask hull, without mesh facial topology.',
      'A shared neutral FLAME alignment and a shared100dim identity were fitted independently from the same fresh neutral initialization. No former subject model loaded.',
      '140 source frames fitted;35 entire frames held out from initialization and optimization. Apparent jaw contours excluded. Visibility computed by ray-triangle intersection.',
      'All12 native-camera overlays were actually inspected. The fitted central face generally approaches the source but ears and lower neck remain mismatched in several views. Hair and glasses are not skin-surface ground truth.'],
    measurements=dict(rigid=a,candidate=b,nose_mouth_median_gain=gain),gates=gates,
    candidate_adopted=False,full_head_geometry_accepted=False,gaussian_updates=0,new_gaussian_ply=False,
    reason='Identity reaches2.96097, exceeding the preregistered2.95 saturation gate. The neck/collar interface is not accepted. Do not relax this gate or use the personalized candidate as training initialization.',
    cause_not_established='This does not prove that camera error or local motion is the sole cause of multiple facial surfaces.',
    next_experiment='Test only the initialization of central facial points on a freshly rigid-aligned neutral FLAME surface, versus the current hull. Keep ears, hair, neck, clothing and total200000point budget unchanged. No learned identity, mesh binding or skin-distance penalty. All fits and points must be independently recomputed. This tests initialization location, not a complete GaussianAvatars reproduction.',
    next_preflight='Define the central-face region in the official neutral template, replace only the same number of hull points in that region by area-sampled surface points, preserve all other coordinates exactly, inspect native-camera initial renders, verify masks/depth and continuity before Gaussian training.',
    no_weight_or_step_sweep=True,no_dynamic=True,no_glasses_branch=True,pc00_deployed=False,
    visual_review=images,quality_accepted=False,goal_complete=False)
save(O/'decision.json',decision)
save(O/'visual-review.json',dict(actual_visual_review=True,images=images,summary=decision['facts'][-1],neckline_accepted=False))
print(json.dumps(dict(round='082',gain=gain,gates=gates,candidate_adopted=False,actual_images_reviewed=len(images))))
