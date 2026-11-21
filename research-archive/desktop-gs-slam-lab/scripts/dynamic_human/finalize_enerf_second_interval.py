"""Record the completed paired experiment and actual 52-panel visual inspection."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib
import numpy as np
from PIL import Image

L=Path(__file__).resolve().parents[2]
R=L/'outputs/dynamic-human-enerf-second-interval-079'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,x:p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=read(R/'comparison/summary.json')
assert len(s['panels'])==52 and s['PNG_checks']==960
early=['camera04-frame000721-both_hands_and_objects.png','camera04-frame000721-both_person_feet.png','camera06-frame000721-both_hands_and_objects.png']
for f in early:
    assert np.array_equal(np.asarray(Image.open(R/'early-review'/f)),np.asarray(Image.open(R/'comparison'/f)))
for p in s['panels']:
    assert sha(R/'comparison'/p['file'])==p['sha256']
v=dict(status='52_actual_native_comparison_panels_visually_reviewed',full_person_preplanned=24,detail_preplanned=24,additional_worst_change=4,
    reviewed=s['panels'],early_review_pixel_equivalent=early,
    findings=[
        'Across all eight planned cameras, body silhouette, faces, garment graphics, broad folds and legs are recognizable. The new temporal basis provides modest changes rather than a qualitative recovery of every detail.',
        'Camera04/frame731 and frontal00/01/frame731 retain severe semi-transparent ghosting around the light-clothed person\'s opening hand and cow toy. Both branches fail this region despite sharp source photographs.',
        'Camera04/frame727/731 raised black shoe and dark person shoe edges remain soft/doubled. Cameras06/07/12 also show blurred laces and soles; leg/foot positions are broadly consistent.',
        'Cameras06/07 show wall-textured floaters between the people; camera12 and other views contain floor/shadow patches around the shoes. Mask contamination is a hypothesis requiring direct training-target inspection.',
        'Camera12 shows substantial person-person occlusion. The local time model does not resolve hidden hand/object surfaces.',
        'All four additional worst-change panels were inspected. No evidence supports discarding these views or reporting average improvement as complete human clarity.'
    ],video_visually_reviewed=False,clarity_accepted=False,goal_complete=False)
save(R/'visual-review.json',v)
rows=read(R/'comparison/records.json')['records']
camera_changes={c:float(np.mean([r['foreground_change_db'] for r in rows if r['camera']==c and r['split']!='train'])) for c in sorted(set(r['camera'] for r in rows))}
branches={b:read(R/b/'completion.json') for b in ['global_cubic','local_bspline']}
assert all(x['step']==15000 for x in branches.values())
out=dict(status='paired_second_interval_training_evaluation_and_visual_review_complete_clarity_not_accepted',
    scene='ENeRF-Outdoor actor5_6',frames=[720,732],reference_frame=726,
    training_images=105,actual_updates_per_branch=15000,branches=branches,
    native_predictions=240,evaluation_pairs=120,metric_png_checks=960,actual_panels_reviewed=52,
    metrics=s['splits'],temporal_difference_residual=s['temporal_residuals'],camera_mean_change_db=camera_changes,
    identical_initialization_and_sampling=s['identical_initialization_and_sampling'],checkpoints=s['checkpoints'],
    scope='Same-scene disjoint motion interval; one seed. Old camera clock estimates frozen, new body/head motion and canonical appearance initialized for reference726. No appearance checkpoint reused.',
    decision='Retain the local B-spline temporal extension as a development candidate: both evaluated intervals have positive mean changes. Do not replace the preserved formal reference or declare human clarity solved.',
    unresolved=['Opening hand and held-object geometry','Raised shoe/sole and fine garment detail','Foreground mask/background contamination diagnosis','Long sequences and independent scene generalization','Joint static large-scene reconstruction'],
    next='Inspect the actual training mattes and composited targets at the failed hand and background regions before another capacity/training increase.',
    retain_as_temporal_candidate=True,clarity_accepted=False,goal_complete=False,updated_at=datetime.now(timezone.utc).isoformat())
save(R/'final-summary.json',out)
s['status']='120pairs_independently_verified_52panels_reviewed_clarity_not_accepted'
s['visual_review']='../visual-review.json'
save(R/'comparison/summary.json',s)
p=read(R/'progress-summary.json');p.update(status=out['status'],next=out['next'],final_summary='final-summary.json',updated_at=out['updated_at'])
for b in p['branches']:p['branches'][b]['evaluation_complete']=True
save(R/'progress-summary.json',p)
m=read(L/'manifests/dynamic-human-work.json');m.update(status=out['status'],outdoor_second_interval='outputs/'+R.name+'/final-summary.json',current_runs=[],next_action=out['next'],updated_at=out['updated_at'])
m['current_temporal_candidate_validation']='outputs/'+R.name+'/final-summary.json'
entry='079: New720..732motion/reference726, two15000update trainings, 240native renders/120paired CPU metrics,960PNGs checked,52actual panels viewed. Mean foreground +0.37984dB known/+0.20760dB other cameras; opening hand/toys/shoes still fail. Same-scene disjoint interval, not independent scene.'
if entry not in m['completed']:m['completed'].append(entry)
m['goal_turn_classification']='Progress: second-interval matched training and independent evaluation complete; visual clarity remains unaccepted. Next directly inspect native training masks; no external blocker.'
save(L/'manifests/dynamic-human-work.json',m)
print(json.dumps(dict(status=out['status'],reviewed=52,camera_mean_change_db=camera_changes)))
