"""Record actual image review without changing the rejected fit or its inputs."""
from pathlib import Path
from datetime import datetime, timezone
import audit_gaussianavatars_params123 as a

def now():
    return datetime.now(timezone.utc).isoformat()

root=a.REC/'cross-subject-sh0-124/unique-wrist-input124/sofa16-control'
assert not a.lp(root/'conclusion.json').exists()
r=a.read(root/'result.json');v=a.read(root/'visual-review.json')
assert not v['actual_visual_review'] and not r['comparative_gate_passed']
assert all(r['interface_checks'].values())
v.update(updated=now(),actual_visual_review=True,state='reviewed_rejected',
    quality_accepted=False,full_sequence_admitted=False,gaussian_admitted=False,
    actual_review_notes=dict(
        body='Actual original/old/new native comparison viewed. The left knee is visibly closer to the trouser knee and the sitting depth is improved; shoulders and upper-arm alignment remain off. Clothes are not bare-body joint ground truth.',
        face='Viewed original-scale face crop. Gross facial direction is preserved and measured average improves slightly; facial skin surface accuracy and subpixel truth are not established.',
        hands='Viewed both native hands, including the unused root observations in quantitative scoring. Average improves, but fingers and wrists do not perfectly cover the visible hands.',
        feet='Viewed both native foot crops. Average decreases slightly; shoe shape and barefoot mesh differ and do not constitute full clothing-surface acceptance.'),
    conclusion='Only duplicate hand-root confidence changed. Knee and all regional average metrics improve, but both shoulders and right elbow regress beyond the registered1px per-joint guard. Reject promotion; do not omit these joints or change the gate.')
a.save(root/'visual-review.json',v)
old=r['metrics']['saved_full_parameter_endpoint'];new=r['metrics']['unique_wrist_endpoint']
next_action=('The one registered unique-wrist frame16 fit is complete and rejected by protected shoulders/right elbow; do not rerun it, loosen its gate, reverse source priority, add joint freedoms or launch full sequence/GS. '
    'Its exact camera/shape/optimizer parity and stage-comparison prove a causal sensitivity: changing only two root confidences removes most knee drift, while shoulder regression still starts in final face/hand stage. '
    'Next use the saved original and unique-wrist stage3/4 states for ONE bounded CPU-only gradient/observation audit: separate body, face,40 fingers,duplicate roots,priors and temporal contributions to VPoser latent; inspect the native shoulder/elbow targets. '
    'Do not fit or change weights in that audit. Determine whether the residual conflict is target semantics/visibility, shared-body latent coupling or a frozen shape/camera limitation before proposing any additional candidate. '
    'Original2D detector labels are observations, not physical truth. All original data, rejected controls, demo SH0 viewer model and four-person objective remain preserved.')
summary='唯一手腕输入对照完成：只关闭两个重复手根目标，左膝101.23→32.81px，脸4.19→4.04、手6.52→6.44、脚11.44→10.80px；全体原始点仍参与验收。但左肩43.54→58.67px、右肩18.58→22.82px、右肘10.68→15.75px，未通过逐关节保护。原生全身、脸、手、脚均已实看，未推广为整段或高斯。'
a.save(root/'conclusion.json',dict(updated=now(),summary=summary,next_action=next_action,
    actual_visual_review=True,quality_accepted=False,full_sequence_admitted=False,gaussian_admitted=False,
    camera_shape_optimizer_parity=True,input_changed_indices=[[0,25,2],[0,46,2]],
    diagnostics_only=True,actual_fitter_calls=1,fit_seconds=r['fit_seconds'],
    metrics=dict(original=old,unique_wrist=new),gates=r['gates'],images=v['images'],
    stage_comparison=str(root/'stage-comparison.json'),
    causal_scope='One controlled frame, same seed and all settings; identifies effect of removing duplicate root targets, not a generally correct observation rule or unique physical reconstruction cause. No claim of independent facial1px quality.'))
print(summary)
