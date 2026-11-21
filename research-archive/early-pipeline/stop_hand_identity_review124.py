"""Preserve the user-corrected depth rejection and stop via the review gate."""
from datetime import datetime,timezone
import audit_gaussianavatars_params123 as a

root=a.REC/'cross-subject-sh0-124';out=root/'grass-temporal-handidentity'
request=a.read(out/'review-request.json');result=a.read(out/'review-013/result.json')
assert len(request['frames'])==13
assert not a.lp(out/'review-decision.json').exists()
stamp=datetime.now(timezone.utc).isoformat()
reason='User identified the actual side-view failure: raised hand is extended away from the chest, while the raw image shows it held close to the chest/hair. Being in front of the body and matching the previous fit are insufficient. Stop at13, do not continue27/94.'
visual=dict(updated=stamp,checkpoint_passed=False,quality_accepted=False,gaussian_admitted=False,
    actual_overview_pages_viewed=['page-01.png','page-02.png'],actual_all13_native_face_feet_review=False,
    decisive_user_review_image='C:/Users/ADMINI~1/AppData/Local/Temp/codex-clipboard-bc441f8b-6f45-47c1-8f1e-a44406e33112.png',
    reason=reason,parameter_parity_does_not_establish_quality=True,
    previous_intermediate_review_correction='Previous grass-refinement13/27 front-arm approvals were too permissive about hand-to-chest depth. Their saved records remain historical, and cannot serve as present quality approval.',
    latest_user_scope='Select about8well-fitted frames with approximately uniform observed view coverage, then train only those. All-frame success is no longer a prerequisite. Preserve continuous constraints as an optional fitting tool; do not spend every run repairing every frame.')
a.save(out/'review-013/visual-review.json',visual)
a.save(out/'review-decision.json',dict(updated=stamp,continue_same_candidate=False,
    reviewed_frames=request['frames'],review_result_sha256=a.sha(out/'review-013/result.json'),
    visual_review_sha256=a.sha(out/'review-013/visual-review.json'),actual_visual_review=True,
    reason=reason,quality_accepted=False,gaussian_admitted=False))
a.save(out/'conclusion.json',dict(updated=stamp,state='stopped_at13_user_depth_rejection',computed_frames=13,
    remaining_frames_not_run=81,reason=reason,quality_accepted=False,gaussian_admitted=False,
    guard_fit_effect_tested=False,guard_fit_effect_note='The confidence change starts at74, never reached. No claim that this fit evaluated or disproved the guard effect.',
    new_user_scope=visual['latest_user_scope'],next_action='Implement reproducible per-person quality-first/view-balanced~8frame selection from existing coherent fits. Retain exclusions, uncertainty and unselected records; do not restart this run or require all frames good. Candidate selected frames still need correct3Dpose, including supported hand-to-torso depth.'))
correction=root/'grass-temporal-refinement/review-correction-20261103.json'
a.save(correction,dict(updated=stamp,source=str(out/'review-013/visual-review.json'),reason=reason,
    supersedes_quality_inference_not_original_evidence=True,old_files_and_poses_preserved=True))
print('GRASS_HANDIDENTITY_STOP_REQUESTED_AT13')
