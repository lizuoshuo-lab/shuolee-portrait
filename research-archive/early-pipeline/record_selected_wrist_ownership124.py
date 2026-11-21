"""Save actual native-image ownership decisions without modifying observations."""
from datetime import datetime,timezone
import audit_gaussianavatars_params123 as a

OUT=a.REC/'cross-subject-sh0-124/selected-wrist-ownership124'


def main():
    assert not a.lp(OUT/'visual-review.json').exists()
    data=a.read(OUT/'result.json');records=[]
    for r in data['records']:
        assert a.sha(r['path'])==r['sha256']
        rejected=r['frame']=='frame_000085'
        reason=('Only the near raised hand is clearly visible at the chest. Both left and right detector hand clusters occupy it; the far left wrist is not independently visible at its claimed chest location. Its confidence is not visibility evidence. The hidden left-hand gesture is unknown, not manually restored to a downward pose.' if rejected else
            'The raised right hand and lowered left hand have separately visible source support. Small detector/mesh misalignment remains; ownership support does not accept the fitted depth or every finger target.')
        records.append(dict(frame=r['frame'],source_image_sha256=r['source_sha256'],
            evidence=r['path'],evidence_sha256=r['sha256'],actual_native_image_review=True,
            input_ownership='unresolved_occluded_left_hand' if rejected else 'separate_visible_hand_support',
            usable_for_same_gesture_input_control=not rejected,reason=reason,
            source_wrist_distance_px=r['wrist_distance_px'],fit_pose_accepted=False))
    result=dict(updated=datetime.now(timezone.utc).isoformat(),state='source_ownership_review_completed',
        records=records,rejected_input_frames=[r['frame'] for r in records if not r['usable_for_same_gesture_input_control']],
        retained_input_frames=[r['frame'] for r in records if r['usable_for_same_gesture_input_control']],
        difference_from_alias_guard='Same anatomical-wrist consistency can pass when both estimates point to the wrong visible hand. This requires visibility/ownership review, not a larger alias-gap threshold.',
        selection_basis='Native source support, not ordering frames by postfit reprojection error. Whole-frame input control, no point-confidence or label edits. Genuine hand contact cannot be rejected from small2Ddistance alone.',
        physical_3d_truth=False,hidden_hand_target_invented=False,original_inputs_unchanged=True,
        new_fit_calls=0,new_gaussian_steps=0,
        next='One from-scratch seven-view input-control fit may exclude the unresolved85frame. Keep the just-completed shared-beta candidate as the frozen8view comparator, including original seed,weights,K,body-sharing and optimizer; no reuse of its optimized pose/beta.')
    a.save(OUT/'visual-review.json',result);a.save(OUT/'conclusion.json',result)
    print('OWNERSHIP_REVIEW_RETAIN7_EXCLUDE_UNRESOLVED85')


if __name__=='__main__':main()
