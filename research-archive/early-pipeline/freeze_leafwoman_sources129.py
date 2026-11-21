"""Freeze one already-reviewed source-only fit pool; never fit or edit pixels."""
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import select_leafwoman_half12_sources129 as selector


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def main():
    out = selector.OUT
    target = out / 'plan.json'
    assert not target.exists(), 'Frozen source plan already exists; inspect, do not replace.'
    draft_path = out / 'plan-before-source-review.json'
    draft = read(draft_path)
    assert sha(selector.__file__) == '8571cffee9ba56a134bf246cdd318cca97720c8818bc92f0526cd6bc04806e1f'
    assert sha(draft_path) == '9e03b8a22b5d598cee4ebf421f6d5123b33296cf2a2b6577aac41eca9b6f031e'
    group_path = selector.SOURCE_REVIEW / 'pose-groups.json'
    assert sha(group_path) == '7b5774d1a58bad6775b40dbf628f7e714438aa0c68b0f789db792acafe926ac3'
    group = read(group_path)
    assert group['actual_all57_source_visual_review'] is True
    assert group['source_video_sha256'] == selector.VIDEO_SHA
    assert group['source_only_stable_episode_candidate_pool'] == [f'frame_{i:06d}' for i in range(1,41)]
    assert group['exact_static_3D_pose_guaranteed'] is False
    for entry in draft['source_review_reuse']:
        assert sha(entry['path']) == entry['sha256']
    for p, digest in draft['source_hashes'].items():
        assert sha(p) == digest, p
    fit_ids = sorted(draft['fit_candidate_frames'])
    held = sorted(s['candidate_order'][0] for s in draft['holdout_reservation'])
    reserves = {f for s in draft['holdout_reservation'] for f in s['candidate_order']}
    assert len(fit_ids) == 24 and len(held) == 4 and not set(fit_ids) & reserves
    assert set(fit_ids + held) <= set(group['source_only_stable_episode_candidate_pool'])
    source_rows = {r['frame']: r for r in draft['source_rows']}
    group_rows = {r['frame']: r for r in group['rows']}
    features = {r['frame']: r for r in draft['features']}
    ids = fit_ids + held
    p = copy.deepcopy(draft)
    p['face_visibility'] = {f: group['face_visibility'][f] for f in ids}
    p['joint_group_visibility'] = {f: group['joint_group_visibility'][f] for f in ids}
    rows = {}
    for f in ids:
        s = group_rows[f]
        assert s['source_pose_episode'] == 'early_chest_leaf'
        assert not s['source_mask_gross_defect_observed']
        r = copy.deepcopy(source_rows[f])
        r.update(face_region_is_observed=p['face_visibility'][f],
                 visible_eyes_open=group['visible_eyes_open'][f],
                 source_joint_group_supported=p['joint_group_visibility'][f],
                 source_visibility_review=str(group_path),
                 source_pose_episode='early_chest_leaf', foot_out_of_frame=True)
        rows[f] = r
    eligible_ref = [features[f] for f in fit_ids if p['face_visibility'][f] and group['visible_eyes_open'][f]]
    assert eligible_ref, 'No source-supported open-eye output reference.'
    face_rank = selector.extractor.rank([-r['face_sharpness'] for r in eligible_ref])
    body_rank = selector.extractor.rank([-r['raw_body_score'] for r in eligible_ref])
    ref_order = sorted(zip(eligible_ref, face_rank, body_rank), key=lambda x: (.6*x[1]+.4*x[2], x[0]['frame']))
    reference = ref_order[0][0]['frame']
    p.update(updated=datetime.now(timezone.utc).isoformat(), state='source_reviewed_fit24_pool_frozen_before_first_fit',
        training_frames=fit_ids, frames=fit_ids, heldout_frames=held, review_only_frames=held,
        rows=[rows[f] for f in fit_ids], heldout_rows=[rows[f] for f in held],
        reference_frame=reference, reference_order=[r[0]['frame'] for r in ref_order],
        reference_source_eyes_open=True, reference_is_source_only_label_not_initialization=True,
        training_roster_ready=True, actual_source_visual_review=True, source_mask_actual_visual_review=True,
        static_gesture_plausible=True,
        static_gesture_note='One continuous early chest-leaf episode1..40, broad arm gesture only; head/gaze/fingers vary. Exact static3D geometry is not certified.',
        source_rejected_frames=[f'frame_{i:06d}' for i in range(41,58)],
        source_pose_grouping=dict(actual_source_only_review=True, selected_training_pose_group='early_chest_leaf',
            selected_training_arms_stable=True, training_group_by_frame={f:'early_chest_leaf' for f in fit_ids},
            source_group_receipt=str(group_path), source_group_receipt_sha256=sha(group_path),
            exact_static_3D_pose_guaranteed=False),
        postfit_selection_protocol={k:selector.PROTOCOL[k] for k in ('postfit','postfit_measurement','side_review','reference','confidence','evidence_scope')},
        fit_candidate_count=24, gaussian_target_count=12, gaussian_roster_frozen=False,
        heldout_roles={f:dict(fit_excluded=True, training_statistics_excluded=True, Gaussian_selection_excluded=True,
            physical_joint_ground_truth=False, source_entrance_reviewed=True,
            source_joint_group_supported=p['joint_group_visibility'][f],
            scope='Fit-excluded source observations only; no fitted held pose, no current geometric or unseen-video validation.') for f in held},
        known_unobserved='Knees/feet under long skirt or beyond native frame; ignored by source-support gates, never blank negative supervision.',
        mask_limitations='Original mask retained. Leaf prop inclusion differs; hair/scarf fine edges uncertain. No mask edit or dilation.',
        official_fit_calls_allowed=1, new_gaussian_steps=0, fitting_started=False, training_started=False,
        gaussian_admitted=False, quality_accepted=False, end_to_end_15min_accepted=False,
        source_pixels_modified=False, source_masks_modified=False, old_optimized_pose_or_beta_used=False,
        old_fit_residuals_used_for_selection=False,
        record_rules='D:/00/记录/022 images only. Engineering JSON/logs stay under C experiment evidence.')
    p['source_hashes'].update({str(group_path):sha(group_path),str(Path(__file__)):sha(__file__)})
    for entry in draft['source_review_reuse']:
        p['source_hashes'][entry['path']] = entry['sha256']
    p['source_hashes'][str(draft_path)] = sha(draft_path)
    target.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(marker='LEAFWOMAN129_FIT24_HELD4_SOURCE_PLAN_FROZEN',sha256=sha(target),
                          fit24=fit_ids,held4=held,reference=reference,GS12_selected=False),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
