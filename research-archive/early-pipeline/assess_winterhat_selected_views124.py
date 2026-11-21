"""Close the reviewed cross-person control without turning 2D scores into 3D truth."""
from datetime import datetime, timezone
import numpy as np
from scipy.spatial.transform import Rotation
import audit_gaussianavatars_params123 as a
from select_quality_frames124 import POLICY
from quality_budget124 import select_budget
from publish_winterhat_selected_views124 import OUT, state


def main():
    assert not a.lp(OUT/'conclusion.json').exists()
    stamp = datetime.now(timezone.utc).isoformat()
    root = OUT.parent
    result = a.read(OUT/'result.json')
    evidence = {r['frame']: r for r in a.read(OUT/'review/result.json')['records']}
    inputs = {r['frame']: r for r in a.read(root/'winterhat-temporal/official-fit/data/training-inputs.json')['records']}
    old = {r['frame']: r for r in a.read(root/'quality-first-eight124/winterhat/shortlist.json')['records']}
    digest = a.sha(OUT/'selected-fit.npz')
    assert digest == result['fit_sha256']
    with np.load(a.lp(OUT/'selected-fit.npz'), allow_pickle=False) as z:
        ids = z['frame_ids'].astype(str).tolist()
        observed = z['observed_native_keypoints'].copy()
        projected = z['joints_projected_native'].copy()
        rotations = Rotation.from_rotvec(z['global_orient']).as_matrix()
    assert ids == a.read(OUT/'plan.json')['frames']
    yaw = np.degrees(np.unwrap(np.arctan2(rotations[:, 0, 2], rotations[:, 2, 2])))
    body = np.array([0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11])
    records, decisions = [], {}
    for i, f in enumerate(ids):
        for item in evidence[f]['images'].values():
            assert a.sha(item['path'].replace('/mnt/d/', 'D:/')) == item['sha256']
        height = inputs[f]['initial_box_xyxy'][3] - inputs[f]['initial_box_xyxy'][1]
        errors = np.linalg.norm(projected[i] - observed[i, :, :2], axis=1)
        metrics, flags = {}, []
        for name, idx in [('body',body), ('face',np.arange(67,118)), ('feet',np.arange(19,25))]:
            weights = observed[i,idx,2]**2
            maximum = float(errors[idx][weights > 0].max())
            metrics[name+'_weighted_rms_px'] = float(np.sqrt(np.sum(weights*errors[idx]**2)/weights.sum()))
            metrics[name+'_supported_max_over_height'] = maximum/height
            if name in ('body','feet') and maximum/height > POLICY[name+'_max_over_height']:
                flags.append(name+'_reprojection_over_existing3percent_height')
        records.append(dict(frame=f, time_seconds=inputs[f]['time_seconds'], source_fit_sha256=digest,
            yaw_proxy_deg=float(yaw[i]), face_sharpness=old[f]['face_sharpness'],
            body_confidence=float(observed[i,body,2].mean()), person_box_height_px=height,
            screen_flags=flags, **metrics))
        reason = ('Native overlay improves the raised-arm configuration, but no supported whole-pose depth acceptance. '
                  'Recorded overview shows unresolved trunk inclination; camera pitch and body pose are not independently separated. '
                  'Skirt-occluded knees are unknown, not verified legs. Existing numerical gate also fails.')
        if f in ('frame_000056','frame_000057'):
            reason = ('Numerical screen passes and native face/feet/body overlays are broadly plausible. '
                      'Four synthetic directions still show a forward-inclined configuration; this alone cannot distinguish camera pitch from body pose. '
                      'Raised-hand contact/depth and skirt-hidden legs remain unverified. These two nearly identical views do not provide diverse coverage.')
        decisions[f] = dict(status='uncertain', reason=reason,
            depth_reviewed=f in ('frame_000056','frame_000057'),
            overview_reviewed=True, evidence=evidence[f]['images'],
            evidence_sha256=evidence[f]['images']['sides']['sha256'])
    selection = select_budget(records, decisions)
    assert selection['actual_count'] == 0
    numeric = [r['frame'] for r in records if not r['screen_flags']]
    assert numeric == ['frame_000056','frame_000057']
    public = (f"冬帽8视角拟合完成，用时{result['seconds']:.2f}秒；原先向侧面张开的手臂有所改善。"
              "二维筛选只有56／57两帧通过，视角几乎相同；侧面躯干倾斜与手部深度仍未确认，没有准入高斯。")
    next_action = ('Keep this fit closed. Audit the original observation mapping, confidence, ignored-joint semantics and '
                   'visibility for the existing grass/winterhat selected frames, read-only; no threshold deletion or new fitting. '
                   'Separate camera pitch from body pose before judging synthetic uprightness. Preserve the about8 good distributed '
                   'selected-frame scope, no requirement that all original frames pass. No automatic Gaussian launch.')
    conclusion = dict(updated=stamp, state='completed_shared_pose_improved_depth_unaccepted',
        seconds=result['seconds'], selected_frames=[], numeric_candidates=numeric,
        overview_body_side_reviewed=ids,
        native_body_reviewed=['frame_000013','frame_000056','frame_000057'],
        native_face_feet_and_four_sides_reviewed=['frame_000056','frame_000057'],
        source_fit_sha256=digest, decisions=decisions, public_summary=public,
        quality_accepted=False, gaussian_admitted=False, new_gaussian_steps=0,
        thresholds_unchanged=True, no_framewise_fit_mixing=True,
        no_independent_physical_3d_truth=True, next=next_action)
    selection.update(updated=stamp, policy=POLICY, thresholds_unchanged=True)
    for name, value in [('selection-reviewed.json',selection), ('visual-review.json',conclusion), ('conclusion.json',conclusion)]:
        a.save(OUT/name, value)
    a.save(OUT/'progress.json', dict(updated=stamp,state=conclusion['state'],pid=None,done=8,total=8))
    state('winterhat_selected_views124_closed_depth_unaccepted',public,next_action)
    print(public)


if __name__ == '__main__':
    main()
