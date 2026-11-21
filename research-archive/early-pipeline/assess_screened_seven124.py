"""Close the single source-ownership roster control; never starts a fit."""
from datetime import datetime, timezone
import numpy as np
from scipy.spatial.transform import Rotation
import audit_gaussianavatars_params123 as a
from select_quality_frames124 import POLICY
from quality_budget124 import select_budget
from screen_wrist_consistency124 import apply_to_candidates
from publish_screened_seven124 import OUT, state


def main():
    assert not a.lp(OUT/'conclusion.json').exists()
    stamp = datetime.now(timezone.utc).isoformat()
    root = OUT.parent
    result = a.read(OUT/'result.json')
    review = a.read(OUT/'review/result.json')
    ownership = a.read(root/'selected-wrist-ownership124/conclusion.json')
    evidence = {r['frame']: r for r in review['records']}
    inputs = {r['frame']: r for r in a.read(root/'grass-temporal-refinement/official-fit/data/training-inputs.json')['records']}
    ranked = {r['frame']: r for r in a.read(root/'grass/selection.json')['records']}
    digest = a.sha(OUT/'selected-fit.npz')
    assert digest == result['fit_sha256']
    with np.load(a.lp(OUT/'selected-fit.npz'), allow_pickle=False) as z:
        ids = z['frame_ids'].astype(str).tolist()
        obs, proj = z['observed_native_keypoints'].copy(), z['joints_projected_native'].copy()
        rots = Rotation.from_rotvec(z['global_orient']).as_matrix()
        assert z['betas'].shape == (1, 10) and z['body_pose_axis_angle'].shape == (1, 63)
    assert ids == a.read(OUT/'plan.json')['frames'] == ownership['retained_input_frames']
    with np.load(a.lp(root/'grass-selected-shape124/selected-fit.npz'), allow_pickle=False) as z:
        baseline_ids = z['frame_ids'].astype(str).tolist()
        slots = [baseline_ids.index(f) for f in ids]
        assert np.array_equal(obs, z['observed_native_keypoints'][slots])
        old_proj = z['joints_projected_native'][slots].copy()
    yaw = np.degrees(np.unwrap(np.arctan2(rots[:, 0, 2], rots[:, 2, 2])))
    body = np.asarray([0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11])
    joint_names = {0:'nose',2:'right_shoulder',3:'right_elbow',4:'right_wrist',5:'left_shoulder',6:'left_elbow',7:'left_wrist',9:'right_hip',10:'right_knee',11:'right_ankle',12:'left_hip',13:'left_knee',14:'left_ankle',15:'right_eye',16:'left_eye',17:'right_ear',18:'left_ear'}
    admitted = ['frame_000030','frame_000046','frame_000049']
    records, decisions, comparison = [], {}, []
    for i, f in enumerate(ids):
        for im in evidence[f]['images'].values():
            assert a.sha(im['path'].replace('/mnt/d/','D:/')) == im['sha256']
        height = inputs[f]['initial_box_xyxy'][3] - inputs[f]['initial_box_xyxy'][1]
        e = np.linalg.norm(proj[i] - obs[i,:,:2], axis=1)
        old_e = np.linalg.norm(old_proj[i] - obs[i,:,:2], axis=1)
        metrics, before, flags = {}, {}, []
        for name, idx in [('body',body),('face',np.arange(67,118)),('feet',np.arange(19,25))]:
            w = obs[i,idx,2]**2
            maximum = float(e[idx][w > 0].max())
            metrics[name+'_weighted_rms_px'] = float(np.sqrt(np.sum(w*e[idx]**2)/w.sum()))
            metrics[name+'_supported_max_px'] = maximum
            metrics[name+'_supported_max_over_height'] = maximum/height
            before[name+'_weighted_rms_px'] = float(np.sqrt(np.sum(w*old_e[idx]**2)/w.sum()))
            if name in ('body','feet') and maximum/height > POLICY[name+'_max_over_height']:
                flags.append(name+'_reprojection_over_existing3percent_height')
        # Explain existing rejections without changing either the fitter or gate.
        violations = [dict(index=int(j),name=joint_names[int(j)],error_px=float(e[j]),
                           ignored_by_original_fit=int(j) in (1,9,12))
                      for j in body if obs[i,j,2]**2 > 0 and e[j]/height > POLICY['body_max_over_height']]
        records.append(dict(frame=f,time_seconds=inputs[f]['time_seconds'],source_fit_sha256=digest,
            yaw_proxy_deg=float(yaw[i]),face_sharpness=ranked[f]['face_sharpness'],
            body_confidence=float(obs[i,body,2].mean()),person_box_height_px=height,
            screen_flags=flags,body_gate_violations=violations,**metrics))
        comparison.append(dict(frame=f,baseline=before,current={k:metrics[k] for k in before},
            body_gate_violations=violations))
        accepted = f in admitted
        decisions[f] = dict(status='accepted' if accepted else 'uncertain',depth_reviewed=accepted,
            reason=('Native body/face/feet and four synthetic sides reviewed. Raised right hand remains near the chest; no old long forward reach. Small face, fingers, ankle and shoe-shape residuals remain.' if accepted else
                    'Overview and native face/feet reviewed. Existing body projection gate still fails. No admission from lower total loss or source visibility alone.'),
            evidence=evidence[f]['images'],evidence_sha256=evidence[f]['images']['sides']['sha256'],
            acceptance_scope='Coarse visible pose only; not independent anatomical depth, exact chest contact, calibrated camera or high-quality surface acceptance.')
    wrist = a.read(root/'selected-observation-audit124/wrist-preselection/grass.json')
    records = apply_to_candidates(records,wrist['records'])
    selection = select_budget(records,decisions)
    assert selection['selected_frames'] == admitted
    ys = [r['yaw_proxy_deg'] for r in records if r['frame'] in admitted]
    selection.update(updated=stamp,policy=POLICY,thresholds_unchanged=True,no_framewise_fit_mixing=True,
        actual_visual_review=True,selected_yaw_proxy_span_deg=max(ys)-min(ys),duplicate_wrist_screen_applied=True)
    changes = {name:dict(improved=sum(r['current'][name+'_weighted_rms_px'] < r['baseline'][name+'_weighted_rms_px'] for r in comparison),
                        worsened=sum(r['current'][name+'_weighted_rms_px'] > r['baseline'][name+'_weighted_rms_px'] for r in comparison))
               for name in ('body','feet','face')}
    conclusion = dict(updated=stamp,state='completed_source_roster_control_no_coverage_gain',
        seconds=result['seconds'],source_fit_sha256=digest,source_ownership_sha256=a.sha(root/'selected-wrist-ownership124/conclusion.json'),
        frames_reviewed=ids,native_face_feet_reviewed=ids,overview_body_side_reviewed=ids,
        native_body_and_four_sides_reviewed=admitted,decisions=decisions,
        selected_frames=admitted,selected_count=len(admitted),selected_yaw_proxy_span_deg=selection['selected_yaw_proxy_span_deg'],
        comparison_on_identical_seven_observations=comparison,common_view_rms_changes=changes,
        excluded_input_frames=['frame_000085'],unchanged_targets_and_confidences=True,
        original_initialization_reused=True,optimized_baseline_not_used_as_seed=True,
        one_shared_beta=True,beta_delta_l2=result['beta_delta_l2'],
        finding='The source image supports excluding frame85 because its hidden left hand has no separate observed ownership. The one fresh seven-view control does not enlarge the pose-qualified set: the same30/46/49remain. Input cleaning is justified but is not a sufficient fix for the remaining side-view fit.',
        loss_comparison_warning='Total losses from eight and seven views are not a same-observation improvement test. Only the common seven-view metrics and images are compared here.',
        numeric_contract_note='Body gate still includes hips9/12 ignored by the original fit; the violation list identifies whether optimized points also fail. No point or gate was removed.',
        quality_accepted=False,gaussian_admitted=False,new_gaussian_steps=0,
        next='Freeze this control; no further roster, beta or weight sweeps. Inspect the saved per-joint rejection evidence to distinguish ignored-hip semantics from real shoulder/elbow misfit, without changing gates. Continue quality-first approximately8 selection across people, preserving depth/ownership review and coherent fit archives; no new GS from these three concentrated views.')
    for name,value in [('selection-reviewed.json',selection),('visual-review.json',conclusion),('conclusion.json',conclusion)]:
        a.save(OUT/name,value)
    a.save(OUT/'progress.json',dict(updated=stamp,state=conclusion['state'],pid=None,done=len(ids),total=len(ids)))
    summary='草地7帧输入归属对照133.28秒完成；第85帧因遮挡错归属排除，但仍只有30／46／49通过原筛选，未增加侧视覆盖。未开始新高斯。'
    state('source_screened_seven124_closed_no_coverage_gain',summary,conclusion['next'])
    print(dict(selected=admitted,yaw_span=selection['selected_yaw_proxy_span_deg'],common_view_rms_changes=changes,
               body_failures={r['frame']:r['body_gate_violations'] for r in records if r['body_gate_violations']}))


if __name__ == '__main__': main()
