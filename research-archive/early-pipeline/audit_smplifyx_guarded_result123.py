"""Read-only verification of the preregistered guarded-fit criteria."""
from datetime import datetime, timezone
import json
import numpy as np
import audit_gaussianavatars_params123 as a
from audit_smplifyx_temporal_result123 import rotations, distances


def main():
    body = a.REC/'demo-vhap-initialization/fullbody-smplifyx'
    root = body/'guarded-root-sequence'
    plan = a.read(body/'guarded-root-sequence-plan.json')
    selection = a.read(root/'selection-audit.json')
    verified = {r['frame']: r for r in selection['rows']}
    rows, pairs, loaded = [], [], {}
    for frame in plan['temporal_roster']:
        path = root/'review/frames'/f'{frame}.json'
        if not a.lp(path).exists():
            continue
        meta = a.read(path)
        assert frame in verified
        assert meta['candidate_sha256'] == verified[frame]['selected_npz_sha256']
        before, after = meta['baseline'], meta['candidate']
        protections = {}
        for group, tolerance in [('body17', 1.), ('feet6', 1.), ('face51', .5)]:
            delta = after['groups'][group]['median'] - before['groups'][group]['median']
            protections[group] = dict(delta_px=delta, tolerance_px=tolerance, passed=delta <= tolerance)
        elbow = dict(before_px=before['left_elbow_px'], after_px=after['left_elbow_px'],
                     remaining_ratio=after['left_elbow_px']/before['left_elbow_px'])
        decision = a.read(root/'frames'/frame/'objective-selection.json')
        scores = decision['original_saved_objectives']
        selected = decision['selected_index']
        assert scores[selected] <= scores[1]
        frame_result = a.read(root/'frames'/frame/'result.json')
        rows.append(dict(frame=frame, protections=protections, left_elbow=elbow,
            selected_index=selected, selected_original_objective=scores[selected],
            independent_original_objective=scores[1], fit_seconds=frame_result['elapsed_seconds'],
            unchanged_camera_shape_topology=meta['unchanged_camera_shape_topology'],
            candidate_sha256=meta['candidate_sha256']))
        current = {}
        for label, folder in [('baseline', body/'full-sequence'), ('candidate', root)]:
            with np.load(a.lp(folder/'frames'/frame/'winner-camera-space.npz'), allow_pickle=False) as data:
                current[label] = dict(root=data['root_rotation'].copy(), body=data['body_pose_axis_angle'].copy())
        previous = f'frame_{int(frame.split("_")[-1])-1:06d}'
        if previous in loaded:
            pair = dict(previous=previous, frame=frame)
            for label, data in current.items():
                earlier = loaded[previous][label]
                angles = distances(rotations(data['body']), rotations(earlier['body']))
                pair[label] = dict(root_deg=float(distances(data['root'], earlier['root'])[0]),
                    local_joint_max_deg=float(angles.max()), left_elbow_deg=float(angles[17]),
                    right_elbow_deg=float(angles[18]))
            pairs.append(pair)
        loaded[frame] = current
    required = {f'frame_{i:06d}' for i in range(18, 24)}
    target_rows = [r for r in rows if r['frame'] in required]
    failures = [dict(frame=r['frame'], region=g, **v) for r in rows
                for g, v in r['protections'].items() if not v['passed']]
    baseline_result = a.read(body/'full-sequence/fit/official-fit-result.json')
    result = dict(updated=datetime.now(timezone.utc).isoformat(),
        state='all59_numerical_audit_complete' if len(rows) == 59 else 'partial_numerical_audit',
        frames=len(rows), total=59, pending=[f for f in plan['temporal_roster'] if f not in loaded],
        actual_saved_selection_checked=selection['checked_frames'], rows=rows, adjacent_pairs=pairs,
        protection_failures=failures, all59_protections_passed=(len(rows) == 59 and not failures),
        six_elbows_available=len(target_rows), six_elbows=target_rows,
        all_six_elbows_halved=(len(target_rows) == 6 and all(r['left_elbow']['remaining_ratio'] <= .5 for r in target_rows)),
        plan_sha256=a.sha(body/'guarded-root-sequence-plan.json'),
        independent_cache_fit_seconds=baseline_result.get('elapsed_seconds'),
        completed_frame_fit_seconds=sum(r['fit_seconds'] for r in rows),
        cache_excluded_from_incremental_time=True, fresh_video_to_export_timing=False,
        candidate_fit_modified=False, actual_visual_review=False, quality_accepted=False,
        goal_complete=False, Gaussian_training_started=False,
        limitations=['Original detector fitting residuals are not independent physical truth.',
            'SO3 roots are camera-relative; this is not a physical world-camera path.',
            'Complete original-image review is separate; numerical checks do not certify anatomy or clothing.'])
    a.save(root/'review/criteria-audit.json', result)
    print(json.dumps({k:result[k] for k in ('state','frames','protection_failures','six_elbows_available','all_six_elbows_halved')}))


if __name__ == '__main__':
    main()
