"""Record the completed, actually reviewed control; no fitting or model promotion."""
from datetime import datetime, timezone
from pathlib import Path
import json
import re
import subprocess
import numpy as np
import audit_gaussianavatars_params123 as a
from audit_smplifyx_temporal_result123 import rotations, distances


def main():
    now = datetime.now(timezone.utc).isoformat()
    body = a.REC/'demo-vhap-initialization/fullbody-smplifyx'
    root = body/'face-priority-guarded-sequence'
    result = a.read(root/'result.json')
    criteria = a.read(root/'review/criteria-audit.json')
    selection = a.read(root/'selection-audit.json')
    preflight = a.read(body/'face-priority-preflight.json')
    assert result['frames'] == criteria['frames'] == selection['checked_frames'] == 59
    assert criteria['all59_protections_passed'] and criteria['all_six_elbows_halved']
    assert preflight['source_sha256'] == a.sha(a.H/'run_smplifyx_face_priority123.py')
    assert result['plan_sha256'] == a.sha(body/'face-priority-guarded-sequence-plan.json')
    assert result['fit_sha256'] == a.sha(root/'fit/fitted.npz')
    handoff = a.read(body/'face-priority-guarded-sequence-gpu-handoff.json')
    assert handoff['state'] == 'resumed' and handoff['control_finished']
    probe = "import pathlib,json; p=pathlib.Path('/proc/331'); print(json.dumps(dict(pid=331,identity=p.joinpath('stat').read_text().split()[21],command=p.joinpath('cmdline').read_bytes().replace(b'\\0',b' ').decode(),state=p.joinpath('stat').read_text().split()[2])))"
    proc = subprocess.run(['wsl','-d','Ubuntu-20.04','--','/usr/bin/python3','-c',probe],capture_output=True,check=True)
    live = json.loads(proc.stdout.decode('utf-8'))
    assert live['identity'] == handoff['identity'] and live['state'] != 'T'
    assert 'train_demo_body_upstream_offset123.py train' in live['command']
    a.save(root/'closeout-live-process.json',dict(updated=now,baseline=live,control_worker_finished=True))

    visual_path = root/'review/visual-review.json'
    visual = a.read(visual_path)
    original = root/'review/visual-review-through029.json'
    if not a.lp(original).exists():
        assert visual['reviewed_frame_count'] == 29
        a.lp(original).write_bytes(a.lp(visual_path).read_bytes())
    existing = {r['frame'] for r in visual['frames']}
    for n in range(30,60):
        frame = f'frame_{n:06d}'
        review = a.read(root/'review/frames'/f'{frame}.json')
        for item in review['files']:
            local = Path(item['record_path'].replace('/mnt/d/','D:/'))
            assert a.sha(local) == item['sha256']
        if frame in existing:
            continue
        note = ('Original/native body, face, both hands and feet were inspected. '
                'No new visible reversal or collapse versus the independent baseline was seen. '
                'The bare mesh remains inside loose clothing; cap and boot surfaces are not fitted. '
                'The pocket hand is occluded, so its articulation is not validated by appearance.')
        if n in (40,51):
            note += ' The selected raised-arm direction visibly improves on the independent baseline.'
        if n in (51,59):
            note += ' Native face panels show no obvious new deformation; numerical face protection also passes.'
        visual['frames'].append(dict(frame=frame,regions_seen=['body','face','left_hand','right_hand','feet'],
            images=review['files'],findings=note,original_observation_error_not_ground_truth=True))
    seen = {r['path'] for r in visual.get('native_sheets_seen',[])}
    names = [f'frames_{n:03d}_{n+4:03d}_face_hands_feet_native.png' for n in (30,35,40,47,52)]
    names += ['frames_045_046_face_hands_feet_native.png','frames_057_059_face_hands_feet_native.png']
    names += [f'frames_{n:03d}_{n+1:03d}_body_native.png' for n in range(30,60,2)]
    for name in names:
        p = root/'review/native-sheets'/name
        if str(p) not in seen:
            visual.setdefault('native_sheets_seen',[]).append(dict(path=str(p),sha256=a.sha(p)))
    assert len(visual['frames']) == 59 and len({r['frame'] for r in visual['frames']}) == 59
    visual.update(updated=now,state='all59_actual_visual_review_complete',actual_visual_review=True,
        reviewed_frame_count=59,reviewed_image_groups=295,no_new_visible_reversal_or_collapse=True,
        controlled_change_passed=True,full59_quality_passed=False,quality_accepted=False,goal_complete=False,
        remaining_findings=['The inherited pocket-hand protrusion around frame12 is unresolved.',
            'Hidden wrist articulation remains unstable and is not validated by 2D residuals.',
            'Clothing, cap and shoes need separate surface fitting; bare-body alignment is not surface acceptance.'])
    a.save(visual_path,visual)

    # All 58 adjacent SO3 comparisons, preserving the original saved poses.
    joint_rows = []
    for n in range(2,60):
        with np.load(a.lp(root/f'frames/frame_{n-1:06d}/winner-camera-space.npz'),allow_pickle=False) as x, np.load(a.lp(root/f'frames/frame_{n:06d}/winner-camera-space.npz'),allow_pickle=False) as y:
            angles = distances(rotations(x['body_pose_axis_angle']),rotations(y['body_pose_axis_angle']))
            j = int(angles.argmax())
            joint_rows.append(dict(previous=f'frame_{n-1:06d}',frame=f'frame_{n:06d}',
                body_joint_index_excluding_root=j,angle_deg=float(angles[j]),
                joint='right_wrist' if j == 20 else 'other'))
    maxima = {side:{key:max(p[side][key] for p in criteria['adjacent_pairs'])
        for key in ('root_deg','local_joint_max_deg','left_elbow_deg','right_elbow_deg')}
        for side in ('baseline','candidate')}
    a.save(root/'review/adjacent-joint-identity.json',dict(updated=now,pairs=joint_rows,
        maxima=maxima,coordinate_scope='Camera-relative root, not recovered world camera motion.',
        no_pose_modified=True,quality_accepted=False))

    # Read-only handoff check, using the existing input metadata and all saved winners.
    inputs = a.read(body/'full-sequence/data/training-inputs.json')['records']
    assert len(inputs) == 59
    bridge_rows = []
    beta = faces = None
    for row in inputs:
        p = root/'frames'/row['frame']/'winner-camera-space.npz'
        with np.load(a.lp(p),allow_pickle=False) as fit:
            assert str(fit['frame_id']) == row['frame']
            assert np.array_equal(fit['K_integer_centers'],row['K_integer_centers'])
            assert np.array_equal(fit['w2c'],np.eye(4))
            assert str(fit['image_sha256']) == row['source_sha256']
            assert str(fit['mask_sha256']) == row['mask_sha256']
            if beta is None:
                beta,faces = fit['betas'].copy(),fit['faces'].copy()
            assert np.array_equal(beta,fit['betas']) and np.array_equal(faces,fit['faces'])
            for key in ('pose_embedding','camera_translation','global_orient','jaw_pose','leye_pose','reye_pose',
                        'expression','left_hand_pose','right_hand_pose'):
                assert np.isfinite(fit[key]).all()
        bridge_rows.append(dict(frame=row['frame'],saved_fit_sha256=a.sha(p),identity_contract_passed=True))
    a.save(root/'vhap-input-contract.json',dict(updated=now,state='all59_read_only_identity_contract_passed',frames=bridge_rows,
        upstream_bridge=str(a.H/'fit_smplify_vhap_surface123.py'),upstream_bridge_sha256=a.sha(a.H/'fit_smplify_vhap_surface123.py'),
        source_fit_root=str(root),input_metadata=str(body/'full-sequence/data/training-inputs.json'),
        action_needed='Existing bridge reads metadata under SOURCE/data. Separate the metadata path from the new fit path using a minimal input adapter; then verify the GPU forward projection before any optimizer update.',
        vertex_forward_parity_executed=False,renderer_parity_executed=False,optimizer_updates=0,
        no_input_copied_or_modified=True,quality_accepted=False))
    endpoints = []
    for n in (51,59):
        row = a.read(root/f'review/frames/frame_{n:06d}.json')
        prior = a.read(body/f'body-weight-guarded-sequence/review/frames/frame_{n:06d}.json')
        endpoints.append(dict(frame=n,baseline_face_median_px=row['baseline']['groups']['face51']['median'],
            previous_body2_face_median_px=prior['candidate']['groups']['face51']['median'],
            candidate_face_median_px=row['candidate']['groups']['face51']['median']))
    retained = sum(r['selected_index'] == 1 for r in selection['rows'])
    closeout = dict(updated=now,state='controlled_change_passed_geometry_not_accepted',
        controlled_change_passed=True,frames=59,actual_visual_reviewed_groups=295,
        regional_protections_passed=True,six_elbows_halved=True,selected_new=59-retained,
        retained_independent=retained,endpoint_face_protection=endpoints,
        adjacent_so3_maxima=maxima,incremental_fit_seconds=result['elapsed_seconds'],
        independent_cache_fit_seconds=criteria['independent_cache_fit_seconds'],fresh_video_to_export_timing=False,
        source_sha256=a.sha(a.H/'run_smplifyx_face_priority123.py'),plan_sha256=result['plan_sha256'],fit_sha256=result['fit_sha256'],
        visual_review_sha256=a.sha(visual_path),criteria_sha256=a.sha(root/'review/criteria-audit.json'),
        all_frame_anatomical_truth_accepted=False,clothing_surface_accepted=False,gaussian_admitted=False,
        automatic_followup=False,new_gaussian_started=False,baseline_resumed=live,quality_accepted=False,goal_complete=False,
        next='Prepare minimal existing VHAP input adapter and zero-update forward/projection parity. Preserve all59 frames and shared shape. Register any surface fitting separately with frozen pose protection; do not use low RGB loss to approve geometry or launch new Gaussians.')
    a.save(root/'closeout.json',closeout)
    print(json.dumps({k:closeout[k] for k in ('state','frames','selected_new','retained_independent','endpoint_face_protection','adjacent_so3_maxima','incremental_fit_seconds')},ensure_ascii=False))


if __name__ == '__main__':
    main()
