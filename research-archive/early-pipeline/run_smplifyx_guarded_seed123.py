"""Original per-frame solver, with a uniform original-objective fallback.

The existing independent fit cache is explicit and immutable. This is not a
fresh video-to-model timing run and never selects frames manually.
"""
import argparse
from pathlib import Path
import os
import signal
import subprocess
import sys
import time
import numpy as np
import run_smplifyx_temporal_root123 as temporal
import control_smplifyx_orientations123 as lease
import official_smplifyx_entry123 as entry

BODY, BASE = temporal.BODY, temporal.BASE
OUT = BODY/'guarded-root-sequence'
PLAN = BODY/'guarded-root-sequence-plan.json'
HANDOFF = BODY/'guarded-root-sequence-gpu-handoff.json'
AUDIT = BODY/'initialization-objective-audit/result.json'
read, save, sha, now = temporal.read, temporal.save, temporal.sha, temporal.now
ORIGINAL_TEMPORAL_SOURCE = Path(temporal.__file__).resolve()


def bind():
    temporal.OUT, temporal.PLAN, temporal.HANDOFF = OUT, PLAN, HANDOFF


def prepare():
    assert not OUT.exists() and not PLAN.exists()
    evidence = read(AUDIT)
    assert evidence['body_forwards'] == 124 and not evidence['changed_order_frames']
    reviewed = read(BODY/'temporal-root-sequence/review/closeout.json')
    assert reviewed['actual_reviewed_frames'] == 59 and not reviewed['quality_accepted']
    bind()
    temporal.prepare()
    p = read(PLAN)
    p['hypothesis'] = ('A seeded fit worse than the original independent candidate must not be propagated to the next frame. '
        'For every frame use the same original saved-state objective to compare the newly fitted temporal candidate with '
        'the verified immutable independent candidate; propagate only the lower objective result, keeping the independent candidate on a tie.')
    p['single_change_from_failed_full59'] = 'Uniform original-objective selection guard at each frame, before its result becomes the next seed.'
    p['difference_from_probe'] = 'All59 with within-run selected neighbors; cached independent candidate is compared uniformly, never hand-picked by frame.'
    p['cache_policy'] = dict(used=True, root=str(BASE), purpose='Uniform original independent candidate for every same-input frame.',
        audit=str(AUDIT), audit_sha256=sha(AUDIT), fresh_video_to_export_timing=False,
        new_video_requires_computing_independent_candidates=True,
        cache_reuse_is_not_15min_acceptance=True)
    p['selection_policy'] = dict(candidate0='new within-run temporal body/root fit', candidate1='unchanged original independent same-frame fit',
        rule='0 if recomputed_original_loss0 < recomputed_original_loss1 else 1',
        residual_filtering=False, visual_frame_selection=False, manual_overrides=False,
        original_five_stage_solver_and_optimizer_unchanged=True)
    p['no_best_result_stitching'] = False
    p['selection_disclosure'] = 'This new algorithm explicitly selects between two candidates by the same fixed original objective on every frame; original cache reuse is recorded, not presented as a fresh independent full59 run.'
    p['no_manual_selection'] = True
    p['no_result_based_retry'] = True
    p['budget'] = dict(new_per_frame_fits=59, alternative_prior_sweeps=0, max_seconds=4500,
        expected_seconds=2300, automatic_followup=False)
    p['criteria'] = [
        'Anchor33 must reproduce the unchanged original fit; all59 frames preserve complete K, independent camera initialization, beta, observations and topology.',
        'Both candidate scores are recomputed through the original final objective at the saved parameters; every selected score must be <= the same-frame independent score.',
        'No frame is omitted, interpolated, replaced manually or retried. Preserve both candidate evidence and the exact selection decision.',
        'All six previously wrong left elbows18..23 must improve by at least50percent versus the original independent fit.',
        'For every frame, body/feet median original-observation residual must not worsen >1px and face median not >0.5px versus the independent fit.',
        'Actual all59 original-resolution body/face/hands/feet review and chronological SO3 inspection must show no new visible reversal/collapse.',
        'Lower fitting objective is not independent anatomical truth or clothing-surface acceptance. No automatic VHAP or Gaussian admission.',
        'Report incremental fitting time and the omitted independent-cache construction time separately; no video-to-export or15min success claim.']
    p['risks'] = ['A lower original objective can still describe a wrong arm or unobserved hand.',
        'Guarded initialization can still inherit an incorrect candidate with a lower objective.',
        'Cached independent fit cost is excluded from this diagnostic timing; fresh end-to-end runtime remains unqualified.']
    p['source_hashes'][str(Path(__file__).resolve())] = sha(__file__)
    p['source_hashes'][str(ORIGINAL_TEMPORAL_SOURCE)] = sha(ORIGINAL_TEMPORAL_SOURCE)
    p['input_hashes'][str(AUDIT)] = sha(AUDIT)
    p['input_hashes'][str(BODY/'temporal-root-sequence/review/closeout.json')] = sha(BODY/'temporal-root-sequence/review/closeout.json')
    for frame in p['temporal_roster']:
        for name in ('winner-camera-space.npz', 'official-result.pkl', 'result.json'):
            path = BASE/'frames'/frame/name
            p['input_hashes'][str(path)] = sha(path)
    save(PLAN, p)
    print('PREPARED_GUARDED_FULL59', p['anchor'], flush=True)


def worker():
    bind()
    lease.verify_handoff()
    p = temporal.verify_plan()
    original_source, original_fit = entry.adapted_entry_source, entry.fit_one
    expected_scores = {r['frame']:r['original_saved_objective'] for r in read(AUDIT)['rows'] if r['arm']=='baseline'}
    with np.load(BASE/'fit/fitted.npz', allow_pickle=False) as z:
        ids = z['frame_ids'].astype(str).tolist()
        targets = {frame:z['observed_native_keypoints'][i:i+1].copy() for i,frame in enumerate(ids)}

    def source(root):
        text, diff = original_source(root)
        marker = "    return {'winner': results[min_idx], 'orientation_count': len(results)}"
        capture = "    __import__('official_smplifyx_entry123')._guard_context = dict(loss=loss, camera=camera, gt_joints=gt_joints, joints_conf=joints_conf, joint_weights=joint_weights)\n"
        assert text.count(marker) == 1
        return text.replace(marker, capture+marker), diff+'\n# Read-only final objective context captured after original winner selection.\n'

    def fit(**kwargs):
        import torch
        frame = Path(kwargs['result_path']).parent.name
        folder = Path(kwargs['result_path']).parent
        assert np.array_equal(np.asarray(kwargs['keypoints']), targets[frame]), frame
        arrays, report = original_fit(**kwargs)
        context = entry._guard_context
        seed_arrays = {key:value.copy() for key,value in arrays.items()}
        with np.load(BASE/'frames'/frame/'winner-camera-space.npz', allow_pickle=False) as z:
            fallback = {key:z[key].copy() for key in arrays}
        for key in ('K_integer_centers', 'K_compute_float32', 'camera_translation', 'betas', 'faces'):
            assert np.array_equal(seed_arrays[key], fallback[key]), (frame, key)
        body, camera, vposer = kwargs['body_model'], context['camera'], kwargs['vposer2']
        scores = []
        with torch.no_grad():
            for candidate in (seed_arrays, fallback):
                body.reset_params(**{name:candidate[name] for name,_ in body.named_parameters()})
                camera.translation.copy_(torch.as_tensor(candidate['camera_translation'], device=kwargs['device']))
                latent = torch.as_tensor(candidate['pose_embedding'], device=kwargs['device'])
                output = body(body_pose=vposer.decode(latent)['pose_body'].reshape(1,63), return_verts=True, return_full_pose=True)
                vertices = output.vertices+camera.translation[:,None]
                projection = camera(output.joints)
                assert float((vertices-torch.as_tensor(candidate['vertices_camera'],device=kwargs['device'])).abs().max()) < 2e-6
                assert float((projection-torch.as_tensor(candidate['joints_projected_native'],device=kwargs['device'])).abs().max()) < .002
                value = context['loss'](output, camera=camera, gt_joints=context['gt_joints'], joints_conf=context['joints_conf'],
                    body_model_faces=body.faces_tensor.view(-1), joint_weights=context['joint_weights'],
                    pose_embedding=latent, use_vposer=True)
                scores.append(float(value))
            assert abs(scores[1]-expected_scores[frame]) < .01, (frame, scores[1], expected_scores[frame])
            chosen = 0 if scores[0] < scores[1] else 1
            selected = (seed_arrays, fallback)[chosen]
            body.reset_params(**{name:selected[name] for name,_ in body.named_parameters()})
            camera.translation.copy_(torch.as_tensor(selected['camera_translation'], device=kwargs['device']))
        temporal.sequence.save_npz(folder/'candidate-00-temporal.npz', **seed_arrays, frame_id=np.asarray(frame))
        save(folder/'candidate-00-report.json', report)
        parameters = Path(kwargs['result_path'])
        (folder/'candidate-00-official-result.pkl').write_bytes(parameters.read_bytes())
        if chosen == 1:
            parameters.write_bytes((BASE/'frames'/frame/'official-result.pkl').read_bytes())
        decision = dict(frame=frame, policy=p['selection_policy'], selected_index=chosen,
            original_saved_objectives=scores, seeded_upstream_reported_objective=report['winning_loss'],
            independent_cache_path=str(BASE/'frames'/frame/'winner-camera-space.npz'),
            independent_cache_sha256=sha(BASE/'frames'/frame/'winner-camera-space.npz'),
            independent_score_replay_agrees=True, manual_override=False, selected_fit_quality_accepted=False)
        save(folder/'objective-selection.json', decision)
        report.update(winning_loss=scores[chosen], winning_loss_origin='Recomputed original final objective at selected saved parameters',
            objective_selection=decision, independent_cache_used=True,
            upstream_solver_loss_optimizer_unchanged=True, local_input_output_selection_adaptation=True)
        entry._guard_context = None
        return selected, report

    entry.adapted_entry_source, entry.fit_one = source, fit
    try:
        temporal.worker()
        result = read(OUT/'result.json')
        result.update(selection_policy=p['selection_policy'], cache_policy=p['cache_policy'],
            independent_cache_used=True, fresh_video_to_model_timing=False, quality_accepted=False)
        save(OUT/'result.json', result)
    finally:
        entry.adapted_entry_source, entry.fit_one = original_source, original_fit


def run(pid):
    import fcntl
    bind()
    p = temporal.verify_plan()
    assert not OUT.exists()
    control = open('/tmp/shuolee-body-export123.lock','a+')
    fcntl.flock(control, fcntl.LOCK_EX|fcntl.LOCK_NB)
    command,state,identity = lease.process(pid)
    assert 'train_demo_body_upstream_offset123.py train' in command and state not in ('T','Z')
    receipt = dict(started=now(), supervisor_pid=os.getpid(), baseline_pid=pid, identity=identity, command=command, state='suspending')
    save(HANDOFF, receipt)
    try:
        os.kill(pid,signal.SIGSTOP)
        for _ in range(100):
            if lease.process(pid)[1]=='T': break
            time.sleep(.05)
        assert lease.process(pid)[1]=='T'
        env = dict(os.environ,SHUOLEE_CONTROL_BASELINE_PID=str(pid),SHUOLEE_CONTROL_BASELINE_IDENTITY=identity)
        with Path(__file__).with_suffix('.log').open('x',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'worker'],env=env,
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=p['runtime_budget_seconds'])
        receipt['control_finished']=True
    finally:
        try:
            if lease.process(pid)[2]==identity:
                os.kill(pid,signal.SIGCONT)
                receipt.update(state='resumed',baseline_state=lease.process(pid)[1])
            else: receipt['state']='baseline_identity_changed_not_signaled'
        except (FileNotFoundError,ProcessLookupError): receipt['state']='baseline_exited_before_resume'
        receipt['finished']=now()
        save(HANDOFF,receipt)
        print('GUARDED_ROOT_BASELINE',receipt['state'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','run','worker'])
    parser.add_argument('--pid',type=int)
    args=parser.parse_args()
    if args.mode=='prepare': prepare()
    elif args.mode=='run': run(args.pid)
    else: worker()
