"""Bounded original SMPLify-X two-orientation diagnostic; no new solver."""
import argparse
import difflib
import inspect
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import official_smplifyx_entry123 as entry
import run_official_sequence123 as sequence
from run_official_sequence123_topology import cache_static_topology

REC = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
DEMO = REC / 'demo-vhap-initialization'
BODY = DEMO / 'fullbody-smplifyx'
BASE = BODY / 'full-sequence'
OUT = BODY / 'orientation-control'
PLAN = BODY / 'orientation-control-plan.json'
AUDIT = BODY / 'orientation-selection-audit.json'
HANDOFF = BODY / 'orientation-control-gpu-handoff.json'
FRAMES = ['frame_000033', 'frame_000016', 'frame_000018']
read, save, sha, now = sequence.read, sequence.save, sequence.sha, sequence.now


def arguments(inspect_only=False):
    project = '/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118'
    return SimpleNamespace(official_root=str(REC/'smplifyx-reference-control/official'),
        data_root='/opt/gs-slam-lab/shuolee-local-surface-123/demo-vhap-fps3',
        runtime_root=project, assets_root='/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets',
        project_root=project, length_contract_json=str(DEMO/'preprocess-fps3/full-sequence-inputs.json'),
        reference_frame_id='frame_000033', shared_betas_npz=str(BODY/'frame_000033/winner-camera-space.npz'),
        shared_betas_key='betas', output_dir=str(OUT), frame_ids=FRAMES, device='cuda',
        entry_sha256=sequence.ENTRY_SHA256, inspect_only=inspect_only)


def process(pid):
    root = Path('/proc')/str(pid)
    fields = (root/'stat').read_text().split(') ', 1)[1].split()
    return (root/'cmdline').read_bytes().replace(b'\0', b' ').decode(), fields[0], fields[19]


def verify_handoff():
    cmd, state, identity = process(int(os.environ['SHUOLEE_CONTROL_BASELINE_PID']))
    assert 'train_demo_body_upstream_offset123.py train' in cmd and state == 'T'
    assert identity == os.environ['SHUOLEE_CONTROL_BASELINE_IDENTITY']


def prepare():
    assert not PLAN.exists() and not OUT.exists()
    args = arguments(True)
    _, rows, _, _, _, bound, provenance = sequence.load_inputs(args)
    for row in rows:
        sequence.checked_image(row)
    with np.load(BASE/'fit/fitted.npz', allow_pickle=False) as fit:
        ids = fit['frame_ids'].astype(str)
        obs, pred = fit['observed_native_keypoints'], fit['joints_projected_native']
        audit_rows = []
        for i, frame in enumerate(ids):
            receipt = read(BASE/'frames'/frame/'result.json')
            distance = float(np.linalg.norm(obs[i, 5, :2]-obs[i, 2, :2]))
            count = receipt['orientation_count']
            assert count == (2 if distance < 25 else 1)
            errors = np.linalg.norm(pred[i]-obs[i, :, :2], axis=-1)
            audit_rows.append(dict(frame=frame, shoulder_distance_px=distance, orientation_count=count,
                left_elbow_px=float(errors[6]), right_elbow_px=float(errors[3])))
    audit = dict(updated=now(), state='all59_original_branch_verified', rows=audit_rows,
        frames=len(audit_rows), one_orientation=sum(r['orientation_count']==1 for r in audit_rows),
        two_orientations=sum(r['orientation_count']==2 for r in audit_rows), threshold_px=25,
        baseline_fit_sha256=sha(BASE/'fit/fitted.npz'), no_optimization=True,
        interpretation='Association only: omitted alternate orientation is not yet proven to cause wrong arms.')
    save(AUDIT, audit)
    source_files = [Path(__file__), Path(sequence.__file__), Path(entry.__file__),
        Path(__file__).with_name('run_official_sequence123_topology.py')]
    source_files += [Path(args.official_root)/p for p in entry.PINNED]
    bound[args.shared_betas_npz] = sha(args.shared_betas_npz)
    for frame in FRAMES:
        path = BASE/'frames'/frame/'winner-camera-space.npz'
        bound[str(path)] = sha(path)
    threshold = max(float(np.hypot(*r['wh'])) for r in rows) + 1
    plan = dict(updated=now(), state='prepared', frames=FRAMES, output=str(OUT),
        hypothesis='The original 25px shoulder gate may omit an alternate optimization basin. Try both original orientations uniformly on the fixed diagnostic roster.',
        single_change=dict(parameter='side_view_thsh', baseline=25, candidate=threshold,
            meaning='Exceeds each image diagonal; both original orientations are attempted for every diagnostic frame.'),
        unchanged=['native observations and confidence', 'complete K', 'camera initialization',
            'shared beta', 'original five stages', 'loss weights', 'LBFGS', 'maxiters30'],
        criteria=['Candidate0 must replay the baseline within 1e-3 native pixel and 1e-6 model meter.',
            'Original selected candidate must reduce both frame16 and18 left-elbow errors by at least50percent.',
            'On all3 frames, body and feet median regression <=1px; face median regression <=0.5px.',
            'Actual native-pixel body/face/feet review must show no new reversal or collapse.',
            'Compare original returned-loss ranking with the recomputed saved-state objective; do not choose candidates manually.'],
        runtime_budget_seconds=600, no_automatic_full59=True, no_Gaussian_training=True,
        no_frame_filtering=True, no_threshold_sweep=True, fresh_optimizer=True,
        risks=['Approximately two body solves per frame', '180degree alternative may not help elbow ambiguity'],
        source_hashes={str(p.resolve()):sha(p) for p in source_files}, input_hashes=bound,
        input_check=provenance, audit_sha256=sha(AUDIT), quality_accepted=False)
    save(PLAN, plan)
    print('PREPARED', len(rows), 'frames; one/two original orientations', audit['one_orientation'], audit['two_orientations'], flush=True)


def verify_plan():
    plan = read(PLAN)
    assert plan['frames'] == FRAMES
    for path, digest in {**plan['source_hashes'], **plan['input_hashes']}.items():
        assert sha(path) == digest, path
    return plan


def worker():
    verify_handoff()
    plan = verify_plan()
    assert not OUT.exists()
    old_config, old_source, old_fit = entry.load_official_config, entry.adapted_entry_source, entry.fit_one

    def config(root):
        value = old_config(root)
        assert value['maxiters'] == 30
        value['side_view_thsh'] = plan['single_change']['candidate']
        return value

    def source(root):
        before, original_diff = old_source(root)
        marker = "    return {'winner': results[min_idx], 'orientation_count': len(results)}"
        capture = "    __import__('official_smplifyx_entry123')._orientation_capture = dict(results=results, selected=min_idx, loss=loss, camera=camera, vposer=vposer, gt_joints=gt_joints, joints_conf=joints_conf, joint_weights=joint_weights)\n"
        assert before.count(marker) == 1
        after = before.replace(marker, capture+marker)
        diff = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
            fromfile='existing_output_adapter', tofile='candidate_observation_only'))
        return after, original_diff+'\n'+diff

    completed = []
    def fit(**kwargs):
        import torch
        verify_handoff()
        verify_plan()
        topology = cache_static_topology(kwargs['body_model'])
        arrays, report = old_fit(**kwargs)
        context = entry._orientation_capture
        assert len(context['results']) == 2 and report['orientation_count'] == 2
        body, camera, vposer = kwargs['body_model'], context['camera'], context['vposer']
        folder = Path(kwargs['result_path']).parent
        frame = folder.name
        summaries = []
        with torch.no_grad():
            for index, candidate in enumerate(context['results']):
                result = candidate['result']
                body.reset_params(**{name:result[name] for name, _ in body.named_parameters()})
                camera.translation.copy_(torch.as_tensor(result['camera_translation'], device=kwargs['device']))
                latent = torch.as_tensor(result['body_pose'], device=kwargs['device'])
                axis_angle = vposer.decode(latent, output_type='aa').reshape(1,63)
                output = body(body_pose=axis_angle, return_verts=True, return_full_pose=True)
                total = context['loss'](output, camera=camera, gt_joints=context['gt_joints'],
                    body_model_faces=body.faces_tensor.view(-1), joints_conf=context['joints_conf'],
                    joint_weights=context['joint_weights'], pose_embedding=latent, use_vposer=True)
                candidate_arrays = {name:np.asarray(result[name]).copy() for name, _ in body.named_parameters()}
                candidate_arrays.update(body_pose_axis_angle=axis_angle.cpu().numpy(), pose_embedding=latent.cpu().numpy(),
                    camera_translation=camera.translation.detach().cpu().numpy().copy(),
                    vertices_camera=(output.vertices+camera.translation[:,None]).cpu().numpy(),
                    joints_camera=(output.joints+camera.translation[:,None]).cpu().numpy(),
                    joints_projected_native=camera(output.joints).cpu().numpy(), faces=np.asarray(body.faces),
                    K_integer_centers=np.asarray(kwargs['K_integer_centers']).copy(),
                    K_compute_float32=camera.K.cpu().numpy(), frame_id=np.asarray(frame))
                path = folder/f'candidate-{index:02d}.npz'
                sequence.save_npz(path, **candidate_arrays)
                detail = dict(index=index, selected=index==context['selected'],
                    returned_loss=float(candidate['loss']), saved_state_loss=float(total), path=str(path), sha256=sha(path))
                if index == 0:
                    with np.load(BASE/'frames'/frame/'winner-camera-space.npz', allow_pickle=False) as original:
                        keys = ('vertices_camera','joints_projected_native','global_orient','camera_translation',
                            'body_pose_axis_angle','pose_embedding','betas','expression','jaw_pose','left_hand_pose','right_hand_pose')
                        parity = {key:dict(exact=bool(np.array_equal(original[key], candidate_arrays[key])),
                            max_abs=float(np.max(np.abs(original[key]-candidate_arrays[key])))) for key in keys}
                    detail['baseline_parity'] = parity
                    detail['parity_passed'] = (parity['joints_projected_native']['max_abs'] <= 1e-3 and
                        parity['vertices_camera']['max_abs'] <= 1e-6)
                summaries.append(detail)
            selected = context['results'][context['selected']]['result']
            body.reset_params(**{name:selected[name] for name, _ in body.named_parameters()})
            camera.translation.copy_(torch.as_tensor(selected['camera_translation'], device=kwargs['device']))
        selected_saved = 0 if summaries[0]['saved_state_loss'] < summaries[1]['saved_state_loss'] else 1
        trace = dict(frame=frame, selected_original=context['selected'], selected_saved_objective=selected_saved,
            objective_selection_agrees=context['selected']==selected_saved, candidates=summaries,
            baseline_candidate_parity_passed=summaries[0]['parity_passed'], no_selection_override=True)
        save(folder/'candidate-selection.json', trace)
        report.update(runtime_optimization=topology, orientation_control=trace,
            single_changed_parameter=plan['single_change'])
        completed.append(trace)
        entry._orientation_capture = None
        return arrays, report

    # The baseline owns the native GPU lock while stopped. The supervisor owns
    # the existing export/control lease and verifies its exact process identity.
    original = inspect.getsource(sequence.run)
    lock_lines = "    lock = open('/tmp/shuolee-native122-gpu.lock', 'a+')\n    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)"
    assert original.count(lock_lines) == 1
    runner = original.replace(lock_lines, '    verify_handoff()')
    namespace = dict(sequence.run.__globals__, verify_handoff=verify_handoff)
    exec(compile(runner, '<existing-official-sequence-serialized-gpu>', 'exec'), namespace)
    entry.load_official_config, entry.adapted_entry_source, entry.fit_one = config, source, fit
    started = time.monotonic()
    try:
        namespace['run'](arguments())
        verify_plan()
        assert len(completed) == 3
        save(OUT/'result.json', dict(state='complete_pending_review', updated=now(), frames=completed,
            elapsed_seconds=time.monotonic()-started, plan_sha256=sha(PLAN),
            baseline_parity_passed=all(r['baseline_candidate_parity_passed'] for r in completed),
            quality_accepted=False, full59_admitted=False, gaussian_training_started=False))
        (OUT/'runner-source.py').write_text(runner, encoding='utf-8')
    finally:
        entry.load_official_config, entry.adapted_entry_source, entry.fit_one = old_config, old_source, old_fit


def run(pid):
    import fcntl
    plan = verify_plan()
    assert not OUT.exists()
    lease = open('/tmp/shuolee-body-export123.lock', 'a+')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cmd, state, identity = process(pid)
    assert 'train_demo_body_upstream_offset123.py train' in cmd and state not in ('T','Z')
    receipt = dict(started=now(), supervisor_pid=os.getpid(), baseline_pid=pid, identity=identity, command=cmd, state='suspending')
    save(HANDOFF, receipt)
    try:
        os.kill(pid, signal.SIGSTOP)
        for _ in range(100):
            if process(pid)[1] == 'T': break
            time.sleep(.05)
        assert process(pid)[1] == 'T'
        env = dict(os.environ, SHUOLEE_CONTROL_BASELINE_PID=str(pid), SHUOLEE_CONTROL_BASELINE_IDENTITY=identity)
        with Path(__file__).with_name('smplifyx-orientation-control.log').open('x', encoding='utf-8') as log:
            subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'worker'], env=env,
                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=plan['runtime_budget_seconds'])
        receipt['control_finished'] = True
    finally:
        try:
            if process(pid)[2] == identity:
                os.kill(pid, signal.SIGCONT)
                receipt.update(state='resumed', baseline_state=process(pid)[1])
            else:
                receipt['state'] = 'baseline_identity_changed_not_signaled'
        except (FileNotFoundError, ProcessLookupError):
            receipt['state'] = 'baseline_exited_before_resume'
        receipt['finished'] = now()
        save(HANDOFF, receipt)
        print('ORIENTATION_CONTROL_BASELINE', receipt['state'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare','run','worker'])
    parser.add_argument('--pid', type=int)
    args = parser.parse_args()
    if args.mode == 'prepare': prepare()
    elif args.mode == 'run': run(args.pid)
    else: worker()
