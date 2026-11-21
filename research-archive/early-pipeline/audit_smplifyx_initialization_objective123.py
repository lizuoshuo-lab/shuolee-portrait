"""Replay the original objective at all saved initialization-control outputs.

Read-only: no fitting calls, optimizer, winner export, frame selection or splicing.
"""
import argparse
from pathlib import Path
import os
import signal
import subprocess
import sys
import time
import traceback
import numpy as np
import replay_demo_limb_losses123 as base
import control_smplifyx_orientations123 as lease
import official_smplifyx_entry123 as entry

BODY = base.ROOT.parent
PLAN = BODY/'initialization-objective-plan.json'
OUT = BODY/'initialization-objective-audit'
HANDOFF = BODY/'initialization-objective-gpu-handoff.json'
read, save, sha = base.read, base.save, base.sha
now = lease.now


def prepare():
    assert not PLAN.exists() and not OUT.exists()
    frames = [f'frame_{i:06d}' for i in range(1, 60)]
    roots = [('baseline', base.ROOT, frames),
             ('temporal', BODY/'temporal-root-sequence', frames)]
    roots += [(arm, BODY/'root-body-seed-control'/arm,
               ['frame_000033', 'frame_000016', 'frame_000018'])
              for arm in ('body_only', 'body_and_root')]
    samples = []
    bound = {}
    for arm, root, roster in roots:
        for frame in roster:
            path, receipt = root/'frames'/frame/'winner-camera-space.npz', root/'frames'/frame/'result.json'
            samples.append(dict(arm=arm, frame=frame, path=str(path), receipt=str(receipt)))
            bound[str(path)], bound[str(receipt)] = sha(path), sha(receipt)
            with np.load(path, allow_pickle=False) as values, np.load(base.ROOT/'frames'/frame/'winner-camera-space.npz', allow_pickle=False) as original:
                assert str(values['frame_id'].item()) == frame
                for key in ('K_integer_centers', 'betas', 'camera_translation', 'faces'):
                    assert np.array_equal(values[key], original[key]), (arm, frame, key)
    for path in (base.ROOT/'fit/fitted.npz', base.ROOT/'data/training-inputs.json',
                 BODY/'temporal-root-sequence/review/closeout.json'):
        bound[str(path)] = sha(path)
    rows = read(base.ROOT/'data/training-inputs.json')['records']
    assert len(rows) == 59 and [row['frame'] for row in sorted(rows, key=lambda row:int(row['source_frame_index']))] == frames
    source = [Path(__file__).resolve(), Path(base.__file__), Path(lease.__file__), Path(entry.__file__)]
    source += [base.OFFICIAL/p for p in entry.PINNED]
    save(PLAN, dict(state='prepared', updated=now(), frames=frames, samples=samples,
        hypothesis='Before considering a uniform original-objective selection guard, verify actual saved-state losses for all59 baseline/temporal pairs and the six already completed matched probes. Does lower loss reject new visible failures, and do wrong elbows remain?',
        original_solver_unchanged=True, new_fit_runs=0, optimizer_updates=0,
        body_forwards_max=124, wall_seconds_max=120, no_output_fit=True,
        criteria=['All124 forwards must preserve saved vertices within2e-6m and native projected joints within0.002px.',
                  'Recompute the unchanged final original objective and compare its ordering to reported last-step losses.',
                  'Report all59 pairs including original-observation elbow/face/feet errors; no threshold scan or dropped observations.',
                  'Relate the fixed objective rule to the already completed all59 visual review; no new candidate is accepted or exported.',
                  'Preserve every source/input hash and resume the exact original Gaussian process.'],
        limits=['A lower fitting objective is not independent anatomical truth.',
                'A post-hoc comparison of saved outputs does not validate a fresh multi-start pipeline.',
                'The matched probes use frozen original neighbors and are not the full temporal sequence.'],
        source_hashes={str(p.resolve()):sha(p) for p in source}, input_hashes=bound,
        quality_accepted=False, goal_complete=False))
    print('PREPARED_ORIGINAL_OBJECTIVE', len(samples), flush=True)


def verify():
    p = read(PLAN)
    assert len(p['samples']) == p['body_forwards_max'] == 124
    for path, digest in {**p['source_hashes'], **p['input_hashes']}.items():
        assert sha(path) == digest, path
    return p


def worker():
    lease.verify_handoff()
    p = verify()
    assert not OUT.exists()
    OUT.mkdir()
    save(OUT/'progress.json', dict(state='readonly_original_objective', pid=os.getpid(), completed=0, total=124))
    start = time.monotonic()
    try:
        import torch
        import smplx
        entry.adapted_entry_source(base.OFFICIAL)
        config = entry.load_official_config(base.OFFICIAL)
        raw = (base.OFFICIAL/'smplifyx/fit_single_frame.py').read_text()
        assert 'data_weight = 1000 / H' in raw
        utils, prior, fitting, _ = entry.load_official(base.OFFICIAL)
        mapping = utils.smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True,
            use_face_contour=False, openpose_format='coco25')
        body = smplx.SMPLX(str(base.ASSETS/'smplx/SMPLX_NEUTRAL.npz'), batch_size=1,
            num_betas=10, num_expression_coeffs=10, use_pca=True, num_pca_comps=12,
            flat_hand_mean=False, use_face_contour=False, create_body_pose=False, create_transl=False,
            joint_mapper=utils.JointMapper(mapping), dtype=torch.float32).cuda()
        body.parents = body.parents.cpu()
        body.eval()
        for parameter in body.parameters():
            parameter.requires_grad_(False)
        loader = entry.module(base.PROJECT/'src/avatar_core/shuolee_vposer.py', '_objective123_vposer')
        loader.MODEL, loader.CODE = base.ASSETS/'vposer/V02_05', base.PROJECT/'third_party/human_body_prior'
        vposer, _, vposer_meta = loader.load_vposer('cuda')
        vposer.eval()
        for parameter in vposer.parameters():
            parameter.requires_grad_(False)
        l2 = lambda: prior.L2Prior().cuda()
        loss = fitting.create_loss(loss_type='smplify', rho=config['rho'], body_pose_prior=l2(),
            shape_prior=l2(), expr_prior=l2(), angle_prior=prior.SMPLifyAnglePrior().cuda(), jaw_prior=l2(),
            left_hand_prior=l2(), right_hand_prior=l2(), use_joints_conf=True, use_face=True,
            use_hands=True, interpenetration=False, dtype=torch.float32).cuda()
        with np.load(base.ROOT/'fit/fitted.npz', allow_pickle=False) as z:
            ids, targets = z['frame_ids'].astype(str).tolist(), z['observed_native_keypoints'].copy()
        records = {row['frame']:row for row in read(base.ROOT/'data/training-inputs.json')['records']}
        values = []
        with torch.no_grad():
            for index, item in enumerate(p['samples']):
                lease.verify_handoff()
                assert time.monotonic()-start < p['wall_seconds_max']
                frame, arm = item['frame'], item['arm']
                with np.load(item['path'], allow_pickle=False) as z:
                    saved = dict(z)
                body.reset_params(**{name:saved[name] for name, _ in body.named_parameters()})
                latent = torch.tensor(saved['pose_embedding'], device='cuda')
                result = body(body_pose=vposer.decode(latent)['pose_body'].reshape(1,63),
                              return_verts=True, return_full_pose=True)
                K = torch.tensor(saved['K_integer_centers'], device='cuda', dtype=torch.float32)
                translation = torch.tensor(saved['camera_translation'], device='cuda')
                def camera(points):
                    q = (points+translation[:,None])@K.T
                    return q[...,:2]/q[...,2:]
                predicted = camera(result.joints)
                vertex_delta = float((result.vertices+translation[:,None]-torch.tensor(saved['vertices_camera'],device='cuda')).abs().max())
                projection_delta = float((predicted-torch.tensor(saved['joints_projected_native'],device='cuda')).abs().max())
                assert vertex_delta < 2e-6 and projection_delta < .002, (frame, arm, vertex_delta, projection_delta)
                target = torch.tensor(targets[ids.index(frame):ids.index(frame)+1], device='cuda')
                weights = torch.ones((1,118), device='cuda')
                weights[:,[1,9,12]] = 0
                weights[:,25:67] = config['hand_joints_weights'][-1]
                weights[:,67:] = config['face_joints_weights'][-1]
                final = dict(data_weight=1000/records[frame]['wh'][1], body_pose_weight=config['body_pose_prior_weights'][-1],
                    shape_weight=config['shape_weights'][-1], expr_prior_weight=config['expr_weights'][-1],
                    bending_prior_weight=3.17*config['body_pose_prior_weights'][-1],
                    hand_prior_weight=config['hand_pose_prior_weights'][-1],
                    jaw_prior_weight=[float(x) for x in config['jaw_pose_prior_weights'][-1].split(',')])
                loss.reset_loss_weights(final)
                total = loss(result, camera=camera, gt_joints=target[...,:2], joints_conf=target[...,2],
                    body_model_faces=body.faces_tensor.view(-1), joint_weights=weights,
                    pose_embedding=latent, use_vposer=True)
                residual = predicted-target[...,:2]
                terms = ((weights*target[...,2]).square()[...,None]*loss.data_weight.square()*loss.robustifier(residual)).sum(-1)[0]
                groups = dict(body=sum(terms[:19]), feet=sum(terms[19:25]), hands=sum(terms[25:67]), face=sum(terms[67:118]))
                distance = torch.linalg.vector_norm(residual, dim=-1)[0]
                receipt = read(item['receipt'])
                values.append(dict(frame=frame, arm=arm, path=item['path'], fit_sha256=sha(item['path']),
                    original_saved_objective=float(total), reported_objective=receipt['winning_loss'],
                    reported_minus_saved=receipt['winning_loss']-float(total),
                    group_data_terms={name:float(v) for name,v in groups.items()},
                    prior_terms_total=float(total-terms.sum()), left_elbow_px=float(distance[6]), right_elbow_px=float(distance[3]),
                    vertex_parity_max_m=vertex_delta, projection_parity_max_px=projection_delta))
                if (index+1)%10 == 0 or index == len(p['samples'])-1:
                    save(OUT/'progress.json', dict(state='readonly_original_objective', pid=os.getpid(), completed=index+1, total=124))
        by_key = {(r['arm'], r['frame']):r for r in values}
        pairs = []
        for frame in p['frames']:
            original, temporal = by_key['baseline',frame], by_key['temporal',frame]
            preferred = 'temporal' if temporal['original_saved_objective'] < original['original_saved_objective'] else 'baseline'
            reported = 'temporal' if temporal['reported_objective'] < original['reported_objective'] else 'baseline'
            pairs.append(dict(frame=frame, baseline=original['original_saved_objective'], temporal=temporal['original_saved_objective'],
                diagnostic_lower_objective=preferred, reported_order_agrees=preferred==reported,
                baseline_left_elbow=original['left_elbow_px'], temporal_left_elbow=temporal['left_elbow_px'],
                no_result_selected_or_exported=True))
        verify()
        report = dict(state='complete_readonly_original_objective', updated=now(), elapsed_seconds=time.monotonic()-start,
            body_forwards=len(values), optimizer_updates=0, rows=values, all59_pairs=pairs, vposer=vposer_meta,
            changed_order_frames=[r['frame'] for r in pairs if not r['reported_order_agrees']],
            max_reported_saved_difference=max(abs(r['reported_minus_saved']) for r in values),
            no_fit_modified=True, no_selection_export=True, plan_sha256=sha(PLAN),
            quality_accepted=False, goal_complete=False, Gaussian_training_started=False)
        save(OUT/'result.json', report)
        save(OUT/'progress.json', dict(state='complete', pid=None, completed=124, result_sha256=sha(OUT/'result.json')))
        print('OBJECTIVE_AUDIT', __import__('json').dumps({k:report[k] for k in ('elapsed_seconds','body_forwards','changed_order_frames','max_reported_saved_difference')}), flush=True)
    except BaseException as error:
        save(OUT/'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), fit_files_modified=False))
        save(OUT/'progress.json', dict(state='failed_preserved', pid=None))
        raise


def run(pid):
    import fcntl
    p = verify()
    assert not OUT.exists()
    control = open('/tmp/shuolee-body-export123.lock', 'a+')
    fcntl.flock(control, fcntl.LOCK_EX|fcntl.LOCK_NB)
    command, state, identity = lease.process(pid)
    assert 'train_demo_body_upstream_offset123.py train' in command and state not in ('T','Z')
    receipt = dict(started=now(), supervisor_pid=os.getpid(), baseline_pid=pid, baseline_identity=identity,
                   command=command, state='suspending')
    save(HANDOFF, receipt)
    try:
        os.kill(pid, signal.SIGSTOP)
        for _ in range(100):
            if lease.process(pid)[1] == 'T': break
            time.sleep(.05)
        assert lease.process(pid)[1] == 'T'
        env = dict(os.environ, SHUOLEE_CONTROL_BASELINE_PID=str(pid), SHUOLEE_CONTROL_BASELINE_IDENTITY=identity)
        with Path(__file__).with_suffix('.log').open('x', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), 'worker'], env=env,
                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=p['wall_seconds_max'])
        receipt['audit_completed'] = True
    finally:
        try:
            if lease.process(pid)[2] == identity:
                os.kill(pid, signal.SIGCONT)
                receipt.update(state='resumed', baseline_state=lease.process(pid)[1])
            else: receipt['state'] = 'baseline_identity_changed_not_signaled'
        except (FileNotFoundError, ProcessLookupError):
            receipt['state'] = 'baseline_exited_before_resume'
        receipt['finished'] = now()
        save(HANDOFF, receipt)
        print('OBJECTIVE_AUDIT_BASELINE', receipt['state'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare','run','worker'])
    parser.add_argument('--pid', type=int)
    args = parser.parse_args()
    if args.mode == 'prepare': prepare()
    elif args.mode == 'run': run(args.pid)
    else: worker()
