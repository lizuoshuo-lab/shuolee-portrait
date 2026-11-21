"""Bounded matched initialization control around the pinned original solver."""
import argparse
import difflib
import inspect
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import numpy as np
import control_smplifyx_orientations123 as common
import official_smplifyx_entry123 as entry
import run_official_sequence123 as sequence
from run_official_sequence123_topology import cache_static_topology

BODY, BASE = common.BODY, common.BASE
OUT = BODY/'root-body-seed-control'
PLAN = BODY/'root-body-seed-plan.json'
HANDOFF = BODY/'root-body-seed-gpu-handoff.json'
FRAMES = ['frame_000033', 'frame_000016', 'frame_000018']
ARMS = ['body_only', 'body_and_root']
read, save, sha, now = common.read, common.save, common.sha, common.now


def arguments(arm, inspect_only=False):
    args = common.arguments(inspect_only)
    args.output_dir = str(OUT/arm)
    args.frame_ids = FRAMES
    return args


def adapted_source(root):
    source, diff = entry._root_seed_original_source(root)
    body_hook = 'pose_embedding.fill_(0)'
    root_hook = '        if try_both_orient:\n'
    assert source.count(body_hook) == source.count(root_hook) == 1
    body_start = "pose_embedding.copy_(torch.as_tensor(kwargs['_seed_pose_embedding'], dtype=dtype, device=device))"
    root_start = """        _seed_trace = dict(camera_translation=camera.translation.detach().cpu().numpy().copy(),
                           camera_initial_root=body_model.global_orient.detach().cpu().numpy().copy())
        if kwargs['_seed_root_enabled']:
            with torch.no_grad():
                body_model.global_orient.copy_(torch.as_tensor(kwargs['_seed_global_orient'], dtype=dtype, device=device))
        _seed_trace['body_fit_initial_root'] = body_model.global_orient.detach().cpu().numpy().copy()
        __import__('official_smplifyx_entry123')._root_seed_trace = _seed_trace
"""
    changed = source.replace(body_hook, body_start).replace(root_hook, root_start+root_hook)
    compile(changed, '<original-smplifyx-explicit-body-root-start>', 'exec')
    delta = ''.join(difflib.unified_diff(source.splitlines(True), changed.splitlines(True),
        fromfile='existing-official-input-adapter', tofile='explicit-neighbor-body-root-start'))
    return changed, diff+'\n'+delta


def prepare():
    assert not OUT.exists() and not PLAN.exists()
    args = arguments('body_only', True)
    _, rows, _, _, _, bound, provenance = sequence.load_inputs(args)
    for row in rows:
        sequence.checked_image(row)
    roster = sorted(read(Path(args.data_root)/'data/inputs.json')['records'],
        key=lambda r: (int(r['source_frame_index']), r['frame']))
    ids = [r['frame'] for r in roster]
    assert len(ids) == len(set(ids)) == 59
    assert len({int(r['source_frame_index']) for r in roster}) == 59
    schedule = [{'target': frame, 'seed': ids[ids.index(frame)-1]} for frame in FRAMES]
    audit = []
    for item in schedule:
        assert ids.index(item['target']) > 0
        paths = {role: BASE/'frames'/item[role]/'winner-camera-space.npz' for role in ['seed','target']}
        with np.load(paths['seed'], allow_pickle=False) as s, np.load(paths['target'], allow_pickle=False) as t:
            assert str(s['frame_id'].item()) == item['seed'] and str(t['frame_id'].item()) == item['target']
            assert s['pose_embedding'].shape == (1,32) and s['global_orient'].shape == (1,3)
            for key in ['K_integer_centers','w2c','source_w2c','betas','faces']:
                assert np.array_equal(s[key], t[key]), (item,key)
            assert np.allclose(t['w2c'], np.eye(4), atol=0, rtol=0)
            assert np.isfinite(s['pose_embedding']).all() and np.isfinite(s['global_orient']).all()
            audit.append(dict(**item, matching_K_gauge_shape_topology=True,
                camera_relative_gauge_not_recovered_world=True,
                seed_body_latent=s['pose_embedding'].tolist(), seed_root=s['global_orient'].tolist()))
        for path in paths.values():
            bound[str(path)] = sha(path)
    old = BODY/'neighbor-seed18/frames/frame_000018/winner-camera-space.npz'
    assert old.exists()
    bound[str(old)] = sha(old)
    bound[args.shared_betas_npz] = sha(args.shared_betas_npz)
    source_paths = [Path(__file__), Path(common.__file__), Path(sequence.__file__), Path(entry.__file__),
        Path(__file__).with_name('run_official_sequence123_topology.py')]
    source_paths += [Path(args.official_root)/p for p in entry.PINNED]
    entry._root_seed_original_source = entry.adapted_entry_source
    adapted_source(args.official_root)
    plan = dict(updated=now(), state='prepared', frames=FRAMES, arms=ARMS, temporal_schedule=schedule,
        hypothesis='The body-only predecessor initialization can retain an inconsistent whole-body root. Test adding that same predecessor root after the unchanged original camera initialization.',
        single_difference_between_arms='Initial global_orient for the five original body-fit stages only',
        unchanged=['deterministic immediate-predecessor body latent', 'native observations/confidence',
            'complete K', 'original camera initialization and resulting camera translation', 'shared frozen beta',
            'original shoulder-gated orientation branch', 'five stages/losses/weights', 'LBFGS/maxiters30',
            'zero hand/expression start', 'fresh optimizers'],
        no_manual_seed_selection=True, no_source_code_edits=True,
        random_seed=read(BODY/'neighbor-seed-plan.json')['random_seed'],
        full_roster_count=59, input_check=provenance, seed_audit=audit,
        criteria=['Body-only frame18 must replay the historical predecessor-latent control within 1e-3 pixel and 1e-6 meter.',
            'Camera initialization root/translation must be identical between both arms on all3 frames.',
            'Adding root must reduce left-elbow error on both16/18 by at least50percent versus body-only.',
            'Each frame body/feet median must not regress >1px, face median not >0.5px versus both body-only and independent baseline.',
            'Review native body/face/hands/feet for all3 frames; no new reversal/collapse.',
            'These3 frames are a causal probe, not full59 quality or end-to-end acceptance.'],
        risks=['A bad predecessor root can propagate an error.', 'The unchanged camera-init translation can remain inconsistent.',
            'Camera-relative temporal initialization is approximate under moving camera, not a physical world constraint.'],
        runtime_budget_seconds=600, no_automatic_full59=True, no_gaussian_training=True,
        quality_accepted=False, goal_complete=False,
        source_hashes={str(p.resolve()):sha(p) for p in source_paths}, input_hashes=bound)
    save(PLAN, plan)
    print('PREPARED',schedule,flush=True)


def verify_plan():
    p = read(PLAN)
    assert p['frames'] == FRAMES and p['arms'] == ARMS
    for path,digest in {**p['source_hashes'],**p['input_hashes']}.items():
        assert sha(path) == digest, path
    return p


def worker():
    common.verify_handoff()
    p = verify_plan()
    assert not OUT.exists()
    OUT.mkdir()
    old_source, old_config, old_fit = entry.adapted_entry_source, entry.load_official_config, entry.fit_one
    entry._root_seed_original_source = old_source
    active = {}
    def config(root):
        c = old_config(root)
        assert c['maxiters'] == 30 and c.get('side_view_thsh',25) == 25
        c.update(_seed_pose_embedding=active['latent'], _seed_global_orient=active['root'],
            _seed_root_enabled=active['arm']=='body_and_root')
        return c
    traces = {}
    def fit(**kwargs):
        common.verify_handoff()
        verify_plan()
        frame = Path(kwargs['result_path']).parent.name
        item = next(r for r in p['temporal_schedule'] if r['target']==frame)
        seed_path = BASE/'frames'/item['seed']/'winner-camera-space.npz'
        with np.load(seed_path,allow_pickle=False) as s:
            active.update(latent=s['pose_embedding'].copy(),root=s['global_orient'].copy())
        topology = cache_static_topology(kwargs['body_model'])
        arrays,report = old_fit(**kwargs)
        trace = {k:v.tolist() for k,v in entry._root_seed_trace.items()}
        expected = active['root'] if active['arm']=='body_and_root' else np.asarray(trace['camera_initial_root'])
        assert np.array_equal(np.asarray(trace['body_fit_initial_root']),expected)
        assert np.array_equal(arrays['camera_translation'],np.asarray(trace['camera_translation']))
        if active['arm']=='body_and_root':
            other = traces['body_only'][frame]
            for key in ['camera_translation','camera_initial_root']:
                assert np.array_equal(trace[key],other[key]), (frame,key)
        traces.setdefault(active['arm'],{})[frame] = trace
        report.update(initialization='Explicit immediate predecessor body latent, with optional root; diagnostic only',
            seed_frame=item['seed'],seed_sha256=sha(seed_path), arm=active['arm'], initial_state=trace,
            runtime_optimization=topology, temporal_loss_added=False, original_loss_optimizer_unchanged=True)
        return arrays,report
    runner = inspect.getsource(sequence.run)
    lock_lines = "    lock = open('/tmp/shuolee-native122-gpu.lock', 'a+')\n    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)"
    assert runner.count(lock_lines)==1
    runner=runner.replace(lock_lines,'    verify_handoff()')
    namespace=dict(sequence.run.__globals__,verify_handoff=common.verify_handoff)
    exec(compile(runner,'<original-sequence-serialized-root-seed-control>','exec'),namespace)
    entry.adapted_entry_source,entry.load_official_config,entry.fit_one=adapted_source,config,fit
    start=time.monotonic()
    try:
        import torch
        for arm in ARMS:
            random.seed(p['random_seed']);np.random.seed(p['random_seed']);torch.manual_seed(p['random_seed'])
            active['arm']=arm
            save(OUT/'progress.json',dict(state='running',arm=arm,pid=os.getpid(),updated=now()))
            namespace['run'](arguments(arm))
        parity={}
        with np.load(BODY/'neighbor-seed18/frames/frame_000018/winner-camera-space.npz',allow_pickle=False) as old, np.load(OUT/'body_only/frames/frame_000018/winner-camera-space.npz',allow_pickle=False) as new:
            for key in ['vertices_camera','joints_projected_native','global_orient','camera_translation','pose_embedding']:
                parity[key]=dict(exact=bool(np.array_equal(old[key],new[key])),max_abs=float(np.max(np.abs(old[key]-new[key]))))
        assert parity['vertices_camera']['max_abs']<=1e-6 and parity['joints_projected_native']['max_abs']<=1e-3,parity
        verify_plan()
        save(OUT/'result.json',dict(state='complete_pending_review',updated=now(),elapsed_seconds=time.monotonic()-start,
            frames=FRAMES,arms=ARMS,traces=traces,historical_body_only_parity=parity,
            matched_camera_initialization=True,plan_sha256=sha(PLAN),quality_accepted=False,
            full59_admitted=False,gaussian_training_started=False))
        save(OUT/'progress.json',dict(state='complete',pid=None,updated=now()))
    except BaseException as e:
        save(OUT/'failure.json',dict(updated=now(),error=repr(e),arm=active.get('arm')))
        save(OUT/'progress.json',dict(state='failed',pid=None,updated=now()))
        raise
    finally:
        entry.adapted_entry_source,entry.load_official_config,entry.fit_one=old_source,old_config,old_fit


def run(pid):
    import fcntl
    p=verify_plan()
    assert not OUT.exists()
    lease=open('/tmp/shuolee-body-export123.lock','a+')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cmd,state,identity=common.process(pid)
    assert 'train_demo_body_upstream_offset123.py train' in cmd and state not in ('T','Z')
    receipt=dict(started=now(),supervisor_pid=os.getpid(),baseline_pid=pid,identity=identity,command=cmd,state='suspending')
    save(HANDOFF,receipt)
    try:
        os.kill(pid,signal.SIGSTOP)
        for _ in range(100):
            if common.process(pid)[1]=='T':break
            time.sleep(.05)
        assert common.process(pid)[1]=='T'
        env=dict(os.environ,SHUOLEE_CONTROL_BASELINE_PID=str(pid),SHUOLEE_CONTROL_BASELINE_IDENTITY=identity)
        with Path(__file__).with_suffix('.log').open('x',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'worker'],env=env,
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=p['runtime_budget_seconds'])
        receipt['control_finished']=True
    finally:
        try:
            if common.process(pid)[2]==identity:
                os.kill(pid,signal.SIGCONT)
                receipt.update(state='resumed',baseline_state=common.process(pid)[1])
            else:receipt['state']='baseline_identity_changed_not_signaled'
        except (FileNotFoundError,ProcessLookupError):receipt['state']='baseline_exited_before_resume'
        receipt['finished']=now();save(HANDOFF,receipt)
        print('ROOT_SEED_BASELINE',receipt['state'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','worker']);p.add_argument('--pid',type=int)
    args=p.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='run':run(args.pid)
    else:worker()
