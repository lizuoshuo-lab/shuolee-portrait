"""Bounded, fresh SH-coordinate control; never edits the running baseline.

Keep the original training function and its 600k schedule. Stop after the 10k
snapshot (the same pre-optimizer-step save point as the baseline). Only colors
use a fixed reference-root basis via the upstream override_color interface.
"""
import argparse
import ast
import copy
import difflib
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

import train_demo_body_upstream_offset123 as offset
from run_official_gaussians123 import read, save, sha, now

BASE = offset.OUT
OUT = offset.base.DEMO/'body-gaussian-root-sh-control'
HERE = Path(__file__).resolve().parent
LIMIT = 10000


class ControlComplete(Exception):
    pass


def process(pid):
    stat = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
    command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    return command, stat[0], stat[19]


def suspended_baseline():
    pid = int(os.environ['SHUOLEE_CONTROL_BASELINE_PID'])
    cmd, state, identity = process(pid)
    assert 'train_demo_body_upstream_offset123.py train' in cmd
    assert state == 'T' and identity == os.environ['SHUOLEE_CONTROL_BASELINE_IDENTITY']


def source():
    _, _, original = offset.revised_train()
    changed = original
    def replace(old, new):
        nonlocal changed
        assert changed.count(old) == 1, old
        changed = changed.replace(old, new, 1)
    replace("    lock = open('/tmp/shuolee-native122-gpu.lock', 'a+')\n    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)",
            "    suspended_baseline()  # Parent owns the export lease and resumes the baseline in finally.")
    replace('    from gaussian_renderer import render', '    from gaussian_renderer import render as original_render\n    from body_root_sh_colors123 import render_root_frame')
    replace("    assert layout['reference_frame_id'] == layout['initialization_frame_id'] == 'frame_000033'", """    assert layout['reference_frame_id'] == layout['initialization_frame_id'] == 'frame_000033'
    rotations = torch.as_tensor(data['root_rotation'], device='cuda', dtype=torch.float32)
    root_basis = rotations[0][None] @ rotations.transpose(1, 2)
    root_basis[0] = torch.eye(3, device='cuda')
    assert not root_basis.requires_grad
    def render(cam, pc, pipe, background, **kwargs):
        if getattr(pc, 'binding', None) is None:
            return original_render(cam, pc, pipe, background, **kwargs)
        assert int(cam.timestep) == int(pc.timestep)
        return render_root_frame(original_render, cam, pc, pipe, background,
                                 root_basis[int(cam.timestep)], **kwargs)""")
    replace('            saved.append(step)', """            save(folder/'sampling.json', dict(counts=counts.tolist(), seed=0,
                total_samples=int(counts.sum()), only_color_direction_changed=True))
            saved.append(step)""")
    replace('            if step == 1000:', """            if step == 10000:
                suspended_baseline()
                save(run/'result.json', dict(updated=now(), state='bounded_control_completed',
                    snapshot_step=step, optimizer_updates=step-1, configured_schedule_steps=opt.iterations,
                    points=len(model._xyz), seconds=time.monotonic()-started,
                    snapshots=saved, per_frame_counts=counts.tolist(), quality_accepted=False,
                    end_to_end_15min_accepted=False, stop_reason='Preregistered10k comparison budget'))
                status('bounded_control_completed', step, optimizer_waiting=False)
                raise ControlComplete()
            if step == 1000:""")
    replace("    namespace['training'](dataset,opt,pipe,[],SNAPSHOTS,[],None,-1)", """    try:
        namespace['training'](dataset,opt,pipe,[],SNAPSHOTS,[],None,-1)
    except ControlComplete:
        return""")
    namespace = dict(offset.base.train.__globals__)
    namespace.update(OUT=OUT, SNAPSHOTS=[1000, LIMIT], suspended_baseline=suspended_baseline,
                     ControlComplete=ControlComplete)
    exec(compile(changed, '<body-root-SH-color-only-control>', 'exec'), namespace)
    return namespace['train'], original, changed


def prepare():
    assert not OUT.exists(), 'Preserve any previous attempt; do not repeat automatically.'
    p = read(BASE/'plan.json')
    for f, h in p['source_hashes'].items():
        assert sha(f) == h, f
    probe = BASE/'root-gauge-audit/colour-only-probe/result.json'
    result = read(probe)
    assert result['passed'] and result['frames'] == 59 and result['gaussian_updates'] == 0
    OUT.mkdir()
    for sub in ['fit/fitted.npz', 'data/training-inputs.json', 'vhap-gaussian-interface-receipt.json']:
        target = OUT/sub
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(BASE/sub, target)
        assert sha(target) == sha(BASE/sub)
    _, original, changed = source()
    (OUT/'control-launcher-function.py').write_text(changed)
    (OUT/'color-direction-only.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), changed.splitlines(True), fromfile='baseline', tofile='color-control')))
    p = copy.deepcopy(p)
    p.update(updated=now(), state='prepared', baseline_run=str(BASE),
        only_parameter_change='SH direction rotated to frozen reference-root basis via override_color; no parameter values changed.',
        root_rotation_source='Original frozen fitted.npz root_rotation; coordinate metadata, not independently recovered camera poses.',
        comparison_budget=LIMIT, snapshots=[1000, LIMIT], no_automatic_new_experiment=True,
        configured_schedule_steps=600000, from_scratch=True, old_gaussian_checkpoint_loaded=False,
        initial_points=20908, seed=0, quality_accepted=False,
        scope='Single radiance-coordinate diagnostic; does not repair known limb geometry or prove generalization.',
        criteria=dict(preflight_finite=True, unchanged_options=True, review_at_1000=True,
            stop_at_10000=True, review_all59_and_same36_novel_views=True,
            forbid_promotion_without_native_and_UI_checks=True),
        proof_of_interface=dict(path=str(probe),sha256=sha(probe)),
        serialization='Reference33 only: SH basis equals original coordinates there. Other poses require an explicit SH transform.',
        runtime_budget_seconds=1800)
    for f in [Path(__file__).resolve(), HERE/'body_root_sh_colors123.py']:
        p['source_hashes'][str(f)] = sha(f)
    save(OUT/'plan.json', p)
    save(OUT/'progress.json', dict(updated=now(),state='prepared',pid=None,step=0,quality_accepted=False))
    ast.parse(changed)
    assert p['options'] == read(BASE/'plan.json')['options']
    print('ROOT_SH_CONTROL_PREPARED_ORIGINAL_OPTIONS_EQUAL', flush=True)


def run(pid):
    assert not (OUT/'gaussian-run').exists(), 'Never resume or repeat this control.'
    p=read(OUT/'plan.json')
    for f,h in p['source_hashes'].items(): assert sha(f)==h,f
    lock=open('/tmp/shuolee-body-export123.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cmd,state,identity=process(pid)
    assert 'train_demo_body_upstream_offset123.py train' in cmd and state not in ('T','Z')
    started=time.monotonic()
    receipt=dict(started=now(),pid=pid,command=cmd,identity=identity,state='suspending')
    save(OUT/'gpu-handoff.json',receipt)
    try:
        os.kill(pid,signal.SIGSTOP)
        for _ in range(100):
            if process(pid)[1]=='T':break
            time.sleep(.05)
        assert process(pid)[1]=='T'
        env=dict(os.environ,SHUOLEE_CONTROL_BASELINE_PID=str(pid),SHUOLEE_CONTROL_BASELINE_IDENTITY=identity)
        subprocess.run([sys.executable,'-B',__file__,'train'],env=env,check=True,timeout=p['runtime_budget_seconds'])
    finally:
        try:
            current=process(pid)
            if current[2]==identity:
                os.kill(pid,signal.SIGCONT)
                receipt.update(state='resumed',trainer_after_state=process(pid)[1])
            else: receipt.update(state='identity_changed_not_signalled')
        except FileNotFoundError:receipt.update(state='baseline_exited')
        receipt.update(finished=now(),seconds=time.monotonic()-started)
        save(OUT/'gpu-handoff.json',receipt)
        print('BASELINE_HANDOFF',receipt['state'],pid,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','run','train'])
    parser.add_argument('--pid',type=int)
    args=parser.parse_args()
    try:
        if args.mode=='prepare':prepare()
        elif args.mode=='run':run(args.pid)
        else:source()[0]()
    except BaseException as error:
        if OUT.exists():
            save(OUT/(args.mode+'-failure.json'),dict(updated=now(),error=repr(error),traceback=traceback.format_exc(),auto_retry=False))
            save(OUT/'progress.json',dict(updated=now(),state='failed_preserved',pid=None,error=repr(error),quality_accepted=False))
        raise
