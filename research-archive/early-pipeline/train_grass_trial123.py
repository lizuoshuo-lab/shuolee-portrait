"""Fresh, explicitly authorized free-surface trial using the frozen sofa route.

This is an exploratory run, not a physical-camera or subpixel acceptance.
Actual, hash-bound image review is required before each dependent stage.
"""
import argparse
import fcntl
import os
import sys
import time
import traceback
from trial_grass_surface123 import ROOT, OUT, REC, R, ASSETS, PROJECT, read, save, sha, now, status, verify_runtime


def decision(phase, identity, key):
    path = ROOT / 'reviews' / (phase + '-decision.json')
    d = read(path)
    if not d.get('actual_visual_review') or any(d.get(k) != v for k, v in identity.items()):
        raise RuntimeError('Review does not match the current output: ' + phase)
    return d.get(key) is True


def checkpoint(step, snapshot):
    if step == 18000:
        return True
    phase = f'gaussian-step-{step:06d}'
    identity = dict(snapshot_sha256=sha(snapshot / 'world.ply'))
    status('gaussian_review_pending', step=step, optimizer_waiting=True, snapshot=str(snapshot), **identity)
    for folder in [REC, R]:
        q = read(folder / 'queue-state.json')
        q.update(training_running=True, optimizer_waiting=True, step=step)
        save(folder / 'queue-state.json', q)
    deadline = time.monotonic() + 7200
    while time.monotonic() < deadline:
        if (ROOT / 'reviews' / (phase + '-decision.json')).exists():
            proceed = decision(phase, identity, 'continue_same_run')
            status('gaussian_training' if proceed else 'gaussian_stopped_after_review', step=step)
            return proceed
        time.sleep(5)
    save(ROOT / 'reviews' / (phase + '-timeout.json'), dict(updated=now(), continued=False, **identity))
    return False


def main(phase):
    verify_runtime()
    from shuolee_portrait import runtime as rt
    rt.configure(ASSETS, ROOT)
    cfg = read(ROOT / 'config.json')
    assert cfg['initial_points'] == 200000 and cfg['iterations'] == 18000
    if phase == 'surface':
        if not decision('fit', dict(fit_sha256=sha(ROOT / 'fit/fitted.npz')), 'allow_surface_initialization'):
            raise RuntimeError('Coarse extent review stopped initialization.')
        assert not (ROOT / 'clothing-init').exists(), 'Never overwrite an existing initialization.'
        from shuolee_portrait.clothed_surface import initialize
        status('fresh_clothing_surface')
        initialize(ROOT, cfg)
        status('surface_review_pending', points=200000)
    elif phase == 'train':
        if not decision('surface', dict(points_sha256=sha(ROOT / 'clothing-init/points.npz')), 'allow_finite_research_training'):
            raise RuntimeError('Surface review stopped training.')
        assert not (ROOT / 'gaussian-run').exists(), 'Fresh trial only; do not resume or replace an existing run.'
        from shuolee_portrait.clothed_training import train_clothed
        status('gaussian_training', step=0)
        train_clothed(ROOT, cfg, checkpoint_review=checkpoint)
        status('exporting', result=read(ROOT / 'gaussian-run/result.json'))
        from shuolee_portrait.export import export
        export(ROOT, cfg)
        save(OUT / 'result.json', read(ROOT / 'result.json'))
        status('exported_quality_UI_review_pending', result=read(ROOT / 'result.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['surface', 'train'])
    args = parser.parse_args()
    lock = open('/tmp/shuolee-native122-gpu.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        main(args.phase)
    except Exception as exc:
        save(OUT / 'failure.json', dict(updated=now(), phase=args.phase, error=repr(exc),
             traceback=traceback.format_exc(), training_started=(ROOT / 'gaussian-run').exists(), auto_retry=False))
        status('failed_preserved', error=repr(exc))
        raise
