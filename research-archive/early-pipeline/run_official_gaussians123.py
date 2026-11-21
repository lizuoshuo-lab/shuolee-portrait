"""Single fresh Gaussian run on this cycle's completed official 94-frame fit.

Keeps the existing SMPLXGaussianModel, optimizer, losses and densification.
At step 1000 the same live optimizer waits for a hash-bound review decision.
This script does not run fitting, download assets or grant quality acceptance.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import sys
import time
import traceback


PROJECT = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
ASSETS = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
CFG = dict(initial_points=200000, iterations=18000, training_height=899,
           seed=6060, image_cache_frames=8, export_frame_mode='scene_up_keep_axis_order')
SOURCES = {
    'training.py': 'ea5520249cc931e9d63e6ec59d62bcd6ac33fff52ca787d4ae0c950f61830d99',
    'export.py': '4da488438b5fcfb40f6a028d5bab9952c223a522f3446d9a7ba6344d5ead1520',
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def snapshot_identity(snapshot):
    files = [dict(name=p.name, sha256=sha(p)) for p in sorted(Path(snapshot).iterdir()) if p.is_file()]
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(snapshot_sha256=digest, files=files, world_ply_sha256=sha(Path(snapshot)/'world.ply'))


def review_decision(path, request):
    decision = read(path)
    if (decision.get('step') != request['step'] or
            decision.get('snapshot_sha256') != request['snapshot_sha256'] or
            decision.get('decision') not in ('continue', 'stop')):
        raise ValueError('Review decision must identify this exact step-1000 snapshot and say continue or stop.')
    return decision


def run(args):
    if os.name != 'posix':
        raise RuntimeError('Use the existing Linux/WSL Gaussian environment.')
    import fcntl
    root, project = Path(args.root).resolve(), Path(args.project_root).resolve()
    if not root.is_dir():
        raise ValueError('Completed official run directory is missing.')
    for name, digest in SOURCES.items():
        if sha(project/'src/shuolee_portrait'/name) != digest:
            raise ValueError('Reviewed Gaussian interface changed: '+name)
    sys.path.insert(0, str(project/'src'))
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.training import verify_official_fit, train_exploratory_official_smplx
    from shuolee_portrait.export import export
    # Nonblocking: a live fitter or another GPU consumer is never queued/restarted.
    with open('/tmp/shuolee-native122-gpu.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        provenance = verify_official_fit(root)
        receipt = read(root/'fit/official-fit-result.json')
        rows = read(root/'data/training-inputs.json')['records']
        if receipt.get('frames') != 94 or receipt.get('full_roster') is not True or len(rows) != 94:
            raise ValueError('This launcher requires the completed full 94-frame official run.')
        if receipt.get('state') != 'complete':
            raise ValueError('The official fitting process has not recorded terminal completion.')
        if (root/'gaussian-run').exists() or (root/'export').exists():
            raise ValueError('Fresh Gaussian run only; existing outputs are never resumed or overwritten.')
        record = root/'gaussian-launch'
        record.mkdir(exist_ok=False)
        started = time.monotonic()
        def status(state, **fields):
            save(record/'progress.json', dict(state=state, updated=now(), pid=os.getpid(),
                quality_accepted=False, elapsed_seconds=time.monotonic()-started, **fields))
        def checkpoint(step, snapshot):
            if step != 1000:
                return True  # Existing 6k/12k/18k snapshot behavior is retained.
            identity = snapshot_identity(snapshot)
            request = dict(step=step, snapshot=str(snapshot), requested=now(), **identity)
            reviews = root/'reviews'
            decision_path = reviews/'gaussian-step-001000-decision.json'
            if decision_path.exists():
                raise ValueError('A decision already existed before this snapshot was offered for review.')
            save(reviews/'gaussian-step-001000-request.json', request)
            status('review_pending', step=step, optimizer_waiting=True, snapshot=str(snapshot),
                   snapshot_sha256=identity['snapshot_sha256'], decision_path=str(decision_path))
            print(json.dumps(dict(state='review_pending', request=str(reviews/'gaussian-step-001000-request.json'),
                                  decision_path=str(decision_path))), flush=True)
            while not decision_path.exists():
                time.sleep(2)
            decision = review_decision(decision_path, request)
            if snapshot_identity(snapshot) != identity:
                raise ValueError('Snapshot files changed while review was pending.')
            save(reviews/'gaussian-step-001000-consumed.json', dict(consumed=now(),
                 decision_sha256=sha(decision_path), **decision))
            proceed = decision['decision'] == 'continue'
            status('training' if proceed else 'stopped_after_review', step=step, optimizer_waiting=False,
                   decision_sha256=sha(decision_path), same_optimizer_state=True)
            return proceed
        try:
            save(record/'input.json', dict(created=now(), config=CFG, official_fit=provenance,
                 launcher_sha256=sha(__file__), interface_sources=SOURCES,
                 original_fitting_result_preserved=True, no_fit_restart=True,
                 from_scratch=True, quality_accepted=False))
            rt.configure(Path(args.assets_root).resolve(), root)
            os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
            status('training', step=0, optimizer_waiting=False)
            train_exploratory_official_smplx(root, CFG, checkpoint_review=checkpoint)
            trained = read(root/'gaussian-run/result.json')
            status('exporting', step=trained['steps'], optimizer_waiting=False)
            export(root, CFG)
            result = read(root/'result.json')
            state = ('stopped_after_review_exported_quality_pending' if trained['state']=='stopped_after_actual_review'
                     else 'exported_quality_pending')
            save(record/'result.json', dict(state=state, finished=now(), result=result,
                 steps=trained['steps'], elapsed_seconds=time.monotonic()-started,
                 quality_accepted=False, viewer_ui_review='pending', auto_retry=False))
            status(state, step=trained['steps'], optimizer_waiting=False)
        except BaseException as error:
            save(record/'failure.json', dict(finished=now(), error=repr(error), traceback=traceback.format_exc(),
                 training_started=(root/'gaussian-run').exists(), auto_retry=False, quality_accepted=False))
            status('failed_preserved', error=repr(error), optimizer_waiting=False)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='Completed official full94 run directory')
    parser.add_argument('--project-root', default=str(PROJECT))
    parser.add_argument('--assets-root', default=str(ASSETS))
    run(parser.parse_args())


if __name__ == '__main__':
    main()
