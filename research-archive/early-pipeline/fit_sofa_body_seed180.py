"""Matched sofa body-initialization control using the pinned independent worker."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import time
import types

import numpy as np
from PIL import Image

import control_smplifyx_root_seed123 as seed
import official_smplifyx_entry123 as entry
from review_full_sequence123 import bound_file
from update_leafwoman129_state import ROOT, atomic, read, sha


SOURCE = ROOT / 'fixed-flow177-sofa-front24'
PRIOR = ROOT / 'sofa-half8-pose124'
OUT = ROOT / 'sofa-body-seed180'
WORKER = ROOT / 'fixed-flow178-grass-front24/run_leafwoman_front24_cpu145.py'
ARCHIVE = SOURCE / 'fit-observations.npz'
CONTACTS = SOURCE / 'contacts.json'
FRAMES = ['frame_000026', 'frame_000046']


def prepare():
    plan_path = SOURCE / 'shared-cpu/plan.json'
    old = read(plan_path)
    old_result = read(SOURCE / 'shared-cpu/result.json')
    prior_result = read(PRIOR / 'result.json')
    prior_review = read(PRIOR / 'conclusion.json')
    old_poses = {r['frame']: r for r in old_result['frames']}
    source = {r['frame']: r for r in read(CONTACTS)['records']}
    assert old_result['plan_sha256'] == sha(plan_path)
    assert old['worker_sha256'] == sha(WORKER) and old['fit_entry_sha256'] == sha(entry.__file__)
    assert old['baseline_observation_sha256'] == sha(ARCHIVE) and old['contacts_sha256'] == sha(CONTACTS)
    assert prior_review['actual_visual_review'] and prior_review['exploratory_gaussian_admitted']
    prior_archive = PRIOR / 'selected-fit.npz'
    assert sha(prior_archive) == prior_result['fit_sha256'] == prior_review['fit_sha256']
    evidence = {}
    with np.load(prior_archive, allow_pickle=False) as prev, np.load(ARCHIVE, allow_pickle=False) as current:
        assert prev['pose_embedding'].shape == (1, 32) and np.isfinite(prev['pose_embedding']).all()
        assert np.array_equal(prev['K_integer_centers'], current['K_integer_centers'])
        assert current['frame_ids'].tolist() == old['source_frames']
        for frame in FRAMES:
            row = source[frame]
            i = prev['frame_ids'].tolist().index(frame)
            j = old['source_frames'].index(frame)
            assert frame not in old['retained_holdouts']
            assert prev['source_frames'][i] == row['source_frame_index'] == old['native_indices'][j]
            assert prev['image_sha256'][i] == row['source_sha256'] == sha(bound_file(row['paths']['body']))
            x, y = prev['observed_native_keypoints'][i], current['observed_native_keypoints'][j]
            common = (x[:, 2] > 0) & (y[:, 2] > 0)
            assert common.any() and np.array_equal(x[common, :2], y[common, :2])
            baseline = Path(old_poses[frame]['output'])
            assert sha(baseline) == old_poses[frame]['sha256']
            with np.load(baseline, allow_pickle=False) as fit:
                assert np.array_equal(fit['observed_native_keypoints'][0], y)
                assert np.array_equal(fit['betas'], np.asarray(old['shared_shape']['vector'], np.float32))
            evidence[frame] = dict(source_index=row['source_frame_index'], rgb_sha256=row['source_sha256'],
                prior_and_current_common_xy_count=int(common.sum()), baseline=str(baseline),
                baseline_sha256=sha(baseline), roster_index=j)
    pins = [plan_path, SOURCE / 'shared-cpu/result.json', ARCHIVE, CONTACTS, WORKER, Path(__file__),
            Path(entry.__file__), Path(seed.__file__), prior_archive, PRIOR / 'result.json', PRIOR / 'conclusion.json']
    OUT.mkdir(exist_ok=False)
    atomic(OUT / 'plan.json', dict(updated=datetime.now(timezone.utc).isoformat(), frames=FRAMES,
        roles=['prior reference frame 026', 'prior roster late oblique view 046'],
        observation_sha256=sha(ARCHIVE), contacts_sha256=sha(CONTACTS), evidence=evidence,
        retained_holdouts=old['retained_holdouts'], original_fit_plan=old,
        changed_variable='post-camera-init VPoser latent, zero to archived sofa eight-view common latent',
        independent_body_pose=True, change_root=False, change_beta=False, change_observations=False,
        solver='Pinned original five stages on CPU, one process and one Torch thread',
        exact_zero_control='Replay 026 and require bit-identical arrays before evaluating warm start.',
        budget_seconds=240, gaussian_started=False, quality_accepted=False,
        input_hashes={str(p): sha(p) for p in pins}))


def run():
    plan_path = OUT / 'plan.json'
    plan = read(plan_path)
    assert all(sha(path) == digest for path, digest in plan['input_hashes'].items())
    worker = entry.module(WORKER, '_sofa180_archived_cpu_worker')
    with np.load(PRIOR / 'selected-fit.npz', allow_pickle=False) as fit:
        latent = fit['pose_embedding'].copy()
    rows = {r['source_frame_index']: r['paths']['body'] for r in read(CONTACTS)['records']}

    def decode(indices, box, reverse):
        assert not reverse and box == (0, 0, 2704, 1520)
        for index in indices:
            with Image.open(bound_file(rows[index])) as rgb:
                yield index, np.asarray(rgb.convert('RGB'))

    original_fit, original_config = entry.fit_one, entry.load_official_config
    entry._root_seed_original_source = entry.adapted_entry_source
    entry.adapted_entry_source = seed.adapted_source
    state = {'latent': latent}
    traces = {}

    def config(root):
        return dict(original_config(root), _seed_pose_embedding=state['latent'],
                    _seed_root_enabled=False)

    def fit_one(**kwargs):
        frame = Path(kwargs['result_path']).parent.name
        if frame == FRAMES[0]:
            zero = OUT / 'zero-replay-026'
            zero.mkdir()
            state['latent'] = np.zeros_like(latent)
            arrays, _ = original_fit(**dict(kwargs, result_path=zero/'official-result.pkl',
                                             diff_path=zero/'upstream-adapter.diff'))
            with np.load(plan['evidence'][frame]['baseline'], allow_pickle=False) as baseline:
                equality = {k: bool(np.array_equal(v, baseline[k])) for k, v in arrays.items()}
                max_abs = {k: float(np.max(np.abs(v - baseline[k]))) for k, v in arrays.items()}
            assert all(equality.values()), max_abs
            zero_trace = entry._root_seed_trace.copy()
            atomic(zero / 'result.json', dict(array_equal=equality, max_abs=max_abs,
                camera_prefix={k: v.tolist() for k, v in zero_trace.items()}))
            state['latent'] = latent
        arrays, receipt = original_fit(**kwargs)
        trace = entry._root_seed_trace
        assert np.array_equal(trace['camera_initial_root'], trace['body_fit_initial_root'])
        if frame == FRAMES[0]:
            assert all(np.array_equal(v, zero_trace[k]) for k, v in trace.items())
        traces[frame] = {k: v.tolist() for k, v in trace.items()}
        return arrays, dict(receipt, initialization='Archived sofa body latent after unchanged camera initialization')

    entry.fit_one, entry.load_official_config = fit_one, config
    scope = dict(worker.__dict__, ARCHIVE=ARCHIVE, CONTACTS=CONTACTS, decode=decode)
    fit = types.FunctionType(worker.fit_frames.__code__, scope)
    atomic(OUT / 'launch.json', dict(pid=os.getpid(), updated=datetime.now(timezone.utc).isoformat()))
    start = time.monotonic()
    try:
        fit(OUT, 0, plan['original_fit_plan'], [plan['evidence'][f]['roster_index'] for f in FRAMES])
        outputs = [read(OUT/f/'receipt.json') for f in FRAMES]
        assert all(sha(r['output']) == r['sha256'] for r in outputs)
        atomic(OUT / 'result.json', dict(plan_sha256=sha(plan_path), frames=outputs,
            seconds=time.monotonic()-start, successful=True, zero_026_bit_identical=True,
            zero_026_camera_prefix_bit_identical=True, camera_prefix=traces, actual_visual_review=False,
            gaussian_started=False, quality_accepted=False))
    except BaseException as error:
        atomic(OUT / 'failure.json', dict(error=repr(error), seconds=time.monotonic()-start))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'run'))
    {'prepare': prepare, 'run': run}[parser.parse_args().phase]()
