"""Reuse the existing original fitter, temporal loss and body/root seed hooks.

One registered sofa run. No solver replacement, historical pose initializers,
weight search, visibility classifier, or post-hoc frame replacement.
"""
import argparse
import os
from pathlib import Path
import numpy as np
import official_smplifyx_entry123 as entry
import control_smplifyx_root_seed123 as seed
import run_smplify_temporal124 as temporal
from smplify_observation_validity124 import mask_outside_raster, verify
from run_official_gaussians123 import read, save, sha, now

HERE = Path(__file__).resolve().parent
SRC = temporal.ROOT/'sofa-input'
OUT = temporal.ROOT/'sofa-temporal'


def prepare():
    assert not OUT.exists(), 'Preserve the single registered candidate.'
    review = read(SRC/'reference-review/result.json')
    assert review['actual_visual_review'] and review['full_sequence_diagnostic_admitted']
    sel = read(SRC/'selection.json')
    rows = read(SRC/'data/inputs.json')['records']
    ids = [r['frame'] for r in rows]
    ts = np.array([r['time_seconds'] for r in rows])
    assert len(ids) == len(set(ids)) == 56 and np.all(np.diff(ts) > 0)
    for r in rows:
        assert np.array_equal(r['w2c'], np.eye(4))
        assert np.array_equal(r['K_integer_centers'], rows[0]['K_integer_centers'])
        assert sha(r['image']) == r['source_sha256'] and sha(r['mask']) == r['mask_sha256']
    ref = ids.index(sel['reference_frame'])
    order = sorted(range(len(rows)), key=lambda i:(abs(ts[i]-ts[ref]), ts[i], ids[i]))
    entry._root_seed_original_source = entry.adapted_entry_source
    seed.adapted_source(temporal.REC/'smplifyx-reference-control/official')
    old = read(temporal.ROOT/'winterhat-continuation/plan.json')
    for path in [Path(temporal.__file__), Path(seed.__file__), Path(entry.__file__), HERE/'smplify_temporal124.py']:
        assert sha(path) == old['sources'][str(path)], path
    frozen = [Path(__file__), Path(temporal.__file__), Path(seed.__file__), Path(entry.__file__),
        HERE/'smplify_temporal124.py', HERE/'smplify_observation_validity124.py',
        HERE/'run_official_sequence123.py', HERE/'run_official_sequence123_topology.py',
        SRC/'data/inputs.json', SRC/'data/observations.npz', SRC/'selection.json',
        SRC/'length-contract.json', SRC/'reference-fit/winner-camera-space.npz']
    plan = dict(updated=now(), state='registered', subject='sofa', source_root=str(SRC),
        hypothesis='Test the same frozen temporal/continuation interface on an independently fitted seated subject. Winterhat failure is retained; this is cross-subject verification, not proof of a repair.',
        method='Original five stages, LBFGS, maxiters30, full K, VPoser, bending prior; existing body/root temporal weights30 and same-run adjacent initialization after original camera initialization.',
        reference_frame=sel['reference_frame'], ordered_fit_frames=[ids[i] for i in order],
        source_roster=56, training_frames=sel['training_frames'], review_only_frames=sel['review_only_frames'],
        timestamp_seconds=ts.tolist(), nominal_dt_seconds=float(np.median(np.diff(ts))),
        shape='Shared subject beta from the reviewed original reference26. Original zero/mean anchor; no old pose or fitted vertices reused.',
        temporal=dict(body_weight=30., root_weight=30., time_weight='(nominal_dt/actual_dt)^2',
            neighbour='Nearest completed same-run chronological neighbour, detached',
            bidirectional_joint_solver=False, camera_translation=False, face_fingers=False),
        input_change='Only out-of-raster confidence is zero; coordinates and in-raster confidence unchanged. Raster validity is not visibility. No Haar gate or hand confidence experiment.',
        camera_gauge='Identity extrinsic camera-relative body gauge, not a recovered world camera path.',
        limits=['No same-subject full independent baseline: do not claim isolated temporal causal improvement.',
                'Temporal prior can propagate an anatomically wrong pose; side views must be reviewed.',
                'Furniture occlusion and missing shoe mask must not become complete-body negative supervision.'],
        budget=dict(candidates=1, frames=56, max_wall_seconds=3600, review_after_frames=13, review_wait_seconds=900),
        stopping='Pause at13 fits for all completed native body/face/feet and synthetic side review. Stop on severe anatomy failure, nonfinite state, source mismatch or wall budget. Review all56 before any Gaussian. No weight sweep or stitched winners.',
        quality_accepted=False, gaussian_admitted=False,
        sources={str(p):sha(p) for p in frozen})
    OUT.mkdir()
    save(OUT/'plan.json', plan)
    save(OUT/'interface-check.json', dict(updated=now(), raster_checks=verify(),
        temporal_interface_checks=str(temporal.ROOT/'winterhat-temporal/interface-check-repaired.json'),
        same_frozen_temporal_source=True, fullK_and_identity_gauge_checked=True, real_quality_proven=False))
    print('SOFA_TEMPORAL_REGISTERED', plan['ordered_fit_frames'][:13], flush=True)


def run():
    p = read(OUT/'plan.json')
    assert not (OUT/'official-fit').exists()
    for path, digest in p['sources'].items(): assert sha(path) == digest, path
    rows = {r['frame']:r for r in read(SRC/'data/inputs.json')['records']}
    old_source, old_config, old_fit = entry.adapted_entry_source, entry.load_official_config, entry.fit_one
    old_context = temporal.context
    entry._root_seed_original_source = old_source
    bank, active = {}, {}

    def config(root):
        c = old_config(root)
        assert c['maxiters'] == 30
        c.update(_seed_pose_embedding=active['latent'], _seed_global_orient=active['root'],
                 _seed_root_enabled=active['neighbour'] is not None)
        return c

    def fit(**kwargs):
        frame = Path(kwargs['result_path']).parent.name
        assert frame == p['ordered_fit_frames'][len(bank)]
        neighbour = min(bank, key=lambda n:(abs(rows[n]['time_seconds']-rows[frame]['time_seconds']), n)) if bank else None
        active.update(neighbour=neighbour, latent=np.zeros((1,32),np.float32), root=np.zeros((1,3),np.float32))
        if neighbour:
            assert abs(int(frame[-6:])-int(neighbour[-6:])) == 1
            active.update(latent=bank[neighbour]['pose_embedding'].copy(), root=bank[neighbour]['global_orient'].copy())
        before = kwargs['keypoints'].copy()
        guarded, inside = mask_outside_raster(before, rows[frame]['wh'])
        kwargs['keypoints'][...] = guarded
        arrays, receipt = old_fit(**kwargs)
        trace = {k:v.tolist() for k,v in entry._root_seed_trace.items()}
        assert np.array_equal(arrays['camera_translation'], np.asarray(trace['camera_translation']))
        if neighbour: assert np.array_equal(trace['body_fit_initial_root'], active['root'])
        bank[frame] = {k:arrays[k].copy() for k in ('pose_embedding','global_orient')}
        receipt['continuation'] = dict(neighbour=neighbour, same_run_only=True, initial_state=trace,
            initial_body_latent=active['latent'].tolist(), camera_initialization_unchanged=True)
        receipt['raster_validity'] = dict(zeroed_indices=np.flatnonzero((before[0,:,2]>0)&(~inside[0])).tolist(),
            coordinates_unchanged=np.array_equal(before[...,:2], guarded[...,:2]), visibility_proven=False)
        return arrays, receipt

    temporal.context = lambda subject:(SRC, OUT, read(SRC/'selection.json'))
    entry.adapted_entry_source, entry.load_official_config, entry.fit_one = seed.adapted_source, config, fit
    try:
        temporal.run('sofa')
        for path, digest in p['sources'].items(): assert sha(path) == digest, path
        path = OUT/'official-fit/input-provenance.json'
        q = read(path); q['initialization'] = 'Original zero/mean anchor, then same-run adjacent body/root after original camera initialization'
        q['raster_validity_applied'] = True; save(path,q)
    except BaseException as exc:
        save(OUT/'failure.json', dict(updated=now(), error=repr(exc), computed_frames=len(bank)))
        path = OUT/'temporal-progress.json'
        if path.exists():
            q = read(path);q.update(state='stopped_or_failed',pid=None,error=repr(exc));save(path,q)
        raise
    finally:
        temporal.context = old_context
        entry.adapted_entry_source, entry.load_official_config, entry.fit_one = old_source, old_config, old_fit


if __name__ == '__main__':
    ap = argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run']);args=ap.parse_args()
    (prepare if args.mode=='prepare' else run)()
