"""Grass adapter for the existing pinned SMPLify-X temporal runner.

The only new fitting interface is camera-frame transport of the same-run
neighbour root. Original stages, priors, solver and temporal weights remain.
"""
import argparse
import os
import shutil
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import official_smplifyx_entry123 as entry
import control_smplifyx_root_seed123 as seed
import run_smplify_temporal124 as temporal
from smplify_observation_validity124 import mask_outside_raster
from run_official_gaussians123 import read, save, sha, now

HERE = Path(__file__).resolve().parent
ROOT, REC = temporal.ROOT, temporal.REC
SRC, OUT = ROOT/'grass-input', ROOT/'grass-temporal'
RAW = Path('/opt/gs-slam-lab/shuolee-local-surface-123/grass-exploratory')
BASE = REC/'smplifyx-reference-control/official-first-cycle/full94-shared-shape'


def transport(root, previous_E, current_E):
    delta = np.asarray(current_E)[:3, :3] @ np.asarray(previous_E)[:3, :3].T
    rotation = delta @ Rotation.from_rotvec(np.asarray(root).reshape(3)).as_matrix()
    return Rotation.from_matrix(rotation).as_rotvec().reshape(1, 3).astype(np.float32)


def check_transport():
    ea, eb = np.eye(4), np.eye(4)
    ea[:3, :3] = Rotation.from_rotvec([.2, -.4, .1]).as_matrix()
    eb[:3, :3] = Rotation.from_rotvec([-.3, .5, -.2]).as_matrix()
    world = Rotation.from_rotvec([.4, .1, -.3]).as_matrix()
    previous = Rotation.from_matrix(ea[:3, :3] @ world).as_rotvec()
    result = transport(previous, ea, eb)
    got = Rotation.from_rotvec(result[0]).as_matrix()
    checks = dict(same_world_orientation=np.allclose(got, eb[:3, :3] @ world, atol=1e-7),
        identity_camera=np.allclose(transport(previous, ea, ea)[0], previous, atol=1e-7),
        reverse_transport=np.allclose(transport(result, eb, ea)[0], previous, atol=1e-7))
    # Chordal root error is invariant under a common camera/world rotation.
    other = Rotation.from_rotvec([-.1, .7, .2]).as_matrix()
    a = .5*np.square(eb[:3, :3] @ other-got).sum()
    b = .5*np.square(other-world).sum()
    checks['world_camera_loss_equivalence'] = abs(a-b) < 1e-7
    assert all(checks.values()), checks
    return {key:bool(value) for key,value in checks.items()}


def prepare():
    assert not SRC.exists() and not OUT.exists(), 'Preserve every registered candidate.'
    old = read(BASE/'input-provenance.json')
    # Original observations/cameras only; historical per-frame poses stay closed.
    raw_paths = [RAW/'data/inputs.json', RAW/'data/observations.npz']
    for p in raw_paths: assert sha(p) == old['input_files'][str(p)], p
    beta = Path(old['shared_betas_path'])
    assert sha(beta) == old['shared_betas_sha256']
    rows = read(raw_paths[0])['records']
    ids = [r['frame'] for r in rows]
    ts = np.array([r['time_seconds'] for r in rows])
    assert len(ids) == len(set(ids)) == 94 and np.all(np.diff(ts) > 0)
    assert [int(n[-6:]) for n in ids] == list(range(1, 95))
    assert np.all(np.diff([r['source_frame_index'] for r in rows]) > 0)
    for r in rows:
        assert sha(r['image']) == r['source_sha256'] and sha(r['mask']) == r['mask_sha256']
        assert r['wh'] == [2704, 1520]
        k = np.array(r['K_integer_centers']); k[:2, 2] += .5
        assert np.array_equal(k, r['K'])
        R = np.array(r['w2c'])[:3, :3]
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-6) and abs(np.linalg.det(R)-1) < 1e-6
    with np.load(raw_paths[1], allow_pickle=False) as z:
        assert z['frame_ids'].astype(str).tolist() == ids
    ref = old['reference_frame_id']; assert ref == 'frame_000032'
    order = sorted(ids, key=lambda n:(abs(ts[ids.index(n)]-ts[ids.index(ref)]), ts[ids.index(n)], n))
    selected = read(ROOT/'grass/selection.json')['selected_frames']
    assert len(selected) == 8 and set(selected) <= set(ids)
    (SRC/'data').mkdir(parents=True); (SRC/'reference-fit').mkdir(); OUT.mkdir()
    for p in raw_paths: shutil.copyfile(p, SRC/'data'/p.name)
    shutil.copyfile(beta, SRC/'reference-fit/winner-camera-space.npz')
    save(SRC/'selection.json', dict(reference_frame=ref, training_frames=selected,
        review_only_frames=[n for n in ids if n not in selected], no_new_frame_selection=True))
    save(SRC/'length-contract.json', dict(
        frames=dict(path=str(SRC/'data/inputs.json'), sha256=sha(raw_paths[0])),
        observations=dict(path=str(SRC/'data/observations.npz'), sha256=sha(raw_paths[1])),
        camera_length_contract=old['source_camera_length_contract']))
    entry._root_seed_original_source = entry.adapted_entry_source
    seed.adapted_source(REC/'smplifyx-reference-control/official')
    sources = [Path(__file__), Path(temporal.__file__), Path(seed.__file__), Path(entry.__file__),
        HERE/'smplify_temporal124.py', HERE/'smplify_observation_validity124.py',
        HERE/'run_official_sequence123.py', HERE/'run_official_sequence123_topology.py',
        SRC/'data/inputs.json', SRC/'data/observations.npz', SRC/'length-contract.json',
        SRC/'selection.json', SRC/'reference-fit/winner-camera-space.npz']
    sources += [REC/'smplifyx-reference-control/official'/name for name in entry.PINNED]
    save(OUT/'plan.json', dict(updated=now(), state='registered', subject='grass',
        user='SMPLify这个也加上连续帧约束', source_root=str(SRC), source_roster=94,
        source_video=read(ROOT/'grass/plan.json')['source_video'], record_directory='/mnt/d/00/记录/005',
        reference_frame=ref, ordered_fit_frames=order, timestamp_seconds=ts.tolist(),
        nominal_dt_seconds=float(np.median(np.diff(ts))), training_frames=selected,
        review_only_frames=[n for n in ids if n not in selected],
        hypothesis='Same-run adjacent body/root initialization plus a rotation continuity term may reduce monocular depth switches. Nonidentity camera rotations must transport the neighbouring root.',
        method='Reuse original SMPLify-X five stages, maxiters30, LBFGS, VPoser/bending priors, full native K and frozen existing temporal weights30. No stage-tail changes, new solver or loss sweep.',
        temporal=dict(body_weight=30., root_weight=30., time_weight='(median_dt/actual_dt)^2',
            root_transport='R_E_current @ R_E_neighbour.T @ R_root_neighbour_camera',
            neighbour='Nearest completed same-run chronological neighbour, detached',
            camera_translation=False, face_fingers=False, bidirectional_joint_solver=False),
        shape='Reuse only reference32 shared betas for comparison; first anchor pose starts at official zero/mean. Do not reuse old per-frame poses or geometry. Not a cold-start end-to-end benchmark.',
        input_change='Existing raster-validity confidence guard only; original coordinates, in-raster scores, K and E are retained. No face/hand confidence ablation.',
        comparison='All completed same-frame independent94 baselines, native body/face/feet and common synthetic sides. Temporal loss plus same-run initialization is a combined treatment.',
        calibration='Historical E remains a conditional research estimate. Coordinate transport parity does not prove physical calibration.',
        budget=dict(candidates=1, frames=94, max_wall_seconds=5400, review_after_frames=13,
            review_wait_seconds=900, additional_review_after_frames=27),
        stopping='Stop at actual anatomy/depth failure, nonfinite state, source mismatch or budget. Pause13, then27 (includes known44/45 switch) before all94. No threshold scan, old-pose fallback, stitched winners or Gaussian without all-frame pose acceptance.',
        quality_accepted=False, gaussian_admitted=False, sources={str(p):sha(p) for p in sources}))
    save(OUT/'interface-check.json', dict(updated=now(), transport=check_transport(),
        original_observations_bytes_preserved=True, real_PTS_order=True, complete_native_K=True,
        original_temporal_checks=str(ROOT/'winterhat-temporal/interface-check-repaired.json'),
        original_pose_used_as_initialization=False, quality_accepted=False))
    print('GRASS_TEMPORAL_REGISTERED', len(rows), order[:13], flush=True)


def run():
    p = read(OUT/'plan.json')
    assert all(read(OUT/'interface-check.json')['transport'].values())
    assert not (OUT/'official-fit').exists()
    for path, digest in p['sources'].items(): assert sha(path) == digest, path
    rows = {r['frame']:r for r in read(SRC/'data/inputs.json')['records']}
    old_source, old_config, old_fit = entry.adapted_entry_source, entry.load_official_config, entry.fit_one
    old_context, old_loss = temporal.context, temporal.install_loss_factory
    entry._root_seed_original_source = old_source
    bank, active = {}, {}

    def config(root):
        c = old_config(root); assert c['maxiters'] == 30
        c.update(_seed_pose_embedding=active['latent'], _seed_global_orient=active['root'],
            _seed_root_enabled=active['neighbour'] is not None)
        return c

    def loss(fitting, neighbour_pose, dt, nominal_dt, receipt):
        if neighbour_pose is not None:
            neighbour_pose = neighbour_pose.copy()
            neighbour_pose[:, :3] = transport(neighbour_pose[:, :3],
                rows[active['neighbour']]['w2c'], rows[active['frame']]['w2c'])
            receipt.update(root_prior_transported_to_current_camera=True,
                neighbour_used_as_initialization=True, source_E_physically_validated=False)
        return old_loss(fitting, neighbour_pose, dt, nominal_dt, receipt)

    def fit(**kwargs):
        frame = Path(kwargs['result_path']).parent.name
        assert frame == p['ordered_fit_frames'][len(bank)]
        neighbour = min(bank, key=lambda n:(abs(rows[n]['time_seconds']-rows[frame]['time_seconds']), n)) if bank else None
        active.update(frame=frame, neighbour=neighbour, latent=np.zeros((1,32),np.float32), root=np.zeros((1,3),np.float32))
        if neighbour:
            assert abs(int(frame[-6:])-int(neighbour[-6:])) == 1
            active.update(latent=bank[neighbour]['pose_embedding'].copy(),
                root=transport(bank[neighbour]['global_orient'], rows[neighbour]['w2c'], rows[frame]['w2c']))
        before = kwargs['keypoints'].copy()
        guarded, inside = mask_outside_raster(before, rows[frame]['wh'])
        kwargs['keypoints'][...] = guarded
        arrays, receipt = old_fit(**kwargs)
        trace = {k:v.tolist() for k,v in entry._root_seed_trace.items()}
        assert np.array_equal(arrays['camera_translation'], np.asarray(trace['camera_translation']))
        if neighbour: assert np.array_equal(trace['body_fit_initial_root'], active['root'])
        bank[frame] = {k:arrays[k].copy() for k in ('pose_embedding','global_orient')}
        receipt['continuation'] = dict(neighbour=neighbour, same_run_only=True, initial_state=trace,
            initial_body_latent=active['latent'].tolist(), camera_initialization_unchanged=True,
            root_camera_transport_applied=neighbour is not None)
        receipt['raster_validity'] = dict(zeroed_indices=np.flatnonzero((before[0,:,2]>0)&(~inside[0])).tolist(),
            coordinates_unchanged=np.array_equal(before[...,:2], guarded[...,:2]), visibility_proven=False)
        if len(bank) == p['budget']['additional_review_after_frames']:
            np.savez_compressed(OUT/'review-pause-pose-027.npz', **arrays)
            save(OUT/'review-request-027.json', dict(updated=now(), state='awaiting_review',
                frames=list(bank), last_frame=frame, pid=os.getpid()))
            wait = time.monotonic()
            while not (OUT/'review-decision-027.json').exists():
                if time.monotonic()-wait > p['budget']['review_wait_seconds']: raise RuntimeError('27-frame review deadline.')
                time.sleep(2)
            if not read(OUT/'review-decision-027.json').get('continue_same_candidate', False):
                raise RuntimeError('Stopped by actual 27-frame pose review.')
        return arrays, receipt

    temporal.context = lambda subject:(SRC, OUT, read(SRC/'selection.json'))
    temporal.install_loss_factory = loss
    entry.adapted_entry_source, entry.load_official_config, entry.fit_one = seed.adapted_source, config, fit
    try:
        temporal.run('grass')
        for path, digest in p['sources'].items(): assert sha(path) == digest, path
        path = OUT/'official-fit/input-provenance.json'
        q = read(path); q['initialization'] = 'Official zero/mean anchor; same-run adjacent body/root, root transported between conditional camera rotations'
        save(path, q)
    except BaseException as exc:
        save(OUT/'failure.json', dict(updated=now(), error=repr(exc), computed_frames=len(bank)))
        path = OUT/'temporal-progress.json'
        q = read(path) if path.exists() else {}; q.update(state='stopped_or_failed', pid=None, error=repr(exc)); save(path,q)
        raise
    finally:
        temporal.context, temporal.install_loss_factory = old_context, old_loss
        entry.adapted_entry_source, entry.load_official_config, entry.fit_one = old_source, old_config, old_fit


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['prepare','run']); args=ap.parse_args()
    (prepare if args.mode == 'prepare' else run)()
