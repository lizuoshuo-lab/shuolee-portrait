"""Resume the uncomputed frames of a fixed-shape CPU fit after a depth rejection."""
import argparse
import os
from pathlib import Path
from shutil import copyfile

import numpy as np

import fit_video21_native148 as adapter
from pose_training_gate151 import require, select_fit
from update_leafwoman129_state import atomic, read, sha


def recover(run):
    out = run/'recovery-depth'
    stage = run/'shared-cpu'
    source = out/'source'
    runner_path = source/'run_fixed_smplify175.py'
    worker_path = source/'run_leafwoman_front24_cpu145.py'
    plan = read(run/'plan.json')
    worker_plan = read(stage/'plan.json')
    audit_path = run/'failure-depth-audit.json'
    audit = read(audit_path)
    shape_path = run/'shared-shape.json'
    shape = read(shape_path)
    archive = run/'fit-observations.npz'
    contacts = run/'contacts.json'
    require(sha(run/'plan.json') == audit['source_plan_sha256']
            and sha(shape_path) == audit['shared_shape_sha256']
            and sha(runner_path) == plan['runner_sha256']
            and sha(worker_path) == plan['worker_sha256']
            and sha(adapter.entry.__file__) == plan['fit_entry_sha256']
            and sha(archive) == plan['baseline_observation_sha256']
            and sha(contacts) == plan['contacts_sha256'],
            'Recovery requires the exact original plan, shape, source and worker.')
    require(all(worker_plan[key] == value for key, value in plan.items())
            and worker_plan['shared_shape']['vector'] == shape['betas'],
            'Worker plan or frozen shape differs.')
    names = plan['source_frames']
    missing = audit['uncomputed_frames']
    invalid = {item['frame']: item for item in audit['confirmed_invalid']}
    receipts = {f: read(stage/f/'receipt.json') for f in names
                if (stage/f/'receipt.json').is_file()}
    require(audit['worker_exit_code'] == 1 and len(receipts) == audit['complete_poses']
            and len(set(names)) == len(names) == len(receipts) + len(invalid) + len(missing)
            and set(names) == receipts.keys() | invalid.keys() | set(missing)
            and not set(names) & set(plan['retained_holdouts']),
            'Existing receipts, rejected frames and uncomputed frames must partition the frozen roster.')
    for f, row in invalid.items():
        require(row['min_depth_model_units'] <= .05 and sha(stage/f/'official-result.pkl') == row['pkl_sha256'],
                f'Negative-depth evidence changed for {f}.')
    with np.load(archive, allow_pickle=False) as data:
        observed = data['observed_native_keypoints']
        K = data['K_integer_centers']
        q90 = data['confidence_q90']
        require(data['frame_ids'].tolist() == names and np.array_equal(K, plan['K_integer_centers']),
                'Observation identities or K differ.')
    beta = np.asarray(shape['betas'], dtype=np.float32)

    def verify_receipt(f, r):
        i = names.index(f)
        require(r['frame'] == f and r['source_index'] == plan['native_indices'][i]
                and r['freeze_betas'] is True and r['native_crop_bitexact'] is True
                and r['source_hashes'] == adapter.entry.PINNED and sha(r['output']) == r['sha256'],
                f'Saved fit identity changed for {f}.')
        with np.load(r['output'], allow_pickle=False) as fit:
            vertices = fit['vertices_camera']
            require(np.isfinite(vertices).all() and (vertices[..., 2] > .05).all()
                    and np.array_equal(fit['betas'], beta) and np.array_equal(fit['K_integer_centers'], K)
                    and np.array_equal(fit['observed_native_keypoints'], observed[i:i+1]),
                    f'Saved fit is not the bound positive-depth shared-shape result for {f}.')

    for f, row in receipts.items():
        verify_receipt(f, row)
    require(not (out/'plan.json').exists(), 'Recovery result already prepared.')
    print('RECOVERY_EXACT_INPUTS', len(receipts), len(invalid), missing, flush=True)
    runner = adapter.entry.module(runner_path, '_resume_exact_fixed_runner179')
    adapter.original = adapter.entry.module(worker_path, '_resume_exact_fixed_worker179')
    fit_one = adapter.entry.fit_one
    for f in missing:
        compute = out/'compute'/f
        compute.mkdir(parents=True, exist_ok=False)
        for p in (archive, contacts):
            copyfile(p, compute/p.name)
        substage = compute/'shared-cpu'
        atomic(substage/'plan.json', dict(worker_plan, shards=[[names.index(f)]]))
        depth_path = compute/'depth.json'

        def capture_depth(**kwargs):
            arrays, receipt = fit_one(**kwargs)
            depth = arrays['vertices_camera'][..., 2]
            finite = bool(np.isfinite(depth).all())
            atomic(depth_path, dict(frame=f, source_index=plan['native_indices'][names.index(f)],
                finite=finite, min_depth_model_units=float(depth.min()) if finite else None,
                positive_depth_count=int(np.count_nonzero(depth > .05)),
                below_or_at_depth_threshold_count=int(np.count_nonzero(depth <= .05)),
                same_K=bool(np.array_equal(arrays['K_integer_centers'], K)),
                same_beta=bool(np.array_equal(arrays['betas'], beta)),
                pkl_sha256=sha(kwargs['result_path'])))
            return arrays, receipt

        adapter.entry.fit_one = capture_depth
        try:
            print('RECOVERY_ORIGINAL_WORKER', f, 'pid', os.getpid(), flush=True)
            runner.fit_shard(substage, 0)
        except AssertionError:
            evidence = read(depth_path)
            if not (evidence['finite'] and evidence['same_K'] and evidence['same_beta']
                    and evidence['min_depth_model_units'] <= .05):
                raise
            invalid[f] = evidence
            print('RECOVERY_DEPTH_REJECTED', f, evidence['min_depth_model_units'], flush=True)
        else:
            receipts[f] = read(substage/f/'receipt.json')
            verify_receipt(f, receipts[f])
            print('RECOVERY_DEPTH_VALID', f, flush=True)
        finally:
            adapter.entry.fit_one = fit_one

    keep = [i for i, f in enumerate(names) if f in receipts]
    ids = [names[i] for i in keep]
    require(len(ids) + len(invalid) == len(names), 'Incomplete recovery disposition.')
    contact_rows = read(contacts)['records']
    require([r['frame'] for r in contact_rows] == names, 'Contact roster differs.')
    crops = {r['frame']: r for r in read(run/'observation-review/inputs-review.json')['rows']}
    atomic(out/'contacts.json', dict(records=[contact_rows[i] for i in keep]))
    atomic(out/'observation-review/inputs-review.json', dict(rows=[crops[f] for f in ids]))
    adapter.save_npz(out/'fit-observations.npz', frame_ids=np.asarray(ids),
                     observed_native_keypoints=observed[keep], K_integer_centers=K, confidence_q90=q90)
    recovered = dict(plan, source_frames=ids, native_indices=[plan['native_indices'][i] for i in keep],
        contacts_sha256=sha(out/'contacts.json'), baseline_observation_sha256=sha(out/'fit-observations.npz'),
        original_source_plan_sha256=sha(run/'plan.json'), depth_audit_sha256=sha(audit_path),
        recovery_runner_sha256=sha(__file__), depth_rejected_frames=sorted(invalid))
    for key in ('raster_unsupported_frames', 'face_absence_frames'):
        recovered[key] = [f for f in plan[key] if f in receipts]
    recovered['ankle_foot_absence'] = {f: v for f, v in plan['ankle_foot_absence'].items() if f in receipts}
    atomic(out/'plan.json', recovered)
    atomic(out/'result.json', dict(plan_sha256=sha(out/'plan.json'), frames=[receipts[f] for f in ids],
        fixed_shape=shape['betas'], model_unit_scale=1., anchor_count=len(plan['shape_anchors']),
        reused_anchor_count=len(shape['reused_frames']), shared_shape_sha256=sha(shape_path),
        fitting_complete=True, original_full_roster_successful=False, prior_saved_fits_reused=audit['complete_poses'],
        original_candidates=len(names), recovery_computed_frames=missing, invalid_depth=invalid,
        gaussian_started=False, quality_accepted=False))
    print('RECOVERY_COMPLETE', len(ids), len(invalid), flush=True)


def review(run):
    import run_fixed_smplify175 as runner
    import review_video21_fit148 as visual

    out = run/'recovery-depth'
    runner.configured(visual.review, OUT=out, NATIVE=out, CONTACTS=out/'contacts.json')(False)
    atomic(out/'review/pose-training-selection.json', select_fit(out, 'full'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('recover', 'review'))
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    (recover if args.phase == 'recover' else review)(args.run.resolve())
