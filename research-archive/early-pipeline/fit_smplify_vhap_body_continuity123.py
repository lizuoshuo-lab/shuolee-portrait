"""Keep only body initialization from the original VHAP next-frame copy.

Reuse the completed VPoser32 full-copy caller's stages and original solver.
After its unchanged original copy, immediately restore all non-body fields.
No repository edit, new loss, custom sequence optimizer, or Gaussian launch.
"""
from pathlib import Path
import time
import types
import torch
import fit_smplify_vhap_continuous123 as control


sequence = control.sequence
OUT = sequence.ROOT / 'smplify-vhap-body-continuity'
PLAN = sequence.ROOT / 'smplify-vhap-body-continuity-plan.json'
BODY = 'pose_embedding'
RESTORED = tuple(key for key in control.FIELDS if key != BODY)


def checked_body_copy(tracker, timesteps, phase):
    """Run the original full copy, then restore every non-body parameter row."""
    assert tracker.cfg.model.use_vposer and tracker.body_pose is None
    assert tracker.pose_embedding.shape == (59, 32)
    before = control.snapshot(tracker)
    shared_before = control.shared_hashes(tracker)
    step_before = tracker.global_step
    # This existing checker calls the original unmodified method and verifies
    # all nine fields, its exact strict boundary, source and shared parameters.
    original = control.checked_original_copy(tracker, timesteps, phase + ':original_full_copy')
    with torch.no_grad():
        for key in RESTORED:
            value = getattr(tracker, key)
            value.copy_(before[key].to(value))
    after = control.snapshot(tracker)
    source, target = original['source_slot'], original['target_slot']
    copied = original['copied']
    expected_body = before[BODY].clone()
    if copied:
        expected_body[target].copy_(before[BODY][source])
    assert torch.equal(after[BODY], expected_body), (phase, source, target, BODY)
    assert all(torch.equal(after[key], before[key]) for key in RESTORED)
    assert tracker.global_step == step_before
    assert control.shared_hashes(tracker) == shared_before
    fields = {}
    for key in control.FIELDS:
        item = dict(shape=list(before[key].shape), before_sha256=control.digest(before[key]),
                    after_sha256=control.digest(after[key]), source_unchanged=True,
                    all_rows_unchanged=bool(torch.equal(before[key], after[key])))
        if target < 59:
            item.update(target_before_values=before[key][target].tolist(),
                        target_after_values=after[key][target].tolist())
        fields[key] = item
    return dict(phase=phase, source_slot=source, source_frame=f'frame_{source+1:06d}',
        target_slot=target, target_frame=f'frame_{target+1:06d}' if target < 59 else None,
        copied=copied, copied_fields=[BODY] if copied else [], restored_fields=list(RESTORED),
        skipped_by_original_boundary=not copied, fields=fields,
        original_full_copy_verified=True, original_boundary_retained=True,
        shared_before_sha256=shared_before, shared_after_sha256=control.shared_hashes(tracker),
        shared_values_unchanged_by_copy=True, optimizer_updates_during_copy=0,
        global_step_before=step_before, global_step_after=tracker.global_step,
        body_only=True, root_face_hands_restored_bitwise=True)


def preflight(tracker, rows):
    assert len(rows) == tracker.n_timesteps == 59
    assert [row['frame'] for row in rows] == [f'frame_{i:06d}' for i in range(1, 60)]
    assert tracker.cfg.model.use_vposer and tracker.pose_embedding.shape == (59, 32)
    assert tracker.body_pose is None
    assert tracker.cfg.model.use_hand_pca
    assert tracker.left_hand_pose.shape == tracker.right_hand_pose.shape == (59, 12)
    assert not tracker.cfg.model.use_dynamic_offset and tracker.dynamic_offset is None
    assert tracker.calibrated and tracker.scaled and tracker.global_step == 0
    assert tracker.initialize_next_timtestep.__func__.__module__ == 'vhap.model.tracker_smplx'
    assert tracker.cfg.pipeline.rgb_init_offset.num_steps == 500
    assert tracker.cfg.pipeline.rgb_sequential_tracking.num_steps == 50
    assert tracker.cfg.pipeline.rgb_global_tracking.num_epochs == 30
    original = control.snapshot(tracker)
    shared_original = control.shared_hashes(tracker)
    records = []
    try:
        # Distinct temporary row markers ensure even originally equal adjacent
        # initial values test the copy. No forward or optimization uses them.
        with torch.no_grad():
            for field_index, key in enumerate(control.FIELDS):
                value = getattr(tracker, key)
                for slot in range(59):
                    value[slot].reshape(-1)[0] = .01 * (field_index + 1) + .0001 * slot
        probe = control.snapshot(tracker)
        for source in range(59):
            sample = tracker.dataset.one(source)
            assert sample['timestep_index'].tolist() == [source]
            intrinsic = control.digest(sample['intrinsic'])
            extrinsic = control.digest(sample['extrinsic'])
            row = checked_body_copy(tracker, sample['timestep_index'], 'zero_update_copy_sentinel')
            if row['copied']:
                assert not row['fields'][BODY]['all_rows_unchanged']
            else:
                assert row['fields'][BODY]['all_rows_unchanged']
            assert all(row['fields'][key]['all_rows_unchanged'] for key in RESTORED)
            assert control.digest(sample['intrinsic']) == intrinsic
            assert control.digest(sample['extrinsic']) == extrinsic
            row.update(intrinsic_sha256=intrinsic, extrinsic_sha256=extrinsic,
                       synthetic_copy_sentinel_only=True)
            records.append(row)
            with torch.no_grad():
                for key, values in probe.items():
                    value = getattr(tracker, key)
                    value.copy_(values.to(value))
            assert all(torch.equal(control.snapshot(tracker)[key], values) for key, values in probe.items())
            assert control.shared_hashes(tracker) == shared_original
            assert tracker.global_step == 0
    finally:
        with torch.no_grad():
            for key, values in original.items():
                value = getattr(tracker, key)
                value.copy_(values.to(value))
    restored = control.snapshot(tracker)
    assert all(torch.equal(restored[key], values) for key, values in original.items())
    assert control.shared_hashes(tracker) == shared_original
    assert len(records) == 59 and sum(row['copied'] for row in records) == 57
    assert [row['target_slot'] for row in records if row['copied']] == list(range(1, 58))
    receipt = dict(updated=sequence.now(), state='body_only_original_copy_preflight_passed',
        optimizer_updates=0, optimizer_created_by_preflight=False, source_slots_tested=list(range(59)),
        expected_copied_targets=list(range(1, 58)), expected_boundary_noops=[57, 58],
        all_nine_per_frame_parameters_restored_bitwise=True, all_shared_values_unchanged=True,
        temporary_markers_never_rendered_or_optimized=True, real_fit_initialization_restored=True,
        calls=records, original_initializer='vhap.model.tracker_smplx.GlobalTracker.initialize_next_timtestep',
        original_boundary_retained=True, body_only=True, copied_fields=[BODY], restored_fields=list(RESTORED),
        fixed_K_and_sample_extrinsics_unchanged=True, root_translation_and_orientation_are_copied=False,
        quality_accepted=False)
    sequence.save(OUT / 'continuous-copy-preflight.json', receipt)
    return receipt


def main():
    plan = sequence.read(PLAN)
    assert plan['execution_authorized'] is True and plan.get('user_approval')
    assert plan['source_sha256'] == sequence.sha(sequence.__file__)
    assert plan['wrapper_sha256'] == sequence.sha(__file__)
    assert plan['restore_original_offset_initialization']
    assert plan['initialization_frame'] == 'frame_000033'
    assert plan['body_representation'] == 'VPoser32' and plan['hand_pca_dimensions'] == 12
    assert sequence.sha(plan['comparison_plan_path']) == plan['comparison_plan_sha256']
    matched = sequence.read(plan['comparison_plan_path'])
    assert plan['roster'] == matched['roster']
    for group in ('source_hashes', 'input_hashes'):
        assert all(plan[group].get(path) == digest for path, digest in matched[group].items())
    for path, digest in plan['comparison_inputs'].items():
        assert sequence.sha(path) == digest, path
    original_make, original_initialize = sequence.make_tracker, sequence.bridge.initialize
    previous_out, previous_plan = sequence.OUT, sequence.PLAN
    copies = []

    def make_tracker(out):
        tracker, rows = original_make(out)
        original_stage = tracker.optimize_stage

        def optimize_stage(self, stage, *args, **kwargs):
            sample = None
            if stage == 'rgb_sequential_tracking':
                sample = kwargs.get('sample', args[0] if args else None)
                assert sample is not None and len(copies) < 59
                assert sample['timestep_index'].tolist() == [len(copies)]
                assert kwargs.get('lr_scale', 1.) == 1.
            result = original_stage(stage, *args, **kwargs)
            if sample is not None:
                copies.append(checked_body_copy(self, sample['timestep_index'], 'after_original_sequential_stage'))
                sequence.save(OUT / 'continuous-copy-records.json', dict(updated=sequence.now(),
                    copy_calls=len(copies), copied_pairs=sum(row['copied'] for row in copies), calls=copies,
                    body_only=True, copied_fields=[BODY], restored_fields=list(RESTORED),
                    original_boundary_retained=True, quality_accepted=False,
                    note='Only initializer copying is restricted; original optimizer still fits root, face and hands in each stage.'))
            return result

        tracker.optimize_stage = types.MethodType(optimize_stage, tracker)
        return tracker, rows

    def initialize(tracker, rows):
        receipts = original_initialize(tracker, rows)
        preflight(tracker, rows)
        return receipts

    began = time.monotonic()
    sequence.OUT, sequence.PLAN = OUT, PLAN
    sequence.make_tracker, sequence.bridge.initialize = make_tracker, initialize
    try:
        sequence.main()
        assert len(copies) == 59 and sum(row['copied'] for row in copies) == 57
        assert [row['source_slot'] for row in copies] == list(range(59))
        result = sequence.read(OUT / 'result.json')
        assert result['steps'] == 5220
        assert result['offset_initialization_updates'] == 500
        assert result['sequential_refinement_updates'] == 2950
        assert result['global_refinement_updates'] == 1770
        metadata = dict(previous_frame_initialization_copy=True, body_only=True,
            body_representation='VPoser32', hand_pca_dimensions=12,
            copy_calls=59, copied_pairs=57, copied_fields=[BODY], restored_fields=list(RESTORED),
            original_boundary_retained=True, original_last_frame_copy_omission_retained=True,
            comparison_control='smplify-vhap-continuous', control_checkpoint_used_as_initialization=False,
            fixed_K_and_sample_extrinsics_unchanged=True,
            per_frame_root_face_and_hand_initial_values_can_be_overwritten=False,
            root_face_and_hand_fitting_remains_active=True,
            wrapper_sha256=sequence.sha(__file__), plan_sha256=sequence.sha(PLAN),
            continuous_copy_preflight_sha256=sequence.sha(OUT / 'continuous-copy-preflight.json'),
            continuous_copy_records_sha256=sequence.sha(OUT / 'continuous-copy-records.json'))
        result.update(metadata, wrapper_elapsed_seconds=time.monotonic() - began,
                      quality_accepted=False, gaussian_training_started=False)
        sequence.save(OUT / 'result.json', result)
        sequence.save(OUT / 'progress.json', dict(**result, pid=None))
        stage = sequence.read(OUT / 'sequential-stage.json')
        stage.update(metadata)
        sequence.save(OUT / 'sequential-stage.json', stage)
        print(sequence.json.dumps(result), flush=True)
    finally:
        sequence.make_tracker, sequence.bridge.initialize = original_make, original_initialize
        sequence.OUT, sequence.PLAN = previous_out, previous_plan


if __name__ == '__main__':
    main()
