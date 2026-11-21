"""Restore only the repository's previous-frame initialization after each fit.

Reuse the completed VPoser32 sequential/global caller, including its original
57-copy boundary. Do not edit the repository or create a new sequence solver.
"""
from pathlib import Path
import hashlib
import time
import types
import torch
import fit_smplify_vhap_sequential_stage123 as sequence


OUT = sequence.ROOT / 'smplify-vhap-continuous'
PLAN = sequence.ROOT / 'smplify-vhap-continuous-plan.json'
FIELDS = ('transl', 'global_orient', 'jaw_pose', 'leye_pose', 'reye_pose',
          'expression', 'pose_embedding', 'left_hand_pose', 'right_hand_pose')
SHARED = ('betas', 'static_offset', 'lights', 'tex_extra')


def digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def snapshot(tracker):
    return {key: getattr(tracker, key).detach().cpu().clone() for key in FIELDS}


def shared_hashes(tracker):
    return {key: digest(getattr(tracker, key)) for key in SHARED}


def checked_original_copy(tracker, timesteps, phase):
    """Invoke the unchanged original function, then verify every parameter row."""
    assert timesteps.ndim == 1 and timesteps.numel() == 1
    source = int(timesteps[0].item())
    assert 0 <= source < 59 and tracker.n_timesteps == 59
    before = snapshot(tracker)
    shared_before = shared_hashes(tracker)
    step_before = tracker.global_step
    tracker.initialize_next_timtestep(timesteps)
    after = snapshot(tracker)
    shared_after = shared_hashes(tracker)
    assert tracker.global_step == step_before
    assert shared_before == shared_after, 'The copy function changed a shared value.'
    target = source + 1
    copied = target < tracker.n_timesteps - 1  # Retain original strict boundary.
    fields = {}
    for key in FIELDS:
        expected = before[key].clone()
        if copied:
            expected[target].copy_(before[key][source])
        assert torch.equal(after[key], expected), (phase, source, target, key)
        assert torch.equal(after[key][source], before[key][source])
        entry = dict(shape=list(before[key].shape), before_sha256=digest(before[key]),
                     after_sha256=digest(after[key]), source_values=before[key][source].tolist(),
                     source_unchanged=True, only_expected_target_may_change=True)
        if target < 59:
            entry.update(target_before_values=before[key][target].tolist(),
                         target_after_values=after[key][target].tolist(),
                         target_values_changed=not torch.equal(before[key][target], after[key][target]))
        fields[key] = entry
    return dict(phase=phase, source_slot=source, source_frame=f'frame_{source+1:06d}',
                target_slot=target, target_frame=f'frame_{target+1:06d}' if target < 59 else None,
                copied=copied, copied_fields=list(FIELDS) if copied else [],
                skipped_by_original_boundary=not copied, fields=fields,
                shared_before_sha256=shared_before, shared_after_sha256=shared_after,
                shared_values_unchanged_by_copy=True, optimizer_updates_during_copy=0,
                global_step_before=step_before, global_step_after=tracker.global_step)


def preflight(tracker, rows):
    assert len(rows) == tracker.n_timesteps == 59
    assert [row['frame'] for row in rows] == [f'frame_{i:06d}' for i in range(1, 60)]
    assert tracker.cfg.model.use_vposer and tracker.pose_embedding.shape == (59, 32)
    assert tracker.body_pose is None
    assert tracker.cfg.model.use_hand_pca and tracker.left_hand_pose.shape == tracker.right_hand_pose.shape == (59, 12)
    assert not tracker.cfg.model.use_dynamic_offset and tracker.dynamic_offset is None
    assert tracker.calibrated and tracker.scaled and tracker.global_step == 0
    assert tracker.initialize_next_timtestep.__func__.__module__ == 'vhap.model.tracker_smplx'
    assert tracker.cfg.pipeline.rgb_init_offset.num_steps == 500
    assert tracker.cfg.pipeline.rgb_sequential_tracking.num_steps == 50
    assert tracker.cfg.pipeline.rgb_global_tracking.num_epochs == 30
    original = snapshot(tracker)
    shared_original = shared_hashes(tracker)
    samples = []
    records = []
    for source in (0, 56, 57, 58):
        sample = tracker.dataset.one(source)
        assert sample['timestep_index'].tolist() == [source]
        samples.append(dict(slot=source, frame=rows[source]['frame'],
                            intrinsic_sha256=digest(sample['intrinsic']),
                            extrinsic_sha256=digest(sample['extrinsic'])))
        records.append(checked_original_copy(tracker, sample['timestep_index'], 'zero_update_preflight'))
        assert digest(sample['intrinsic']) == samples[-1]['intrinsic_sha256']
        assert digest(sample['extrinsic']) == samples[-1]['extrinsic_sha256']
        with torch.no_grad():
            for key, values in original.items():
                getattr(tracker, key).copy_(values.to(getattr(tracker, key)))
        restored = snapshot(tracker)
        assert all(torch.equal(restored[key], values) for key, values in original.items())
        assert shared_hashes(tracker) == shared_original
        assert tracker.global_step == 0
    receipt = dict(updated=sequence.now(), state='original_copy_preflight_passed',
                   optimizer_updates=0, optimizer_created_by_preflight=False,
                   source_slots_tested=[0, 56, 57, 58],
                   expected_copied_targets=[1, 57], expected_boundary_noops=[57, 58],
                   all_nine_per_frame_parameters_restored_bitwise=True,
                   all_shared_values_unchanged=True, samples=samples, calls=records,
                   original_initializer='vhap.model.tracker_smplx.GlobalTracker.initialize_next_timtestep',
                   original_boundary_retained=True, body_only=False,
                   fixed_K_and_sample_extrinsics_unchanged=True,
                   root_translation_and_orientation_are_copied=True,
                   quality_accepted=False)
    sequence.save(OUT / 'continuous-copy-preflight.json', receipt)
    return receipt


def main():
    plan = sequence.read(PLAN)
    assert plan['source_sha256'] == sequence.sha(sequence.__file__)
    assert plan['restore_original_offset_initialization']
    assert plan['initialization_frame'] == 'frame_000033'
    wrapper = Path(__file__).resolve()
    assert any(Path(path).resolve() == wrapper and value == sequence.sha(__file__)
               for path, value in plan['input_hashes'].items()), 'Bind wrapper SHA separately in input_hashes.'
    original_make = sequence.make_tracker
    original_initialize = sequence.bridge.initialize
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
                copies.append(checked_original_copy(self, sample['timestep_index'], 'after_original_sequential_stage'))
                sequence.save(OUT / 'continuous-copy-records.json', dict(updated=sequence.now(),
                    copy_calls=len(copies), copied_pairs=sum(x['copied'] for x in copies),
                    calls=copies, body_only=False, original_boundary_retained=True,
                    note='Shared texture can change during the preceding originalRGB sequential fit; only copy-before/copy-after equality is asserted.',
                    quality_accepted=False))
            return result

        tracker.optimize_stage = types.MethodType(optimize_stage, tracker)
        return tracker, rows

    def initialize(tracker, rows):
        receipts = original_initialize(tracker, rows)
        preflight(tracker, rows)
        return receipts

    began = time.monotonic()
    sequence.OUT, sequence.PLAN = OUT, PLAN
    sequence.make_tracker = make_tracker
    sequence.bridge.initialize = initialize
    try:
        sequence.main()
        assert len(copies) == 59 and sum(x['copied'] for x in copies) == 57
        assert [x['source_slot'] for x in copies] == list(range(59))
        assert [x['target_slot'] for x in copies if x['copied']] == list(range(1, 58))
        result = sequence.read(OUT / 'result.json')
        assert result['steps'] == 500 + 2950 + 1770 == 5220
        assert result['offset_initialization_updates'] == 500
        assert result['sequential_refinement_updates'] == 2950
        assert result['global_refinement_updates'] == 1770
        metadata = dict(previous_frame_initialization_copy=True, body_only=False,
                        copy_calls=59, copied_pairs=57, original_boundary_retained=True,
                        original_last_frame_copy_omission_retained=True,
                        copied_fields=list(FIELDS), comparison_control='smplify-vhap-sequential-stage',
                        fixed_K_and_sample_extrinsics_unchanged=True,
                        per_frame_root_face_and_hand_initial_values_can_be_overwritten=True,
                        wrapper_sha256=sequence.sha(__file__),
                        continuous_copy_preflight_sha256=sequence.sha(OUT / 'continuous-copy-preflight.json'),
                        continuous_copy_records_sha256=sequence.sha(OUT / 'continuous-copy-records.json'))
        result.update(metadata, wrapper_elapsed_seconds=time.monotonic() - began,
                      quality_accepted=False, gaussian_training_started=False)
        sequence.save(OUT / 'result.json', result)
        sequence.save(OUT / 'progress.json', dict(**result, pid=None))
        stage_receipt = sequence.read(OUT / 'sequential-stage.json')
        stage_receipt.update(metadata)
        sequence.save(OUT / 'sequential-stage.json', stage_receipt)
        print(sequence.json.dumps(result), flush=True)
    finally:
        sequence.make_tracker = original_make
        sequence.bridge.initialize = original_initialize
        sequence.OUT, sequence.PLAN = previous_out, previous_plan


if __name__ == '__main__':
    main()
