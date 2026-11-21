"""Finalize only a measured, UI-reviewed winterhat eight-frame delivery.

No training, export, package or GUI work is performed here. This retains the
original1000 timeout and requires the separately audited non-exact recovery.
"""
import inspect
from datetime import datetime, timezone

import audit_gaussianavatars_params123 as a
import finalize_indoor_clock_delivery124 as existing
from publish_winterhat_half8_124 import main as report
from update_indoor_clock_state124 import atomic

FRAMES = [16, 28, 41, 56, 57, 71, 96, 109]
REFERENCE = 'frame_000071'


def main():
    root = a.REC/'cross-subject-sh0-124/winterhat-half8-paired18k124'
    out = root/'delivery/step-018000'
    progress = a.read(root/'progress.json')
    final = a.read(root/'gaussian-run/result.json')
    snapshot = a.read(root/'gaussian-run/step-018000/result.json')
    plan = a.read(root/'plan.json')
    recovery = a.read(root/'recovery124-process-exit.json')
    launch = a.read(root/'recovery124-launch.json')
    validation = a.read(out/'validation.json')
    assert progress['state'] == final['state'] == 'bounded_control_completed'
    assert progress['step'] == final['snapshot_step'] == snapshot['step'] == 18000
    assert final['optimizer_updates'] == 17998
    assert recovery['exit_code'] == 0 and recovery['exact_continuation'] is False
    assert recovery['preserved_originals_verified'] is True
    assert launch['first_iteration'] == 1001
    assert launch['discarded_gradient_update_step'] == 1000
    assert launch['final_shared_offset_optimizer_updates'] == 17998
    assert a.read(root/'process-exit.json')['exit_code'] == 1
    assert 'Review timeout; checkpoint retained' in a.read(root/'train-failure.json')['error']
    assert plan['training_frames'] == 8 and plan['reference_frame'] == REFERENCE
    assert sorted(plan['selected_frames']) == [f'frame_{n:06d}' for n in FRAMES]
    assert snapshot['reference_frame'] == validation['reference_frame'] == REFERENCE
    assert snapshot['frames_rendered'] == 8 and snapshot['active_sh_degree'] == 0
    assert validation['format_passed'] and not validation.get('diagnostic_preview_only', False)
    for key in ('native_source_ply_parity', 'native_rotated_render_parity'):
        parity = validation[key]
        assert parity['passed'] and parity['mean_abs'] < 2e-6 and parity['max_abs'] < .005
    assert a.read(out/'sh0-field-check.json')['passed']
    cameras = a.read(out/'cameras.json')
    assert len(cameras) == 1 and cameras[0]['img_name'] == REFERENCE

    # Reuse the verified receipt mechanism, changing identities and the exit
    # receipt only. It still requires actual saved Gaussian/points/UI proofs.
    source = inspect.getsource(existing.main)
    for old, new, count in (
        ('indoor-zoom-clock18k124', 'winterhat-half8-paired18k124', 3),
        ('training_frames=6', 'training_frames=8', 1),
        ("'frame_000105'", repr(REFERENCE), 1),
        ("a.read(root/'process-exit.json')['exit_code']==0",
         "a.read(root/'recovery124-process-exit.json')['exit_code']==0", 1),
        ('INDOOR_CLOCK_DELIVERY_LOCAL_UI_VERIFIED_QUALITY_OPEN',
         'WINTERHAT_HALF8_DELIVERY_LOCAL_UI_VERIFIED_QUALITY_OPEN', 1),
    ):
        assert source.count(old) == count, old
        source = source.replace(old, new)
    namespace = dict(existing.main.__globals__, report=report)
    exec(compile(source, __file__ + ':standard-delivery-checks', 'exec'), namespace)
    namespace['main']()

    publication = a.read(root/'delivery/publication.json')
    now = datetime.now(timezone.utc).isoformat()
    registry_path = a.T/'shuolee/gaussian-deliveries.json'
    registry = a.read(registry_path)
    registry['rounds'][publication['modelId']].update(
        optimizer_updates=17998, same_experiment_checkpoint_loaded=True,
        exact_continuation=False, discarded_gradient_update_step=1000,
        original_timeout_retained=True, recovery_receipt=str(root/'recovery124-process-exit.json'))
    atomic(registry_path, registry)
    path = a.REC/'cross-subject-sh0-124/four-subject-delivery-status.json'
    status = a.read(path)
    status.setdefault('subsequent_deliveries', {})[publication['modelId']] = dict(
        updated=now, stage='winterhat_half8_delivered_quality_open', publication=publication,
        training_frames=8, reference_frame=REFERENCE, selected_frames=FRAMES,
        capture_arc='Eight reviewed frames; no independently measured half-circle or rear coverage.',
        original_train_exit_code=1, recovery_train_exit_code=0,
        optimizer_updates=17998, exact_continuation=False,
        recovery_receipt=str(root/'recovery124-process-exit.json'), points=snapshot['points'],
        native_strict_pixel_equivalence=True, local_UI_verified=True,
        company00_deployed=False, quality_accepted=False,
        original_source_pixel_review=str(root/'gaussian-run/step-018000/visual-review.json'),
        unknown_view_review=str(out/'visual-review.json'),
        remaining='Source wrist-depth errors, partial mask holes, unobserved rear and cross-person frozen-profile quality remain unresolved. Cold video-to-export15min remains unaccepted.')
    status.update(updated=now,
        next_action='Winterhat8 and sofa8 candidates delivered with quality open. Reconcile four-subject evidence and legal grass training roster without moving heldouts or rerunning completed candidates.',
        full_frozen_pipeline_generalization_accepted=False, end_to_end_15min_accepted=False)
    atomic(path, status)


if __name__ == '__main__':
    main()
