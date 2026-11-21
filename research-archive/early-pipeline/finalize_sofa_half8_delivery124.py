"""Reuse standard completed-run delivery checks for the reviewed sofa eight."""
import inspect
from datetime import datetime, timezone
import audit_gaussianavatars_params123 as a
import finalize_indoor_clock_delivery124 as existing
from publish_sofa_half8_124 import main as report
from update_indoor_clock_state124 import atomic


def main():
    source = inspect.getsource(existing.main)
    for old, new, count in (
        ('indoor-zoom-clock18k124', 'sofa-half8-paired18k124', 3),
        ('training_frames=6', 'training_frames=8', 1),
        ("'frame_000105'", "'frame_000026'", 1),
        ('INDOOR_CLOCK_DELIVERY_LOCAL_UI_VERIFIED_QUALITY_OPEN',
         'SOFA_HALF8_DELIVERY_LOCAL_UI_VERIFIED_QUALITY_OPEN', 1),
    ):
        assert source.count(old) == count, old
        source = source.replace(old, new)
    namespace = dict(existing.main.__globals__, report=report)
    exec(compile(source, __file__ + ':standard-delivery-checks', 'exec'), namespace)
    namespace['main']()
    root = a.REC/'cross-subject-sh0-124/sofa-half8-paired18k124'
    final = a.read(root/'gaussian-run/step-018000/result.json')
    validation = a.read(root/'delivery/step-018000/validation.json')
    assert validation['native_rotated_render_parity']['passed']
    assert not validation.get('diagnostic_preview_only', False)
    path = a.REC/'cross-subject-sh0-124/four-subject-delivery-status.json'
    status = a.read(path)
    publication = a.read(root/'delivery/publication.json')
    status.setdefault('subsequent_deliveries', {})[publication['modelId']] = dict(
        updated=datetime.now(timezone.utc).isoformat(),
        stage='sofa_half8_delivered_quality_open', publication=publication,
        training_frames=8, reference_frame='frame_000026',
        selected_frames=[7, 13, 15, 21, 26, 32, 36, 46],
        angular_coverage='About99.5degrees from fitted direction proxy; no measured half-circle or rear coverage.',
        train_exit_code=0, points=final['points'],
        native_strict_pixel_equivalence=True, local_UI_verified=True,
        company00_deployed=False, quality_accepted=False,
        original_source_pixel_review=str(root/'gaussian-run/step-018000/visual-review.json'),
        unknown_view_review=str(root/'delivery/step-018000/visual-review.json'),
        remaining='Unobserved rear, shoe mask gaps, clothing fragmentation and cross-person frozen-profile quality. Cold video-to-export15min remains unaccepted.')
    status.update(updated=datetime.now(timezone.utc).isoformat(),
        next_action='Sofa8andindoor16 profiles delivered. Next inspect grass source-quality roster and capture arc for the same bounded SH0/ROI2x/paired-background/18k profile; preserve heldouts. Do not rerun completed sofa candidate.',
        full_frozen_pipeline_generalization_accepted=False, end_to_end_15min_accepted=False)
    atomic(path, status)


if __name__ == '__main__':
    main()
