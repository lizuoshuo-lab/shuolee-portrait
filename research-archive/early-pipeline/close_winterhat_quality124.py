"""Record completed winterhat image review without changing the fit or policy."""
from datetime import datetime, timezone
from pathlib import Path
import audit_gaussianavatars_params123 as a
from select_quality_frames124 import select
from prepare_winterhat_quality124 import OUT, RUN

stamp = datetime.now(timezone.utc).isoformat()
assert not a.lp(OUT/'selection-reviewed.json').exists()
short = a.read(OUT/'shortlist.json')
pages = a.read(OUT/'review-pages.json')
assert a.sha(short['source_fit']) == short['source_fit_sha256']
assert len(pages) == 10 and sum(len(p['frames']) for p in pages) == 39
decisions = {}
uncertain = {'frame_000040', 'frame_000041', 'frame_000046'}
for p in pages:
    assert a.sha(p['path']) == p['sha256']
    for f in p['frames']:
        n = int(f[-6:])
        if f in uncertain:
            status = 'uncertain'
            reason = ('Native projection is closer to the source gesture, but side views still '
                      'separate the raised hand from the chest. Clothing/strap thickness, torso '
                      'heading and true hand depth are not established by this overview. Not admitted.')
        else:
            status = 'rejected'
            reason = ('Raised arm extends far outward although the source hand grips the strap at the chest.'
                      if n == 18 else
                      'Synthetic side views show the raised hand extending away from the torso; '
                      'the source shows a chest-side strap grip. Several frames also bend the '
                      'lowered arm backward. Frontal overlap alone does not validate this 3D pose.')
        decisions[f] = dict(status=status, reason=reason, evidence=p['path'],
                            evidence_sha256=p['sha256'], depth_reviewed=True)
review = dict(updated=stamp, source_fit=short['source_fit'],
    source_fit_sha256=short['source_fit_sha256'], records=decisions,
    actual_overview_pages=pages, actual_native_body_detail_frames=[40, 71],
    native_face_feet_acceptance=False, all112_physical_failure_claim=False,
    hidden_knees_and_boot_geometry='Not evaluated as observable ground truth through the skirt/boots.',
    synthetic_views='Inspection of the fitted geometry, not a measurement of the real person.',
    numerical_flags='Screening flags are not per-frame physical failure diagnoses.',
    review_only_frames=short['review_only_frames'])
a.save(OUT/'review-decisions.json', review)
result = select(short['records'], decisions, admitted_only=True)
assert not result['selected_frames'] and not result['training_admission']
result.update(updated=stamp, source_fit=short['source_fit'],
    source_fit_sha256=short['source_fit_sha256'], actual_pool_review_count=39,
    pool_rejected=36, pool_uncertain=3, automatic_screen_or_reserved_count=73,
    review_decisions_sha256=a.sha(OUT/'review-decisions.json'),
    selector_sha256=a.sha(Path(__file__).with_name('select_quality_frames124.py')),
    new_fits=0, solver_changed=False, new_gaussian_training=False,
    review_only_frames=short['review_only_frames'],
    conclusion='No accepted approximately-eight-frame set from this existing fit; do not train the shortlist.',
    next='Apply the same selected-only gate to the coherent cached sofa fit. Never require all frames to pass or restart this closed winterhat fit.')
a.save(OUT/'selection-reviewed.json', result)
print(dict(subject='winterhat', reviewed=39, rejected=36, uncertain=3, selected=0, new_fit_calls=0))
