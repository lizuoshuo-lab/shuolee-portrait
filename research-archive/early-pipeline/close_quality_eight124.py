"""Record actual image review; never translate a numerical rank to quality."""
from datetime import datetime, timezone
from pathlib import Path
import audit_gaussianavatars_params123 as a
from select_quality_frames124 import select, self_test
from prepare_quality_eight124 import OUT,RUN

stamp=datetime.now(timezone.utc).isoformat()
short=a.read(OUT/'grass-shortlist.json'); pages=a.read(OUT/'grass-pool-review-pages.json')
initial=a.read(OUT/'grass-review-decisions.json')
decisions=dict(initial['records'])
uncertain={f'frame_{i:06d}' for i in [9,12,13,14,16,17,18,19,20]}
for p in pages:
    assert a.sha(p['path'])==p['sha256']
    for f in p['frames']:
        n=int(f[-6:])
        if f in uncertain:
            status='uncertain'
            reason='Hand is closer to torso, but crosses toward the opposite chest/shoulder in the synthetic front view; torso heading and actual chest-side/contact are not established. Overview is insufficient for admission.'
        else:
            status='rejected'
            reason=('Both arms are raised/spread although the source has one lowered arm; far-hand identity error.' if n in [75,76] else
                    'Raised hand is visibly separated in front of the chest in both synthetic side views; source grip is at the chest/hair. A good frontal overlap does not pass depth.')
        decisions[f]=dict(status=status,reason=reason,evidence=p['path'],evidence_sha256=p['sha256'],depth_reviewed=True)
assert len(pages)==10 and sum(len(p['frames']) for p in pages)==40
a.save(OUT/'grass-review-decisions-reviewed.json',dict(updated=stamp,records=decisions,
    source_fit_sha256=short['source_fit_sha256'], actual_overview_pages=pages,
    actual_individual_boards=[21,32,41,49,75],all94_native_review=False,
    review_scope='All40 numerical survivors reviewed in raw/overlay/two-side overview. No native face/feet approval claimed. Numerical failures retained without automatic per-frame physical diagnosis.'))
result=select(short['records'],decisions,admitted_only=True)
assert result['selected_frames']==[] and not result['training_admission']
result.update(updated=stamp,source_fit=short['source_fit'],source_fit_sha256=short['source_fit_sha256'],
    review_decisions_sha256=a.sha(OUT/'grass-review-decisions-reviewed.json'),
    selector_sha256=a.sha(Path(__file__).with_name('select_quality_frames124.py')),
    actual_pool_review_count=40,pool_rejected=31,pool_uncertain=9,
    user_rejected_additional=4,automatic_screen_flagged=50,
    physical_all94_failure_claim=False,solver_changed=False,new_fits=0,
    no_new_gaussian_training=True,controls=self_test())
a.save(OUT/'grass-selection-reviewed.json',result)
original=RUN.parent.parent/'smplifyx-reference-control/official/smplifyx/fitting.py'
reference=a.read(RUN/'official-fit/frames/frame_000032/result.json')
assert a.sha(original)==reference['source_hashes']['smplifyx/fitting.py']
a.save(OUT/'cause-review.json',dict(updated=stamp,
    fitting_source=str(original),fitting_sha256=a.sha(original),matches_actual_fit_source=True,
    evidence=['Raw SMPLify-X meshes already show the wrong arm depth before Gaussian training/export.',
              'Original loss compares projected 2D joints with image observations, plus VPoser/body, bending, shape, hand and face priors.',
              'The added temporal term compares neighbouring root/body rotations; it contains no observed wrist-to-chest depth/contact term.',
              'Reference32 already exhibits the depth failure. Prior pose parity does not prove correctness.',
              'A body prior permits many plausible poses; plausible anatomy does not establish this exact recorded gesture.'],
    causal_uniqueness=False,limits='Missing depth evidence explains why this wrong pose can pass these objectives; K, observations, initialization and optimizer contribution are not uniquely isolated.',
    old_initialization_reused=False,neighbour_used_as_temporal_target=True,
    wrong_solver_restart=False,all_sequence_fit_required=False,
    next='Use the reproducible selected-frame gate for other existing subject fits; for grass investigate supported torso/wrist depth from existing side views before any new minimal fitting change. Do not simply increase temporal weight or train GS on rejected poses.'))
a.save(OUT/'render-recovery.json',dict(initial_image_render_failure='Windows extended-path prefix was added twice by the image path wrapper.',
    scope='Only evidence rendering; selection was already saved and never recomputed.',
    repaired=True,fit_data_changed=False,policy_changed=False,new_fit_calls=0))
print('GRASS_REVIEWED_SELECTION',dict(selected=0,reviewed=40,rejected=31,uncertain=9,controls=11))
