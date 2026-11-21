"""Record actual read-only review of the preselected orientation-branch transition."""
from datetime import datetime, timezone
import audit_gaussianavatars_params123 as a

out = a.REC / 'cross-subject-sh0-124/grass-temporal-refinement/orientation-preview'
result = a.read(out / 'result.json')
assert [r['frame'] for r in result['records']] == ['frame_000076', 'frame_000077', 'frame_000078']
assert not a.lp(out / 'visual-review.json').exists()
for row in result['records']:
    for im in row['images'].values():
        path = im['path']
        if path.startswith('/mnt/'):
            path = path[5].upper() + ':/' + path[7:]
        assert a.sha(path) == im['sha256']
a.save(out / 'visual-review.json', dict(
    updated=datetime.now(timezone.utc).isoformat(), actual_visual_review=True,
    reviewed_frames=[r['frame'] for r in result['records']],
    reviewed_body_and_common_side_page=result['pages'][0],
    reviewed_native_face_feet=True, enlarged_body_and_four_sides=['frame_000076'],
    findings=[
        'No new discrete flip at the original one-to-two orientation-branch transition76to78.',
        'Maximum local joint transitions are5.93 and2.94 degrees, versus independent16.07 and20.17; this is continuity, not3D truth.',
        'Native face and foot projections remain close to the independent baseline, retaining existing toe/ankle mismatch.',
        'Both baseline and current model show a raised far arm in common views. Its native side-view visibility is poor; retain this as unresolved pose/occlusion evidence for all94 review.',
        'At77 and78 the far elbow observation has very low confidence, while the far wrist still has0.27/0.38. This does not certify the wrist identity or visibility.',
        'The read-only preview does not change the running candidate or admit Gaussian training.'
    ], result_sha256=a.sha(out / 'result.json'), quality_accepted=False,
    gaussian_admitted=False, candidate_parameters_changed=False,
    all94_review_still_required=True))
print('ORIENTATION_PREVIEW_REVIEWED; NO_GAUSSIAN_ADMISSION')
