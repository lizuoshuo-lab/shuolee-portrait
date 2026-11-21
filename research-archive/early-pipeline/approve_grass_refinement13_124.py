"""Persist the completed manual checkpoint review; allow only the same fit."""
import audit_gaussianavatars_params123 as a
from datetime import datetime,timezone

out=a.REC/'cross-subject-sh0-124/grass-temporal-refinement'
p=a.read(out/'plan.json');request=a.read(out/'review-request.json')
result=a.read(out/'review-013/result.json')
assert len(request['frames'])==result['frames']==13
assert not a.lp(out/'review-decision.json').exists()
def local(path):return path[5].upper()+':/'+path[7:] if path.startswith('/mnt/') else path
for path,digest in p['sources'].items():assert a.sha(local(path))==digest,path
for row in result['records']:
    for im in row['images'].values():assert a.sha(local(im['path']))==im['sha256']
assert a.read(out/'anchor-parity.json')['passed']
visual=dict(updated=datetime.now(timezone.utc).isoformat(),actual_visual_review=True,
    reviewed_frames=sorted(request['frames']),body_and_common_side_pages=result['pages'],
    native_face_feet_pages=a.read(out/'review-013/face-feet-pages.json'),
    enlarged_native_body_frames=[33,35],enlarged_four_direction_frames=[33],
    findings=['All13 raised arms remain in front of the torso instead of flipping behind it.',
        'All native face/feet crops reviewed; no new major local collapse or flip. Existing ankle/toe and naked-body/clothing mismatch remain.',
        'Compared with the immediately preceding free-placement control, largest feet weighted-mean increase is less than1px; this is detector-based support, not independent ground truth.',
        'Frame32->33 max joint change is3.53deg versus80.54deg in the previous control; maximum over this13-frame segment is17.54deg.',
        'This is checkpoint permission to compute later frames, not acceptance of the whole sequence or Gaussian training.'],
    checkpoint_passed=True,full_sequence_quality_accepted=False,gaussian_admitted=False,
    source_hashes_verified=True,reference_initialization_parity_passed=True)
a.save(out/'review-013/visual-review.json',visual)
a.save(out/'review-decision.json',dict(updated=visual['updated'],continue_same_candidate=True,
    reviewed_frames=request['frames'],review_result_sha256=a.sha(out/'review-013/result.json'),
    visual_review_sha256=a.sha(out/'review-013/visual-review.json'),actual_visual_review=True,
    reason='All13 front-arm depths and native body/face/feet protections passed the intermediate review. Continue unchanged to registered27-frame checkpoint including44/45; no GS or final acceptance.',
    quality_accepted=False,gaussian_admitted=False))
print('SAME_CANDIDATE_CONTINUES_TO_27; NO_GAUSSIAN_ADMISSION')
