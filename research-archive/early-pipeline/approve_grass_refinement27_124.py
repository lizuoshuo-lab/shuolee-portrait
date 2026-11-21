"""Save completed27-frame visual inspection and continue the frozen candidate."""
import audit_gaussianavatars_params123 as a
from datetime import datetime,timezone

out=a.REC/'cross-subject-sh0-124/grass-temporal-refinement'
p=a.read(out/'plan.json');request=a.read(out/'review-request-027.json')
result=a.read(out/'review-027/result.json');prior=a.read(out/'review-013/result.json')
assert len(request['frames'])==result['frames']==27
assert not a.lp(out/'review-decision-027.json').exists()
assert a.read(out/'review-decision.json')['continue_same_candidate']
def local(path):return path[5].upper()+':/'+path[7:] if path.startswith('/mnt/') else path
for path,digest in p['sources'].items():assert a.sha(local(path))==digest,path
rows={r['frame']:r for r in result['records']}
for row in result['records']:
    for im in row['images'].values():assert a.sha(local(im['path']))==im['sha256']
for row in prior['records']:
    current=rows[row['frame']]
    assert current['metrics']==row['metrics']
    assert current['images']==row['images']
transition=max(result['transitions'],key=lambda t:t['refinement']['max_local_joint_deg'])
visual=dict(updated=datetime.now(timezone.utc).isoformat(),actual_visual_review=True,
    reviewed_frames=sorted(request['frames']),unchanged_prior13_verified=True,
    prior_visual_review_sha256=a.sha(out/'review-013/visual-review.json'),
    new_body_common_side_pages=[result['pages'][i-1] for i in [1,2,6,7]],
    new_native_face_feet_pages=[v for i,v in enumerate(a.read(out/'review-027/face-feet-pages.json'),1) if i in [1,2,3,4,11,12,13,14]],
    enlarged_native_body_frames=[19,20,21,44,45],enlarged_four_direction_frames=[19],
    findings=['All27 raised arms are in front; the old44->45 front/back switch is absent.',
        'All14 newly computed frames reviewed in native body/face/feet plus common sides. Previous13 pixels and metrics are unchanged.',
        'No new gross shoulder/knee/ankle flip or collapse. Clothing silhouette is not equated to naked SMPL-X.',
        'There remains a24.96deg local joint change from20to21 and about0.12 model-unit relative wrist change; keep this visible for full-sequence assessment.',
        'Native foot mean residuals can increase about1-2px against the independent baseline; overlays retain existing toe/ankle mismatch. Face detector residuals are generally lower, but not physical3D truth.',
        'This intermediate decision permits remaining computation only; no final quality acceptance or GS.'],
    maximum_transition=transition,checkpoint_passed=True,full_sequence_quality_accepted=False,
    gaussian_admitted=False,source_hashes_verified=True)
a.save(out/'review-027/visual-review.json',visual)
a.save(out/'review-decision-027.json',dict(updated=visual['updated'],continue_same_candidate=True,
    reviewed_frames=request['frames'],review_result_sha256=a.sha(out/'review-027/result.json'),
    visual_review_sha256=a.sha(out/'review-027/visual-review.json'),actual_visual_review=True,
    reason='No persistent backward raised-arm direction or gross native local regression in all27. Continue unchanged to all94; retain20->21 movement and foot mismatch for final review. No Gaussian admission.',
    quality_accepted=False,gaussian_admitted=False))
print('SAME_CANDIDATE_CONTINUES_TO_94; NO_GAUSSIAN_ADMISSION')
