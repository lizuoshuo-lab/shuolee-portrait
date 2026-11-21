"""Record the actual native and four-direction review before GS preparation."""
from run_official_gaussians123 import read,save,sha,now
from run_sofa_half8_fit124 import OUT

assert not (OUT/'conclusion.json').exists()
r=read(OUT/'review/result.json')
for row in r['records']:
    for image in row['images'].values():assert sha(image['path'])==image['sha256']
save(OUT/'conclusion.json',dict(updated=now(),actual_visual_review=True,
    inspected_frames=[x['frame'] for x in r['records']],regions=['body','face','hands','feet','four_novel_directions'],
    decision='Exploratory Gaussian admission, not pose-quality acceptance',exploratory_gaussian_admitted=True,
    body='All eight remain coherent seated poses with wrists in lap; no new gross elbow/knee reversal or body collapse observed. Clothing lies outside the initialization.',
    face='Visible nose/mouth and face orientation broadly align; native detector RMS2.78-5.63px, not subpixel or physical truth. Beard/hair/eyeglasses not represented by the naked mesh.',
    hands='Visible near fingers are generally aligned. Overlapping far fingers and contact remain uncertain; no independent 3D ground truth or hand-quality acceptance.',
    feet='Original shoe/mesh mismatch persists, especially far foot in46; some shoes reach/cross frame boundary. No unseen-foot recovery claim.',
    estimated_supported_arc_deg=r['estimated_arc_degrees'],physical_angular_calibration=False,
    coverage='Eight samples do not certify a180degree arc: fitted-body proxy covers about99.5degrees; rear absent. Extra21adds frontal support, not a new side direction.',
    no_frames_removed_after_fit=True,old_fit_or_optimizer_reused=False,source_targets_unchanged=True,
    next='Fresh20908one-per-face SH0 with reviewed ROI2x/pairedbackground/18ktime profile; actual1000gate, at most18000.',
    quality_accepted=False,source_review_sha256=sha(OUT/'review/result.json'),fit_sha256=sha(OUT/'selected-fit.npz')))
print('SOFA_HALF8_EXPLORATORY_GS_ADMITTED')
