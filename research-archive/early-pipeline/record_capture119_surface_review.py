"""Persist actual six-view fresh silhouette/center review, before training."""
from capture119_clothed import ROOT,REC,rt
root=ROOT/'portrait';folder=REC/'portrait-surface-001000-review'
rt.save(root/'reviews/surface-decision.json',dict(created=rt.now(),
    actual_initialization_review=True,actual_center_distribution_review=True,
    reviewed_frames=['frame_000001','frame_000023','frame_000045','frame_000067','frame_000089','frame_000112'],
    points_sha256=rt.sha(root/'clothing-init/points.npz'),image_sha256=rt.sha(folder/'body-contact.png'),
    allow_finite_research_training=True,quality_accepted=False,physical_calibration_accepted=False,
    observations='Actual six raw/envelope-render/one-pixel-center panels reviewed. Expanded sweater, skirt, hat and bag silhouette has center support beyond bare SMPL. No proposal-boundary clipping. Face details, boot separation, occluded concavities and back geometry remain unverified. This is an initialization envelope, not accepted surface reconstruction.'))
print('SURFACE_REVIEW_RECORDED_START_FRESH_FINITE_TRAINING')
