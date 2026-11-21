"""Preregister one physical-texture observation intervention, no camera changes."""
from datetime import datetime,timezone
from PIL import Image,ImageDraw
from audit_grayT123_camera_support import REC,ROOT,read,save,sha,native

OUT=REC/'grayT-skin-texture-probe'
if OUT.exists():raise RuntimeError('Existing bounded probe must not be restarted.')
OUT.mkdir()
rows=read(ROOT/'data/training-inputs.json')['records']
seed_i=53;seed_xy=[786.0,1005.0]  # frame54, manual visible pigment spot below image-left eye.
img=native(rows[seed_i]['image'])
with Image.open(img) as im:
    crop=[754,973,818,1037];patch=im.convert('RGB').crop(crop)
patch.save(OUT/'seed_native.png')
marked=patch.copy();d=ImageDraw.Draw(marked);d.ellipse((29,29,35,35),outline=(0,255,255),width=1)
marked.save(OUT/'seed_locator.png')
plan=dict(created=datetime.now(timezone.utc).isoformat(),status='seed_visual_review_pending',
    causal_variable='Replace ambiguous eye/mouth/silhouette feature identity with one manually identified visible skin pigment spot. Existing K/E and original training remain unchanged.',
    prior_followup='The28component/all279node review has completed. None certified fixed static skin material; saved12HELDprojections reproduced. This is a new bounded observation test, not a rerun of SIFT/BA.',
    seed=dict(frame=rows[seed_i]['frame'],source_frame_index=rows[seed_i]['source_frame_index'],xy=seed_xy,
        description='A small dark skin mark below the outer image-left eye in front/three-quarter views.',
        path=str(img),source_sha256=rows[seed_i]['source_sha256'],crop=crop),
    native_frame_numbers=list(range(40,81)),
    selection='Contiguous41already-decoded native frames40..80 from the visibly exposed eye/cheek side; not selected by geometric residual.',
    anchor_frame_numbers=[40,48,56,64,72,80],
    held_frame_numbers=[i for i in range(40,81) if i not in [40,48,56,64,72,80]],
    observation_method=dict(method='Native-pixel pyramidal Lucas-Kanade from manual seed, tracked both temporal directions; each link checked reverse.',
        window=[31,31],pyramid_levels=3,max_iterations=30,epsilon=.01,min_eigenvalue=.0001,
        forward_backward_max_px=.5,full_source_pixels_preserved=True,no_geometric_prediction=True,
        metric_limit='Flow roundtrip is only algorithmic repeatability, not a physical subpixel uncertainty certificate.'),
    original_scene_negative_controls=['8036d2b7f840','ffef1195aa16'],
    negative_control_rule='Use only already-saved independent check observations at exactly matching frame identities; never turn point anchors into checks. Report missing simultaneous coverage and depth differences.',
    budgets=dict(tracking_runs=1,points=1,frame_count=41,CPU_seconds=120,camera_updates=0,new_BA=0,feature_detector_calls=0,training_runs=0),
    gates=['Inspect seed before running. If wrong/no visible spot, stop this planned probe without parameter search.',
           'Save all41observations and every LKfailure. Actual native point-marker review of allframes must precede geometric residuals.',
           'If any LKcheck or visible physical identity fails, stop before fitting3D; do not shorten the sequence by errors.',
           'Only after all identity checks, one shared3Dpoint from the6registered anchor observations is allowed as a diagnostic with fixed K/E. All35predeclared other observations remain HELD.',
           'No reuse as training initial geometry or fit update; single-point results cannot separate K/E from head motion or certify fullface.'],
    sources=[dict(path=str(p),sha256=sha(p)) for p in [ROOT/'camera/solver-output.npz',ROOT/'data/training-inputs.json',REC/'grayT-material-identity-review/actual-review.json',REC/'grayT-material-static-check/result.json']])
save(OUT/'plan.json',plan)
print(OUT)
