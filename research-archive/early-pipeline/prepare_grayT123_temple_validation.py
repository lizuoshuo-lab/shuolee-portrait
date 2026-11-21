"""Freeze one additional pigment observation test before tracking or geometry."""
from datetime import datetime,timezone
from pathlib import Path
from PIL import Image,ImageDraw
from audit_grayT123_camera_support import REC,ROOT,read,save,sha,native
OUT=REC/'grayT-temple-validation-review'
if OUT.exists():raise RuntimeError('Existing temple validation must not restart.')
review=read(REC/'grayT-visibility-material-review/actual-review.json')
assert review['all41_nose_pairs_and41_heads_actually_reviewed']
OUT.mkdir();rows=read(ROOT/'data/training-inputs.json')['records'];row=rows[53];seed=[792.5,965.5]
with Image.open(native(row['image'])) as im:raw=im.convert('RGB').crop((760,933,824,997))
raw.save(OUT/'seed_native.png');draw=ImageDraw.Draw(raw);x,y=seed[0]-760,seed[1]-933
draw.ellipse((x-4,y-4,x+4,y+4),outline=(0,255,255),width=1);raw.save(OUT/'seed_locator.png')
files=[ROOT/'data/training-inputs.json',ROOT/'camera/solver-output.npz',REC/'grayT-head-rigid-review/frozen-proxy-fits.json',
    REC/'grayT-skin-localization-review/geometry-result.json',REC/'grayT-visibility-material-review/actual-review.json']
p=dict(created=datetime.now(timezone.utc).isoformat(),status='seed_requires_actual_review_before_first_tracking',
    question='Does a second physically distinct skin mark show the same static inconsistency and support the already-frozen head-rigid prediction?',
    causal_variable='Additional physical material observation; camera, all frozen head transforms and all previous measurements remain unchanged. No new pose fitting.',
    material_id='temple_pigment_M1',seed=dict(frame=row['frame'],native_xy=seed,source_sha256=row['source_sha256'],
        description='小深色颞部色素斑，位于原M0眼外侧点上方约30px。手工种子只给待测中心，不代表亚像素真值。'),
    frames=list(range(40,81)),anchors=[40,48,56,64,72,80],held=[i for i in range(40,81) if i not in [40,48,56,64,72,80]],
    tracking=dict(method='One native LK run, two temporal directions from54, independent of all geometry and the oldM0track.',
        window=[31,31],levels=3,iterations=30,epsilon=.01,min_eigenvalue=.0001,forward_backward_threshold_px=.5),
    localization=dict(method='Reuse the frozen13x13 negativeGaussian+planarbackground intensity model on each LKcenter once.',
        helper='localize_grayT123_skin_spot.fit_patch',parameter_count=8,center_bounds_from_LK_px=3,sigma_bounds_px=[.6,3.5],max_nfev=200,
        uncertainty='Conditional Hessian is not a physical accuracy certificate.'),
    geometry=dict(before='All41tracking and localized points must be reviewed for same visible material before reading reprojection results.',
        hypotheses=['static point under originalK/E','canonical point under already-frozen nose-proxy H_i; originalK/E unchanged'],
        point_fit='One3Dpoint per hypothesis from only6anchors; all35HELD retained, no errorfilter.',
        max_nfev_each=100,pose_or_camera_fits=0,held_overlap_note='Images and oldM0helderrors already seen; newM1 is another physical marker, not a newunseenvideo. Neighboring marker errors may be correlated.'),
    budget=dict(tracking_runs=1,localization_runs=1,frames=41,CPU_seconds_each=120,point_fits=2,pose_fits=0,camera_candidates=0,training=0),
    stops=['Seed not clearlyonthismaterial -> stop beforetracking.',
        'Any LKstatus/fb>=.5, photometric boundhit/solverfailure, or physicalidentity/visibilityfailure -> preserveallobservations and stop beforegeometry. Do not shortenframes or retune.',
        'Do not updatecameras, oldfittedgeometry, imagesortraininginitialization. Even bothmarkerspassing doesnotcertify eyes/nose/mouth or wholebody, motion or physicalcalibration.',
        'If predictivefit doesnotpasssubpixel orobservabilityisweak, recordfailedtest and do notstarttraining.'],
    sources=[dict(path=str(f),sha256=sha(f)) for f in files],quality_accepted=False)
save(OUT/'plan.json',p);save(REC/'grayT-temple-validation-plan.json',p)
print('Registered one additional material validation. No tracking or geometry executed.')
