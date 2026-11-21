from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
H=Path(__file__).parent;R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122');D=R/'video10-correspondence-contrast/head-roi-contrast';P=D/'native-photometric-contrast';O=D/'native-affine-contrast'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not O.exists();audit=read(P/'geometry-audit.json');assert audit['preregistered_support']
plan=dict(registered=datetime.now(timezone.utc).isoformat(),state='registered_not_executed',
    evidence='Translation-only real contrast4→3passing rows,37/144bound hits. Read-only missing-affine derivative projection meets its registered criterion in3of4originallypassing rows. That is exploratory method-development evidence, not independent validation.',
    hypothesis='A31px patch changes scale/shear/rotation across viewpoints, so translation-only is insufficient. Change only local geometric degrees of freedom2→6. Keep source queries,fullframeRoMa seeds,31pxwindow,brightness model,30evaluation budget andfixed confidence/cycle rules.',
    synthetic_control='Same24native source patches,seed12313,noise0.5/255,gain1.05,bias2/255. Known matrix[[1.03,.02],[-.015,.98]],translation[.35,-.40]. Identity geometry initializer. Cubic sampling;8analytic derivative columns checked on rows0,8,16. Every control must converge with center error<0.1px and maximum31pxcorner error<0.5px before real comparison.',
    real_contrast='Only if all synthetic controls pass, one same24query six-edge comparison (max144fits). Local affine coefficients each bounded±.25,translation±3px,brightness unchanged. Reject boundary/singular/flipped solutions. No observation classifications/splits changed.',
    scope='This24query set is now method-development evidence; it cannot certify independent face accuracy or generalized performance. No held camera observations enter camera fitting, and no camera/SMPL-X/Gaussian update is permitted.',
    source_sha256={str(p):sha(p) for p in [D/'review-allocation.json',P/'geometry-audit.json',P/'result.json',H/'native_affine_patch123.py']},
    budget=dict(synthetic_controls=24,real_comparisons=1,max_real_patch_fits=144,max_nfev=30,CPU_seconds=180,new_features=0,new_inference=0,camera_fits=0,training=0),
    stop='No affine-bound/window/gate/noise sweep, repeated fit, or posthoc removal of failed controls. Stop this local patch-method branch after review whether accepted or rejected; any future physical assessment must use a frozen independently reserved set.')
O.mkdir();(O/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(str(O/'plan.json'))
