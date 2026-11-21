from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
H=Path(__file__).parent;R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122');P=R/'video10-global-initializer/joint-track-candidate';O=P/'convergence-repair';O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'plan.json').exists();control=read(P/'joint-synthetic-control/result.json');audit=read(P/'stationarity-audit/result.json')
assert control['passed'] and control['optimizer']['nfev']>80 and not audit['solver_reported_converged']
files=[P/'initialization.npz',P/'result.json',P/'plan.json',P/'joint-synthetic-control/result.json',P/'joint-synthetic-control/finite-difference.json',P/'stationarity-audit/result.json',R/'video10-remaining-material-review/global-input-preflight/fit-observations.json',R/'video10-remaining-material-review/global-input-preflight/heldout-observations.json',H/'analytic_fixedk_bundle123.py']
plan=dict(registered=datetime.now(timezone.utc).isoformat(),state='registered_not_executed',
    hypothesis='The80-evaluation real candidate did not converge. Known noisy geometry required175evaluations with analytic derivatives. One separately registered numerical convergence repair will distinguish early stopping from residual inconsistency, without changing K or observations.',
    source_sha256={str(p):sha(p) for p in files},
    initialization='Start from the same original saved fresh relative-rotation/bearing initialization.npz, never the optimized80-step candidate. Exact33frame identities,162FIT/563native observations,45HELD/161observations. No old individual geometry or historical optimized camera.',
    implementation='Verified analytic absoluteRodrigues Jacobian. Before execution compare all initial residuals and gauge/parameters against the prior numerical reference, and12central finite-difference columns. Stop on discrepancy.',
    unchanged='ConditionalK,D0,sharedXYZpertrack,firstcamera,largestinitialcentercomponent scale gauge,negative-depth barrier,soft_l1,f_scale1,x_scalejac,SciPyTRF/LSMR default tolerances. No selected roots, point deletion, feature extraction, weight changes or K refinement.',
    causal_scope='Derivative implementation and numerical evaluation budget change together. Improvement cannot be uniquely attributed to either, to K, or to physical alignment. This is numerical repair, not a single physical-factor experiment.',
    budget=dict(real_solver_runs=1,max_nfev=2000,wall_seconds=120,random_restarts=0,K_changes=0,point_removal=0,training_steps=0),
    rationale='2000evaluations is a fixed bounded safety cap exceeding the known control175evaluations. Stop earlier at the existing solver criteria; never extend budget after observing this result.',
    evaluation='Freeze candidate before opening HELD. Same all-other-views DLT per excluded observation as previous run, retain all161nativeEuclidean errors and invalid/behind states. Compare only identical observations, no filtering or HELD feedback.',
    acceptance='Convergence and pixels are reported independently. Even allHELD<1px is not facial truth: physicalFaceHELD=0 and33diagnosticframes are not all860frames. No publication as admitted cameras,SMPL-X initialization or Gaussian training.',
    stop='On source/derivative mismatch,timeout,nonfinite or budget exhaustion preserve evidence and stop. No repeated budget extension or threshold sweep. If convergence still leaves protected errors, do not keep optimizing to the heldout set.')
(O/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(plan=str(O/'plan.json'),source_count=len(files))))
