from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122');G=R/'video10-global-initializer';U=R/'video10-remaining-material-review/global-input-preflight';O=G/'joint-track-candidate'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
audit=read(G/'multiview-support/result.json');assert audit['material_only_extra_triples']==159 and len(audit['triangle_missing_frames'])==6
O.mkdir(exist_ok=True);assert not (O/'plan.json').exists()
inputs=[U/'fit-observations.json',U/'heldout-observations.json',U/'conditional-camera.json',G/'relative-graph/repair-invalid-pose/pairs.json',G/'multiview-support/result.json',R/'video10-remaining-material-review/generic-observability/synthetic-input-and-jacobian.npz']
plan=dict(registered=datetime.now(timezone.utc).isoformat(),state='registered_not_executed',
 hypothesis='Shared XYZ tracks carry constraints even when a pair has fewer than5matches. One joint fixed-K solve over all162FIT tracks may connect the six views missing from the relative-pose triangles. This tests joint-track geometry, not K changes or a proof of global optimum.',
 evidence='Read-only audit found159material triples outside pair graph, maximum4same tracks per triple. All6missing views retain8to11FIT long tracks. Prior generic Jacobian had full677rank; actual shape/depth/identity are not yet verified.',
 source_sha256={str(p):sha(p) for p in inputs},
 frozen='33diagnosticframes,162FIT/45HELD,563FITnativeUV,conditionalK,D=0. No new features, prior individual geometry, old optimized camera, new RANSAC, or HELD fitting. Original12per-pair protection stays failed; diagnostic output cannot be published as admitted cameras.',
 initialization='Reuse the saved57fresh relative rotations without changing their branches as a single starting state. Connection-Laplacian global rotation average; joint world bearing least squares over all camera centers,points and positive ray depths. First camera fixed, scale normalized by median baseline. This is deliberately conditional on possibly wrong starting rotation branches.',
 control=dict(kind='One fixed known synthetic geometry with existing162track visibility, known rotations,0.2px Gaussian pixel noise,seed12311',
    maximum_projection_error_px=1.5,mean_projection_error_px=0.5,positive_depth_fraction=1.0,position_solver_max_iterations=100,
    meaning='Verify noise-aware positive-depth solver implementation only. No hidden retry or noise/threshold search.'),
 bundle=dict(variables='All remaining camera rotations/centers and one sharedXYZ per FIT representative; K fixed.',
    gauge='R0=I,C0=0; fix largest absolute initialized camera-center component across all cameras exactly.',loss='soft_l1',f_scale_px=1.0,max_nfev=80,
    behind_penalty='Extra negative-depth residual1000*min(z/initial_median_positive_depth-1e-5,0). Final any negative depths reported, never silently filtered.',
    solver='SciPy least_squares sparse two-point Jacobian with lsmr; one continuous candidate, no multistart/point deletion.'),
 evaluation='After FIT optimization freezes, leave one view out of eachHELD track. Triangulate using all remaining observations and predict the excluded view. Report all Euclidean native-pixel values, behind-camera/invalid cases and per-frame coverage. No pair subset selection, residual deletion or HELD feedback.',
 budget=dict(total_wall_seconds=300,noise_controls=1,real_candidates=1,position_solver_max_iterations=100,BA_runs=1,random_restarts=0,K_changes=0,training_steps=0),
 acceptance='Diagnostic only. Does not relax12pairgate or physical face<1px acceptance. All real frame positions, protected material validity, all-view body/face/background and full predeclared training frames still required. No SMPL-X or Gaussian initialization from this output.',
 stop='Control failure, timeout or nonfinite math: preserve error and stop. Real fit failure keeps candidate as failed, no parameter search. Do not claim a local least-squares optimum is a unique physical camera solution.')
(O/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(path=str(O/'plan.json'),sha256=sha(O/'plan.json'))))
