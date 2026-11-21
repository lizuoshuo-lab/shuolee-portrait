"""Freeze a single real relative-pose diagnostic, with no body or Gaussian fit."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
G=R/'video10-global-initializer';U=R/'video10-remaining-material-review/global-input-preflight';D=G/'relative-graph'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert read(G/'result-completed.json')['state']=='four_synthetic_controls_passed_status_write_recovered'
D.mkdir(exist_ok=True);assert not (D/'plan.json').exists()
inputs=[U/'fit-observations.json',U/'heldout-observations.json',U/'conditional-camera.json',U/'essential-minimal-graph.json',G/'result-completed.json']
plan=dict(registered=datetime.now(timezone.utc).isoformat(),state='registered_not_executed',
 hypothesis='Given the reviewed native material identities, test whether calibrated relative poses agree across the fixed 33-view graph. Separate insufficient per-pair support, competing planar explanations and inconsistent rotations before using global initialization.',
 changed='Newly reviewed162FIT families, one global relative-edge assessment instead of sequential seed/PnP. Does not isolate K or prove K correctness.',
 source_sha256={str(p):sha(p) for p in inputs},
 inputs='Native integer-center MP4 coordinates; fixed conditional K and D=0. No new features, no old poses/points, no synthetic truth in real solve. HELD may only be evaluated after each FIT pose is frozen.',
 edge_list='Exactly58 registered edges with at least5 shared FIT representatives. The mathematical minimum is diagnostic-only and never a robustness admission.',
 solver=dict(backend='Existing PoseLib2.0.5 calibrated five-point LO-RANSAC',max_epipolar_error_px=1.5,min_iterations=100,max_iterations=10000,seed=12310,passes=1),
 geometry=dict(decompose_all_four_E_branches=True,triangulation='Native-normalized two-view DLT',min_median_parallax_deg=0.5,min_positive_depth_fraction=0.9),
 robust_edge=dict(min_FIT_correspondences=12,min_FIT_inliers=12,origin='Keep video10 original12sharedFIT seed support; do not replace it by the mathematical5minimum.',min_inlier_fraction=0.75),
 alternative=dict(homography='One fixed OpenCV RANSAC at1.5px,seed12310,max10000. Forward and backward errors reported. A plane fitting at least90percent of E inliers is a competing explanation, not proof of a plane or wrongK.',competing_fraction=0.9),
 cycles=dict(all_triangles=True,report_degrees=True,screening_flag_deg=0.5,interpretation='Screen only. A low cycle error does not certify physical correspondence or subpixel calibration. No edge pruning or re-estimation from cycle errors.'),
 global_rotation='Only run the checked rotation average if all33frames are connected by edges meeting12FIT/inliers,75percent inlier,90percent positive depth,0.5deg parallax and no competing homography. Otherwise preserve relative diagnostics and explicitly refuse full camera output. No absolute positions or BA in this pass.',
 heldout='Report every shared frozen HELD family Sampson distance and full values; none used in RANSAC,model selection,pose refinement or filtering. Epipolar distance is NOT 3D reprojection and does not satisfy face acceptance.',
 budget=dict(CPU_seconds=300,pairs=58,random_restarts=0,threshold_sweeps=0,new_features=0,BA_runs=0,training_steps=0),
 stop='One run; preserve every edge including failure, all HELD evaluation and full components. Any robust graph disconnection blocks global camera output, not a claim that all global methods/data are impossible.',
 acceptance='Diagnostic only. Coarse material review is not physical subpixel ground truth; face HELD0,full860frames unregistered,conditionalK unknown. No person fit/training admission.')
(D/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(plan=str(D/'plan.json'),sha256=sha(D/'plan.json')),ensure_ascii=False))
