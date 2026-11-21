from pathlib import Path
import json,hashlib,shutil
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074';B=L/'outputs/dynamic-human-enerf-residual-graph-067';S=L/'scripts/dynamic_human'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert read(R/'head/independent-audit.json')['status']=='full195context_CPU_SMPL_reprojection_and_affine_recomputation_verified'
assert not (R/'residual_graph/completion.json').exists()
graph=R/'graph.npz'
if not graph.exists():shutil.copy2(B/'graph.npz',graph)
assert sha(graph)==sha(B/'graph.npz')
p=dict(status='fixed_before_RGB_pose_corrected_Gaussian_fit',baseline='067/residual_graph',updates=15000,
 change='CorrectofficialRGBnormalizationforRTMW,refitcubic3Dtrajectories/cameraoffsets,2500bodyupdates,3000headupdates; freeze resultingbodymotion duringGaussianfit.',
 unchanged='Original105nativeRGBandmattes,52109initialGaussians,384graphnodes,graphweights,seed/sampling/backgrounds/optimizer/densitybudgets. No071finergraph,072maskrepairor073handlabels.',
 initial_sha256=sha(L/'outputs/dynamic-human-enerf-articulated-control-064/initial.npz'),graph_sha256=sha(graph),motion_sha256=sha(R/'head/capture-affines.npz'),
 evaluation='102known-camera images include90odd-time developmentimages and12evenchecks. Another18developmentview images require updatedmotionextension before comparison. Final acceptance requires actualimages, not loss.',
 limits='059camerasupportand060canonicalshape remain historical for isolation; observedkeypointagreement isnotGT. Old067remainsretainedpending actualrenderreview.',goal_complete=False)
(R/'gaussian-ablation-protocol.json').write_text(json.dumps(p,indent=2)+'\n');print('Original 384-node graph copied; corrected body audit passed; fixed 15000-update control prepared.')
