from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
ROUTE=T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route'
D=ROUTE/'free-clothing-transport';D.mkdir(exist_ok=True)
def lp(p):return Path('\\\\?\\'+str(Path(p).resolve()))
def save(p,v):lp(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
stamp=datetime.now(timezone.utc).isoformat()
save(D/'plan.json',dict(updated=stamp,parent_plan=str(ROUTE/'training-interface-audit/saved-snapshot/transport-next-plan.json'),
    previous_turn='progress: unit/checkpoint guards, real saved identity audit, and authorized 00 handoff completed',
    hypothesis='Explicit frame/point identities and one free reference field can transport centres/full covariance/reference SH directions together. Consistent per-frame transport may cancel arbitrary camera translation scale in foreground rendering.',
    field='One fixed support triangle per Gaussian (support only, no mesh-position binding). Oriented bases define Q; sqrt(area ratio) defines a; pivot is each triangle centroid. Non-similar support distortion is reported, not claimed to be exactly interpolated.',
    fixed_data='Small deterministic analytic cameras/meshes/off-surface points/full SPD covariance and degree1/3 SH; no real input or learned point cloud',
    changes=['New reusable free_clothing_transport module only','Independent known-geometry tests','Two predeclared positive camera scale gauges 1 and 2, not fitted parameter candidates'],
    gauge_control='Convert camera translations AND per-frame world meshes/reference field consistently. Compare camera-space centres, covariance, SH directions and analytic full-K pixels. Retain omitted-warp negative control.',
    source_causal_review='sequence_fitting.py per-frame camera-space pose and projection residual do not depend on world E. Hence neither shared beta nor own-frame projection identifies SfM translation scale; complete pose warp may cancel E.',
    budget=dict(CPU_seconds=120,GPU_jobs=0,real_fits=0,model_forwards=0,gaussian_steps=0),
    pass_scope='Declared local-similarity field consistency and analytic projection only; not cloth material identity, CUDA rasterization or physical calibration.',
    stop_conditions=['Unknown/mismatched frame or point identities rejected','Missing units/degenerate support/nonfinite data rejected','Any failed check diagnosed before one implementation repair; no thresholds swept'],
    physical_scale_still_unresolved=True,absolute_world_scale_not_automatically_a_foreground_training_blocker=True,
    real_training_admitted=False))
save(D/'scale-identifiability-review.json',dict(updated=stamp,reviewer='independent scale_identifiability_review',
    source='src/shuolee_portrait/sequence_fitting.py:140-148,224-237,314-328',
    confirmed='Current fitting residual uses camera-space joints and full K. E and its translation scale do not enter residual; per-frame root translation/orientation remain independent.',
    equation='T_i(s)=inverse(E_i(s)) o F_i; W_i<-r(s)=T_i(s) o inverse(T_r(s)); E_i(s) o W_i<-r(s) o T_r(s)=F_i',
    inference='With complete equivariant per-frame transport, foreground projection can be invariant to scale and even scene extrinsics. This does not establish accurate scene registration.',
    conditions=['Centres, covariance, reference SH direction, query initialization and actual projection must use the same frame chain.',
                'Current static trainer does not satisfy this condition; it uses identical meshes and no explicit frame transport.'],
    prior_finding_scope='A missing explicit unit declaration and misleading metric-world interpretation are real implementation defects. Neither proves a unique real-ghosting cause.',
    admissible_next='Perform registered known-geometry gauge invariance and omitted-warp control; if proved, explicitly labelled model-coordinate/gauge reconstruction can proceed independently of metric scene registration.',
    physical_scale_estimable_only_with='Additional declared cross-frame world constraints or measured baseline/length/depth. A static-pelvis assumption gives conditional scale, not ground truth.',
    new_fit=False,new_GPU=False))
print(str(D))
