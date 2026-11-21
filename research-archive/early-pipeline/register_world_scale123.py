from datetime import datetime, timezone
import hashlib, json
from pathlib import Path

T = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
ROUTE = T / 'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route'
A = ROUTE / 'training-interface-audit'
P = Path('D:/GS-SLAM-Data/shuolee-portrait-new-capture-118')
out = A / 'saved-snapshot'
out.mkdir(exist_ok=True)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
plan = dict(registered=datetime.now(timezone.utc).isoformat(),
    previous_turn='progress: frame identity API,27 CPU checks, publication and 00 dispatch completed',
    causal_variable='Undeclared camera translation length units versus SMPL-X camera-space model meters',
    finding='Saved diagnostic world vertices were produced by inverse SfM E without the common metric scale used in old static fitting. Pixel roundtrip alone does not constrain this scale.',
    action=['Read original source, fixed camera solver result, saved three-frame arrays and original image/video hashes once; preserve all inputs.',
            'Do not manufacture a real meter-labelled bundle if common scale is absent.',
            'Add explicit common camera-length metadata and conversion to sequence world export; reject missing or per-frame scale. Retain model-conditioned versus measured basis.',
            'Use known synthetic rigid geometry to show that wrong units can preserve own-camera pixels while separating shared 3D points; verify correct conversion and gradients without model forward.'],
    conditional_data='Old static scale estimates are evidence of a previous approach, not certified new initialization or a measured calibration.',
    allowed_changes=['new camera_world_units.py','sequence_fitting.py explicit length contract','focused CPU regression'],
    limits=dict(real_fit=0,NN_forward=0,GPU_jobs=0,Gaussian_steps=0,old_camera_overwrite=False,old_fit_overwrite=False,weights_download=False),
    budget=dict(synthetic_cases='single deterministic analytic geometry; no parameter sweep',CPU_seconds=120),
    stop=['Missing provenance: preserve unknown and refuse real admission','Any source hash changes during reading: stop','No calibration promotion from synthetic test or metadata alone'],
    protected_sources={str(p):sha(p) for p in [ROUTE/'shared-three-frame/fit.npz',ROUTE/'shared-three-frame/initial.npz',ROUTE/'shared-three-frame/plan.json',P/'src/shuolee_portrait/sequence_fitting.py']},
    training_admitted=False)
save(out/'scale-contract-plan.json',plan)
print(json.dumps({'plan':str(out/'scale-contract-plan.json'),'real_fit':False}))
