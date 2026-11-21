from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
H=Path(__file__).parent
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
O=R/'native-residual-fit'
if O.exists():raise FileExistsError('R6 registered root already exists; inspect, never restart')
O.mkdir()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat()
paths=[H/n for n in ['native_residual_camera122.py','native_residual_domain122.py','native_surface122.py','prepare_native_person122.py','train_native_person122.py','native_training_admission122_v2.py']]
paths+=[Path('D:/GS-SLAM-Data/shuolee-portrait-new-capture-118/src/shuolee_portrait')/n for n in ['fitting.py','surface_visibility.py']]
plan=dict(stage='R6',registered=stamp,hypothesis='Nonzero output residual K/E/D must be used consistently for differentiable anatomical projections, mesh self-visibility and clothing-mask support; point-only replacement leaves mixed projectors.',
    change='Versioned frozen-camera Torch projector; CPU vectorized triangle BVH for exact continuous point-ray self-visibility; versioned shared-static coarse fitter and silhouette-support consumer. Original sources unchanged.',
    visibility_policy='Explicit mesh_surface_ray_v2: ray query at all mesh vertices for coarse regional visibility and at all anatomical face/foot surface points. Full residual projection determines output-raster membership; geometric ray/triangle intersection determines self-occlusion. No warped image, pinhole depth lookup, selected face-frame rules or external-occluder inference. This changes old raster-sampled regional visibility policy and must not be labeled a pure-D causal test.',
    bounds=dict(executions=1,CPU_seconds=600,GPU_seconds=60,person_fit=0,person_training=0,new_features=0,new_camera_optimization=0,new_weights=0),
    controls=['Independent OpenCV Brown12 forward and finite-difference XYZ gradients, CPUfloat64 and one CUDAfloat32 pass; varyingK/E/D and explicit halfpixel bridge.',
    'Known layered planar meshes: correct visible/occluded/backfacing/off-image/behind-camera decisions, analytically known ray depths including parallel/edge rays; world similarity transform invariance.',
    'D0 analytic pinhole agreement and nonzeroD point near raster edge changes membership; no lens can change physical self-occlusion for same visible ray.',
    'Same full per-camera K/E/D in versioned silhouette mask sampler; independent masks, unknown pixels abstain, D0 regression, invalidcamera rejection.',
    'Single deterministic approximately10kvertex/20ktriangle known mesh query benchmark; no scan or restart. Source/contract binding and oldinputguard retained.',
    'Actual failed person residual input remains rejected pretorch/preinitializer; no downstream fit or training. Versioned fitter integration inspected and component-executed only, positive realperson execution not claimed.'],
    stop='Stop and preserve evidence on demonstrated implementation failure; repair exact defect only. No threshold tuning, oldcheckpoint reuse or physical calibration inference.',
    frozen={str(p):sha(p) for p in paths},prior_evidence=str(R/'native-residual-admission/review.json'))
save(O/'plan.json',plan)
q=read(R/'queue-state.json');q.update(updated=stamp,phase='R6_registered',running=None)
q['last_goal_turn']=dict(classification='progress',previous='R5 completed and published with39checks/actualUI/PPT52; actualface13.049px retained.',current='R6 pre-registered frozen-camera fit/visibility/surface integration.',publication_complete=False)
save(R/'queue-state.json',q)
print('R6_REGISTERED')
