from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
H=Path(__file__).parent
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
O=R/'native-residual-admission'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not O.exists()
assert read(R/'queue-state.json')['running'] is None
assert read(R/'native-residual-capacity/publication.json')['report_UI_verified']
inputs=[R/'native-training-entry/real-input/input.json',R/'native-training-entry/actual-entry/admission.json',
 R/'native-residual-camera/result.json',R/'native-residual-capacity/review.json',R/'native-residual-ui/review.json']
old=[H/n for n in ['material_geometry122.py','native_training_admission122.py','train_native_person122.py',
 'material_observations122.py','native_residual_camera122.py','native_residual_domain122.py','native_training122_v4.py']]
O.mkdir()
now=datetime.now(timezone.utc).isoformat()
save(O/'plan.json',dict(stage='R5',registered=now,kind='formal_input_admission_projection_contract',
 hypothesis='Current T3/N13 admission explicitly supports pinhole only; a future residual renderer cannot share its camera model until target-excluded material evaluation consumes the same complete per-frame camera.',
 change='Versioned CPU material geometry and input audit accept only explicit output-residual Brown6+2+4 cameras with full K/E/native pixel domain. All target and support measurements bridge integer centers to renderer corners by+0.5; D inverse rays produce support-only DLT, forward D produces target projection. No mutation of old modules.',
 triangulation='Same earliest-support rigid frame and maximum support-baseline gauge. Algebraic DLT rows weighted by fx/fy so explicit D=0 is identical to old pixel-DLT. Not nonlinear geometric refinement; no target-in-XYZ leakage.',
 controls=['OpenCV-generated known nonzero D observations in four varied K/E/D cameras; all independent heldout points recover known XYZ/projected pixels.',
 'Explicit zero-D regression for noisy and exact observations; preserve gauge invariance and target exclusion.',
 'Missing/swapped/inverse/raw-fisheye/invalid/folded D,wrongorigin,wrongsource,unregisteredframes,forged/stale hashes,missingphysicalevidence remain rejected.',
 'Wrong D is not ignored: same observations evaluated withD0 retain mismatches, no dropping observations or fitting.',
 'Existing realT3 input copied only fordiagnostic explicitzero-D transport;102targets/13face preserved and original numericalfailure unchanged. Do not claimzeroDistortionphysical.',
 'New audit CLI returns3 on actual failed person input before torch/model/GPU. Old trainCLI rejects new schema, not silently losesD. Native camera accessor returns exact sameKED used by validated v4render component.'],
 budget=dict(executions=1,CPU_wall_seconds=300,GPU=0,new_features=0,new_cameras=0,coefficient_fitting=0,person_training=0),
 stop='Preserve failure and repair only demonstrated implementation defect. No threshold changes, residualcensoring, observationreinterpretation orphysicalacceptance fromsyntheticpass.',
 limits='This stage integrates explicit projection into CPU input admission. Freshbody/cloth initialization and positive persontraining orchestration with nonzeroD still require separate integration; old engine remains unchanged.',
 input_hashes={str(p):sha(p) for p in inputs},historical_hashes={str(p):sha(p) for p in old}))
q=read(R/'queue-state.json');q.update(updated=now,phase='R5_residual_admission_registered',running=None)
q['last_goal_turn']=dict(classification='progress',previous_turn='R3/R4 authoritative19+20checks/UI/PPT complete',completed='Read currentT3/N13 consumer and registeredR5 complete residual-camera admission',remaining='Implement and run once-only CPU controls; no person training.',publication_complete=False)
save(R/'queue-state.json',q)
print('R5_REGISTERED')
