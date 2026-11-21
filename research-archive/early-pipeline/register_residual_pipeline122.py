from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
H=Path(__file__).parent
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
O=R/'native-residual-pipeline'
if O.exists():raise FileExistsError('R7 already registered; inspect existing execution')
O.mkdir()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat()
names=['native_training122_v4.py','native_covisibility122.py','native_scene_package122_v5.py','native_training_admission122_v2.py','prepare_native_person122_v2.py','train_native_person122.py','native_surface122_v2.py','fitting_output_residual122.py','native_residual_fit122.py','fit_gate_output_residual122.py']
plan=dict(stage='R7',registered=stamp,hypothesis='The actual preparation and person optimizer must transport the same frozen output K/E/D and fresh surface/source identities, not fall back to the previous pinhole optimizer.',
    change='Versioned full residual training loop and person entry, explicit balanced single-view or evidenced two-view CVO policy, all-selected-view gradient/visibility statistics, source-bound fresh preparation and checkpoint/review dispatch.',
    algorithm_policy='200000 free world centers; fixed count is explicit, no adaptive splitting/pruning in this stage. CVO is independently implemented and optional, never inferred from face indices or heldout points. A camera/material-qualified run may explicitly use uniform_single; CVO requires separate frozen FIT foreground physical-family evidence and supported cycle. No change to existing camera/material thresholds.',
    bounds=dict(executions=1,CPU_seconds=600,GPU_seconds=120,small_known_scene_points=96,small_scene_updates_max=16,person_fit=0,person_training=0,new_camera_candidates=0,new_features=0,new_weights=0),
    controls=['Complete-camera propagation and one-image-at-a-time native-pixel cache through actual new entry components; missing or changed coefficients/camera/image identities reject.',
      'Fresh initialization provenance binds same run/source/frameplan/cameras/observations/masks, exact fit and initializer code, prior outputs, actual fit/surface review and zero-step200000array shape. No old checkpoint/optimizer resume.',
      'Deterministic balanced view scheduling and explicit CVO support, mixed-family/heldout leakage rejected; no silent random fallback on unavailable support.',
      'Small known nonzeroD scene: actual repeated fresh optimizer loop, two-view mean gradients checked against independent joint loss; all-view worldgradient/radius statistics include both views.',
      'Review pause uses a hash-bound recorded-camera snapshot and native original-aspect RGB/alpha; no approval means stop; no new optimizer on continuation. Exact native NPZ/scene readback and actual CLI failure-beforeGPU verified.',
      'Actual existing failed102target/13face input at new person and preparation entry retains max13.049px and rejects before GPU/init/optimizer. Real positivepersonfit and18k not executed.'],
    stop='One bounded suite only. Preserve failure; repair demonstrated implementation defect without tuning a quality threshold. No old experiment rerun or synthetic success relabeled physical acceptance.',
    frozen={str(H/n):sha(H/n) for n in names},previous_result=str(R/'native-residual-fit/result-completed.json'))
save(O/'plan.json',plan)
q=read(R/'queue-state.json');q.update(updated=stamp,phase='R7_registered',running=None)
q['last_goal_turn']=dict(classification='progress',previous='R6 completed29checks and actualUI/PPT52 publication.',current='R7 formal full-camera optimizer/preparation dispatch integration.',publication_complete=False)
save(R/'queue-state.json',q)
print('R7_REGISTERED')
