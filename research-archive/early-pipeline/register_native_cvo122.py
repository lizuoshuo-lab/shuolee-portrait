from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,py_compile

H=Path(__file__).resolve().parent
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
O=R/'native-cvo-control'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat()
if O.exists():raise RuntimeError('Existing registration; inspect rather than replace')
files=[H/x for x in ['native_covisibility122.py','native_linear_archive122.py','native_scene_package122_v3.py',
                     'probe_native_cvo122.py','run_native_cvo122.sh']]
for p in files:
    if p.suffix=='.py':py_compile.compile(str(p),doraise=True)
old=read(R/'native-capacity/plan.json')
hashes=dict(old['input_sha256'])
for p,s in hashes.items():
    if sha(p)!=s:raise RuntimeError('Frozen source drift: '+p)
cameras=old['cameras']
e=read(R/'native-training-integration/subpixel-culling-diagnosis/camera.json')['w2c'][0]
for i,c in enumerate(cameras):
    c['id']='t8-'+('portrait' if i==0 else 'landscape');c['image_name']=c['id']+'.png'
    c['w2c']=[row[:] for row in e]
    if i:c['w2c'][0][3]+=.08
plan=dict(stage='T8_CVO_component_and_exact_native_archive',registered=stamp,
    motivation='DirectFisheye paper CVO is projection-independent. Public HEAD loses fullK and uses only last-view densification stats; T7 exposes lossy PLY as insufficient canonical native inference storage.',
    intervention_A='Independently implemented FIT-family covisibility graph/cycle, correct R.T optical axes, average two foreground-normalized view objectives before one Adam step, all-view WORLD gradient/visibility/max-radius statistics. No adaptive split/prune yet.',
    intervention_B='Canonical native linear float32 NPZ without log/logit or quaternion renormalization; standard PLY remains separate preview with measured error. Frozen v2 renderer and camera model unchanged.',
    attribution='A is verified against analytical/joint loss gradients, exposure accounting and independent per-view statistics. B is verified against exact PRE-EXPORT arrays and fullfloat references. No mixed treatment comparison or claim of physical improvement.',
    scope='Synthetic known-geometry engineering controls only. No actual human frame, correspondence or camera fitting; not an official DirectFisheye-GS reproduction or CVO quality ablation.',
    CPU_controls='Whole-family leakage/contradiction exclusion, unsupported graph rejection, matched exposure cycle, world-gauge angle, exact dtype/SH/linear arrays and SHA/path/renderer/camera manifests.',
    GPU_controls='96points/two160x128views: component vs joint averaged loss and one Adam update, points unique to each view, foreground/occlusion guard.200000 frozen points/two full-size views: one fresh mean-objective update, save exact preexport arrays and float RGB/alpha, render both exact archive and standard PLY at same cameras.',
    native_cameras=cameras,optimizer=old['optimizer'],
    limits=dict(cpu_wall_seconds=120,gpu_wall_seconds=300,GPU_jobs=1,large_points=200000,
        large_optimizer_steps=1,small_reference_steps=1,small_component_steps=1,native_roundtrip_max_abs=2e-5,
        gradient_absolute_tolerance=2e-6,gradient_relative_tolerance=2e-4),
    stop='Any implementation failure, nonfinite/OOM/runtime/hash/budget failure: save evidence and stop; no count/resolution/threshold fallback. Demonstrated code bugs may be repaired explicitly, never silently repeated.',
    physical_training_admitted=False,new_weights=False,K_or_human_observations_modified=False,oldT7_gate_unchanged=True,
    continuation='After review integrate this component only through the existing physical admission path. No CVO graph of realpeople accepted yet; density policy and sustained quality ablation are separate explicit remaining work.')
O.mkdir()
write(O/'plan.json',plan)
hashes.update({str(p):sha(p) for p in files})
write(O/'execution-registration.json',dict(registered=stamp,plan_sha256=sha(O/'plan.json'),sources=hashes))
q=read(R/'queue-state.json');q.update(updated=stamp,phase='T8_registered_components',running=None,
    next='Run registered T8 CPU contracts, then one bounded shared-lock synthetic CUDA suite if CPU passes. No human training. Preserve preexport arrays and exact float reference; PLY preview separate.')
q['last_goal_turn']=dict(classification='progress',completed='Independent CVO/linear-native archive implementation and bounded T8 registration.',remaining=q['next'],publication_complete=False)
write(R/'queue-state.json',q)
print(json.dumps(dict(plan=str(O/'plan.json'),files=len(hashes),CPU_started=False,GPU_started=False)))
