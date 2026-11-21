from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, ast
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
D=T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route/free-clothing-transport'
P=Path('D:/GS-SLAM-Data/shuolee-portrait-new-capture-118/src')
G=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/repos/gsplat')
def lp(p):return Path('\\\\?\\'+str(Path(p).resolve()))
def sha(p):return hashlib.sha256(lp(p).read_bytes()).hexdigest()
def save(p,v):
    with lp(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2)
probe=H/'probe_clothing_transport123.py'
assert sha(probe)=='0117e7c78d2812073bcab6a294dae908c83c6344080e30d25d7faac9dceb435e'
assert not lp(D/'native-progress.json').exists() and not lp(D/'native-result.json').exists()
tree=ast.parse(probe.read_text(encoding='utf-8'))
limits=None
for n in tree.body:
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LIMITS' for t in n.targets):
        limits={k.arg:ast.literal_eval(k.value) for k in n.value.keywords}
assert limits['max_forwards']==11 and limits['gpu_wall_seconds']==180
sources=[probe,P/'shuolee_portrait/free_clothing_transport.py',P/'shuolee_portrait/camera_world_units.py',P/'avatar_core/utils/sh_utils.py']
sources += [H/n for n in ['native_portrait_renderer122_v3.py','native_residual_camera122.py','native_scene_package122.py','native_gaussian_ply122.py','native_portrait_renderer122.py']]
sources += [G/n for n in ['gsplat/rendering.py','gsplat/cuda/_wrapper.py','gsplat/utils.py']]
plan=dict(schema='shuolee-clothing-native-probe/1',purpose='synthetic_contract_test',updated=datetime.now(timezone.utc).isoformat(),
    fixture_id='six-free-offset-gaussians-one-directed-triangle-v1',scale_factors=[1.,2.],
    paths=dict(project_src=P.as_posix(),gsplat_repo=G.as_posix()),limits=limits,
    source_sha256={p.as_posix():sha(p) for p in sources},
    hypothesis='Complete per-frame free-centre, full covariance and reference-SH transport agrees with an independently computed camera-space native render and preserves gradients.',
    prior_evidence='18 CPU checks passed; this is one new bounded native CUDA bridge test, not a new person fitting/GS trial.',
    external_timeout_seconds=240,process_preflight='No research Python training/fit process in actual WSL ps; desktop/browser GPU processes present, no observed Python compute process.',
    stop_conditions=['No rerun in an existing native result directory','Any failed fixed check retained and diagnosed without threshold sweep'],
    real_input=False,body_model_forwards=0,gaussian_optimizer_steps=0,training_admitted=False)
save(D/'native-plan.json',plan)
save(D/'native-source-review.json',dict(updated=plan['updated'],reviewer='frame_pixel_bridge_review',source_sha256=sha(probe),
    scope='independent static inspection only, no test execution',remaining_blockers=[],
    verified=['independent camera-space baseline','full covariance factorization','one half-pixel conversion with independent analytic checks','CPU-to-CUDA gradients for free XYZ/quaternion/logscale/SH','single execution and source/plan hashes','fixed 11 forward, one backward, four finite differences'],
    runtime_not_yet_verified=True))
print(json.dumps(dict(registered=str(D/'native-plan.json'),source_sha256=sha(probe),limits=limits)))
