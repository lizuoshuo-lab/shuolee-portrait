"""Register one bounded full native entry experiment; do not repeat old probes."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,time,sys,os
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
D=T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route/free-clothing-transport'
O=D/'native-training-entry'
P=Path('D:/GS-SLAM-Data/shuolee-portrait-new-capture-118')
def lp(p):return Path('\\\\?\\'+str(Path(p).resolve()))
def linux(p):return '/mnt/'+str(p)[0].lower()+'/'+str(p)[3:].replace('\\','/')
def sha(p):return hashlib.sha256(lp(p).read_bytes()).hexdigest()
def save(p,v):
    with lp(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)
assert not lp(O).exists(),'Registered entry exists; inspect instead of restarting'
assert json.loads(lp(D/'native-training-cpu-result-completed.json').read_text(encoding='utf-8'))['outcome']=='passed'
assert json.loads(lp(D/'framewise-initializer-result-completed.json').read_text(encoding='utf-8'))['passed'] is True
assert sha(P/'src/shuolee_portrait/framewise_native_training.py')=='898a05320dee7fea4921b3cbbfcc3520c24ae88f087539a78dd97be0ae013913'
assert sha(P/'src/shuolee_portrait/framewise_clothed_surface.py')=='aa6316cd6156cb7603d5f8debc18c31fa39b0b429887342ef7ff28202cb3b930'
snapshot=subprocess.run(['wsl','-d','Ubuntu-20.04','--','ps','-eo','pid,ppid,etime,args'],capture_output=True)
assert snapshot.returncode==0
active=snapshot.stdout.decode('utf-8',errors='replace')
tokens=['probe_framewise_training123.py','probe_clothing_transport123_v2.py','probe_torch_consumer_gpu123.py']
assert not any(x in active for x in tokens),'A related probe is already alive'
gpu=subprocess.run(['wsl','-d','Ubuntu-20.04','--','nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True)
assert gpu.returncode==0 and not gpu.stdout.decode('utf-8',errors='replace').strip(),'GPU compute process already present'
lp(O).mkdir()
lp(O/'processes-before.txt').write_bytes(snapshot.stdout)
files=[H/'probe_framewise_training123.py',H/'native_training_fixture123.py',H/'native_portrait_renderer122_v3.py',
       H/'native_residual_camera122.py',H/'native_scene_package122.py',H/'native_gaussian_ply122.py',H/'native_portrait_renderer122.py']
files += [P/'src/shuolee_portrait'/name for name in ['framewise_native_training.py','free_clothing_torch.py','free_clothing_transport.py',
    'frame_pose_contract.py','camera_pixel_contract.py','camera_world_units.py','alignment_inputs.py','framewise_clothed_surface.py']]
files += [P/'src/avatar_core/utils/sh_utils.py']
repo=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/repos/gsplat')
files += [repo/'gsplat/cuda/_wrapper.py',repo/'gsplat/cuda/_backend.py']
plan=dict(schema='framewise-native-entry123/1',purpose='synthetic_native_training_entry',created=datetime.now(timezone.utc).isoformat(),
    hypothesis='Previously separate GPU transport and native ray3D renderer can jointly train one shared reference field with exact named frame and pixel contracts.',
    steps=16,performance_points=200000,max_native_forwards=32,max_backwards=18,wall_seconds=180,
    stop_conditions=['Any independent two-pose RGB/alpha difference >=2e-5','Nonfinite loss/gradient/parameter','Unknown frame/camera/layout or checkpoint mismatch','Joint two-pose loss fails to decrease','180-second budget'],
    performance_scope='Actual six-view silhouette initializer produces200000 points; one65x65 native step, not full native-video resolution or18k throughput. Sample validity/votes reported and not pruned to pass.',
    source_sha256={linux(p):sha(p) for p in files},launcher_sha256=sha(__file__),project_src=linux(P/'src'),gsplat_repo=linux(repo),
    extension='/opt/gs-slam-lab/transport123-native-cache/gsplat_cuda/gsplat_cuda.so',
    extension_sha256='3624acc8fabbfce118034a1ec75a2f6359fcfbb9db848e5cffda6b9508a772e9',
    no_person_data=True,no_real_fitting=True,no_real_GS=True,no_camera_fitting=True,no_automatic_threshold_changes=True,
    launch_preflight_repair='First launch checked the initializer receipt using the trainer outcome string instead of its passed boolean. Assertion stopped before creating output or launching GPU; schema-only reader correction.',
    real_training_admitted=False,old_unchanged_probes_rerun=False)
save(O/'plan.json',plan)
cmd=['wsl','-d','Ubuntu-20.04','--','env','PYTHONNOUSERSITE=1','CUDA_VISIBLE_DEVICES=0','OMP_NUM_THREADS=4',
    'timeout','--signal=TERM','--kill-after=10s','210s','/opt/gs-slam-lab/shuolee-portrait-clean-001/env/bin/python','-B',linux(H/'probe_framewise_training123.py'),linux(O)]
started=time.monotonic()
with lp(O/'execution.log').open('xb') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
save(O/'exit.json',dict(returncode=r.returncode,elapsed=time.monotonic()-started,command=cmd))
if not lp(O/'result.json').exists():save(O/'result.json',dict(passed=False,error='ProcessExitedWithoutResult',returncode=r.returncode,synthetic_only=True))
result=json.loads(lp(O/'result.json').read_text())
post=subprocess.run(['wsl','-d','Ubuntu-20.04','--','ps','-eo','pid,ppid,etime,args'],capture_output=True)
lp(O/'processes-after.txt').write_bytes(post.stdout)
residual=[x for x in post.stdout.decode('utf-8',errors='replace').splitlines() if 'probe_framewise_training123.py' in x]
passed=result['passed'] and r.returncode==0 and post.returncode==0 and not residual
save(O/'acceptance.json',dict(passed=passed,result_sha256=sha(O/'result.json'),returncode=r.returncode,residual_processes=residual,
    real_training=False,synthetic_only=True,goal_complete=False))
print(json.dumps(dict(passed=passed,returncode=r.returncode,result=result),indent=2),flush=True)
sys.exit(0 if passed else 1)
