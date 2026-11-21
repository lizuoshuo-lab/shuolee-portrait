from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,time,hashlib
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
D=T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route/free-clothing-transport'
N=D/'native-env-repair'
def lp(p):return Path('\\\\?\\'+str(Path(p).resolve()))
def read(p):return json.loads(lp(p).read_text(encoding='utf-8-sig'))
def save(p,v):
    with lp(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def sha(p):return hashlib.sha256(lp(p).read_bytes()).hexdigest()
def linux(p):return '/mnt/'+str(p)[0].lower()+'/'+str(p)[3:].replace('\\','/')
old=read(D/'native-result.json');assert not old['passed'] and old['message']=="'NoneType' object has no attribute 'CameraModelType'"
assert read(D/'native-exit.json')['returncode']==1
assert 'gsplat: No CUDA toolkit found. gsplat will be disabled.' in lp(D/'native-execution.log').read_text(encoding='utf-8',errors='replace')
lp(N).mkdir(exist_ok=False)
plan=read(D/'native-plan.json')
plan.update(updated=datetime.now(timezone.utc).isoformat(),implementation_repair='Launcher omitted CUDA_HOME/PATH/CPATH. Reuse existing verified CUDA12.8; no renderer, fixture, thresholds or sources changed.',
    failed_attempt=str(D/'native-result.json'),failed_attempt_sha256=sha(D/'native-result.json'),
    failed_attempt_gpu_work='First forward counter was incremented but backend=None prevented any native raster call.',
    launcher=str(H/'run_native_transport123.sh'),launcher_sha256=sha(H/'run_native_transport123.sh'))
save(N/'native-plan.json',plan)
cmd=['wsl','-d','Ubuntu-20.04','--','bash',linux(H/'run_native_transport123.sh'),linux(N)]
start=time.monotonic()
with lp(N/'native-execution.log').open('xb') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
save(N/'native-exit.json',dict(returncode=p.returncode,seconds=time.monotonic()-start,command=cmd))
if lp(N/'native-result.json').exists():
    r=read(N/'native-result.json');print(json.dumps({k:r.get(k) for k in ['passed','message','seconds','gpu_seconds','forward_count','backward_count','checks']},ensure_ascii=False,indent=2))
else:print('NO_RESULT: inspect native-execution.log and real PID before any further action')
raise SystemExit(p.returncode)
