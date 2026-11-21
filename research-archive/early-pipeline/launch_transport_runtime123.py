from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess,time,sys
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
D=T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route/free-clothing-transport/native-runtime'
repo=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/repos/gsplat')
def lp(p):return Path('\\\\?\\'+str(Path(p).resolve()))
def linux(p):return '/mnt/'+str(p)[0].lower()+'/'+str(p)[3:].replace('\\','/')
def sha(p):return hashlib.sha256(lp(p).read_bytes()).hexdigest()
lp(D).mkdir(exist_ok=False)
sources=sorted({p for sub in [repo/'gsplat/cuda',repo/'gsplat'] for p in sub.rglob('*') if p.is_file() and p.suffix in ('.py','.cu','.h','.hpp','.cuh','.cpp','.inl')})
sources += [H/'prepare_transport_runtime123.py',H/'transport_runtime123.sh']
plan=dict(created=datetime.now(timezone.utc).isoformat(),scope='Build/load only; numerical test stays unchanged and cannot start before passed result.',
    python='/opt/gs-slam-lab/shuolee-portrait-clean-001/env/bin/python',cache='/opt/gs-slam-lab/transport123-native-cache',
    gsplat_repo=linux(repo),source_sha256={linux(p):sha(p) for p in sources},
    build_seconds=600,build_workers=2,render_calls=0,model_forwards=0,gaussian_steps=0,
    previous_goal_turn='progress: CPU18 transport plus 00 deployment merge; native attempts yielded environment evidence, no numerical result.',
    preflight='Actual WSL process inventory has no prior probe, compiler, or trainer; native servers exist only as Windows HTTP launchers. FreeRAM23GB Cfree199GB.')
lp(D/'plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
cmd=['wsl','-d','Ubuntu-20.04','--','bash',linux(H/'transport_runtime123.sh'),linux(D)]
started=time.monotonic()
with lp(D/'build.log').open('xb') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
lp(D/'exit.json').write_text(json.dumps(dict(returncode=p.returncode,seconds=time.monotonic()-started,command=cmd),indent=2),encoding='utf-8')
if lp(D/'result.json').exists():
    r=json.loads(lp(D/'result.json').read_text(encoding='utf-8'));print(json.dumps({k:v for k,v in r.items() if k!='source_sha256'},indent=2))
else:print('BUILD_STOPPED_WITHOUT_RESULT: inspect exit/build log and process group before further actions')
sys.exit(p.returncode)
