from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, subprocess, time, tempfile, os
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
N=T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route/free-clothing-transport/torch-consumer-gpu'
def lp(p):return Path('\\\\?\\'+str(Path(p).resolve()))
def linux(p):return '/mnt/'+str(p)[0].lower()+'/'+str(p)[3:].replace('\\','/')
def read(p):return json.loads(lp(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(lp(p).read_bytes()).hexdigest()
def save(p,v):
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=lp(p.parent),delete=False) as f:
            temporary=Path(f.name);json.dump(v,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
        os.link(temporary,lp(p))
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
assert sha(N/'plan.json')=='1c93e6d55ef1063e5fa91bad1384bd16f67dc85062a53cdee4dafb79fdc1f777'
assert sha(H/'probe_torch_consumer_gpu123.py')=='cbb5217decd197a21470cc9074036c61b18dfb97a70b5363ce769fea89b591b5'
for name in ['progress.json','result.json','timeout.json']:assert not lp(N/name).exists()
pre=subprocess.run(['wsl','-d','Ubuntu-20.04','--','ps','-eo','pid,ppid,etime,args'],capture_output=True,check=True)
lp(N/'processes-before.txt').write_bytes(pre.stdout)
for line in pre.stdout.decode('utf-8',errors='replace').splitlines()[1:]:
    fields=line.split(None,3)
    if len(fields)!=4:continue
    command=fields[3];base=command.split()[0].rsplit('/',1)[-1]
    assert base not in ('ninja','nvcc','cc1plus'),line
    assert not ('python' in base and any(x in command for x in ['probe_','train','setup_server','render_native'])),line
cmd=['wsl','-d','Ubuntu-20.04','--','env','CUDA_VISIBLE_DEVICES=0','PYTHONNOUSERSITE=1',
     'timeout','--signal=TERM','--kill-after=10s','85s','/opt/gs-slam-lab/shuolee-portrait-clean-001/env/bin/python',
     '-B',linux(H/'probe_torch_consumer_gpu123.py'),'--output-dir',linux(N)]
start=time.monotonic()
with lp(N/'execution.log').open('xb') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
save(N/'exit.json',dict(returncode=p.returncode,seconds=time.monotonic()-start,command=cmd))
if not lp(N/'result.json').exists():
    value=read(N/'timeout.json') if lp(N/'timeout.json').exists() else dict(passed=False,error='ProcessExitedWithoutResult')
    value.update(final_result_owner='parent_after_process_exit',returncode=p.returncode,training_admitted=False)
    save(N/'result.json',value)
post=subprocess.run(['wsl','-d','Ubuntu-20.04','--','ps','-eo','pid,ppid,etime,args'],capture_output=True)
lp(N/'processes-after.txt').write_bytes(post.stdout)
residual=[line for line in post.stdout.decode('utf-8',errors='replace').splitlines() if 'probe_torch_consumer_gpu123.py' in line]
r=read(N/'result.json');ok=bool(r['passed'] and p.returncode==0 and post.returncode==0 and not residual)
save(N/'acceptance.json',dict(passed=ok,child_passed=r['passed'],returncode=p.returncode,residual_processes=residual,
    result_sha256=sha(N/'result.json'),post_process_returncode=post.returncode,source_changed=False,
    native_raster_validated=False,training_admitted=False))
print(json.dumps(dict(passed=ok,message=r.get('message'),checks=len(r.get('checks',[])),
    seconds=r.get('elapsed_monotonic_seconds'),forward_completed=r.get('cuda_forward_completed'),backward_completed=r.get('cuda_backward_completed'),
    max_abs_by_check={c['name']:c['max_abs'] for c in r.get('checks',[]) if 'max_abs' in c and ('gradient' in c['name'] or 'pixels' in c['name'])}),indent=2))
raise SystemExit(0 if ok else (p.returncode or 1))
