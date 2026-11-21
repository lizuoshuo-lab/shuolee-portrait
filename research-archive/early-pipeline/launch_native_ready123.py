"""Register and execute the unchanged numerical fixture after verified native load."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, subprocess, time, os, tempfile

H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
D=T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route/free-clothing-transport'
N=D/'native-ready'
def lp(p): return Path('\\\\?\\'+str(Path(p).resolve()))
def linux(p): return '/mnt/'+str(p)[0].lower()+'/'+str(p)[3:].replace('\\','/')
def read(p): return json.loads(lp(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(lp(p).read_bytes()).hexdigest()
def save(p,v):
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',newline='\n',dir=lp(p.parent),delete=False) as f:
            temporary=Path(f.name)
            json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
            f.flush(); os.fsync(f.fileno())
        os.link(temporary,lp(p))
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)

runtime=read(D/'native-runtime/readback-result.json')
assert runtime['passed'] and runtime['render_calls']==0
probe=H/'probe_clothing_transport123_v2.py'
assert sha(probe)=='656a0978cb5d28d3b9b037d18b34bd79ded31ffc8fcbab2d9b62f77a77f3ba36'
plan=read(D/'native-plan.json')
for p,d in plan['source_sha256'].items():
    assert sha(p)==d, p
del plan['source_sha256'][(H/'probe_clothing_transport123.py').as_posix()]
plan['source_sha256'][probe.as_posix()]=sha(probe)
plan.update(updated=datetime.now(timezone.utc).isoformat(), native_runtime=runtime,
    native_runtime_receipt_sha256=sha(D/'native-runtime/readback-result.json'),
    runner_sha256=sha(H/'run_native_ready123.sh'), launcher_sha256=sha(__file__),
    implementation_repair='Separate successful build/load; v2 single-owner result finalization, explicit attempt/completion counts. Fixture and all numerical budgets unchanged.',
    previous_attempts_retained=['native-result.json','native-env-repair/native-result.json','native-runtime/result.json'],
    process_preflight='Root inspected actual WSL process table after build exit; no prior probe/compiler/trainer. One native synthetic probe only.')
lp(N).mkdir(exist_ok=False)
save(N/'native-plan.json',plan)
cmd=['wsl','-d','Ubuntu-20.04','--','bash',linux(H/'run_native_ready123.sh'),'probe',linux(N)]
start=time.monotonic()
with lp(N/'native-execution.log').open('xb') as log:
    p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
save(N/'native-exit.json',dict(returncode=p.returncode,seconds=time.monotonic()-start,command=cmd))
result_path=N/'native-result.json'
if not lp(result_path).exists():
    # The wsl command exited. Record an actual process snapshot; never assume all descendants exited.
    receipt=N/'native-timeout.json'
    if lp(receipt).exists():
        r=read(receipt)
        assert r['plan_sha256']==sha(N/'native-plan.json') and not r['passed']
        r.update(final_result_owner='parent_after_child_exit',returncode=p.returncode,
                 checks=[], checks_complete=False, timeout_receipt_sha256=sha(receipt))
    else:
        r=dict(passed=False,error='ProcessExitedWithoutResult',returncode=p.returncode,
               final_result_owner='parent_after_child_exit',checks=[],checks_complete=False,
               plan_sha256=sha(N/'native-plan.json'),training_admitted=False,synthetic_only=True)
    save(result_path,r)
r=read(result_path)
snapshot=subprocess.run(['wsl','-d','Ubuntu-20.04','--','ps','-eo','pid,ppid,etime,args'],capture_output=True)
lp(N/'processes-after.txt').write_bytes(snapshot.stdout)
text=snapshot.stdout.decode('utf-8',errors='replace')
residual=[line for line in text.splitlines() if any(token in line for token in
    ['probe_clothing_transport123_v2.py','/ninja','/nvcc','cc1plus','prepare_transport_runtime123.py'])]
accepted=bool(r['passed'] and p.returncode==0 and snapshot.returncode==0 and not residual)
save(N/'acceptance.json',dict(passed=accepted,child_result_passed=r['passed'],returncode=p.returncode,
    process_snapshot_returncode=snapshot.returncode,residual_processes=residual,
    child_result_sha256=sha(result_path),training_admitted=False,synthetic_only=True))
print(json.dumps(dict(accepted=accepted,**{k:r.get(k) for k in ['passed','message','seconds','gpu_seconds','forward_attempted_count','forward_completed_count','backward_count','checks']}),indent=2))
raise SystemExit(p.returncode if p.returncode else (0 if accepted else 1))
