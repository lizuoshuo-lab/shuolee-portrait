from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,os
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
D=T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route/free-clothing-transport/full-sequence-fit'
O=D/'runtime-repair'
def lp(p):return Path('\\\\?\\'+str(Path(p).resolve()))
def read(p):return json.loads(lp(p).read_text(encoding='utf-8-sig'))
def save(p,j):lp(p).write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(lp(p).read_bytes()).hexdigest()
def wsl(p):
    s=str(Path(p).resolve()).replace('\\','/');return '/mnt/'+s[0].lower()+s[2:]
def now():return datetime.now(timezone.utc).isoformat()
assert read(D/'exit.json')['returncode']==1 and not lp(D/'initial.npz').exists()
assert "unexpected keyword argument 'v_shaped'" in read(D/'failure.json')['traceback']
assert not lp(O).exists(),'No repeated runtime repair.'
lp(O).mkdir()
plan=read(D/'plan.json');plan['output']=wsl(O)
plan.update(registered=now(),runtime_repair=dict(prior_plan_sha256=sha(D/'plan.json'),failure_sha256=sha(D/'failure.json'),
    prior_runtime='/opt/gs-slam-lab/dynamic-human/.venv/bin/python',
    repaired_runtime='/opt/gs-slam-lab/shuolee-portrait-clean-001/env/bin/python',
    cause='Installed drivestudio smplx fork body_models.py calls SMPLXOutput(v_shaped=...) but its dataclass supports v_posed instead. Standard existing smplx supports v_shaped.',
    source_changed=False,inputs_changed=False,objectives_changed=False,prior_fitting_steps=0,new_scientific_candidate=False,
    no_monkey_patch=True,no_install_or_download=True))
save(O/'plan.json',plan)
cmd=['wsl.exe','-d','Ubuntu-20.04','--','/usr/bin/env','CUDA_VISIBLE_DEVICES=0','OMP_NUM_THREADS=4','PYTHONUNBUFFERED=1',
    'timeout',str(plan['budget']['external_timeout_seconds']),plan['runtime_repair']['repaired_runtime'],'-B',wsl(H/'fit_full_sequence123.py'),'--plan',wsl(O/'plan.json')]
save(O/'launch.json',dict(started=now(),windows_pid=os.getpid(),command=cmd,plan_sha256=sha(O/'plan.json')))
with lp(O/'run.log').open('w',encoding='utf-8') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
save(O/'exit.json',dict(finished=now(),returncode=r.returncode,result_exists=lp(O/'result.json').exists()))
print(json.dumps(dict(returncode=r.returncode,result_exists=lp(O/'result.json').exists())))
raise SystemExit(r.returncode)
