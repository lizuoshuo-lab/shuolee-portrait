"""Execute the two already registered CLI roundtrips, sequentially."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys,time,traceback
import numpy as np
H=Path(__file__).parent
G=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/gsplat')
O=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/native-renderer-v2')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
result=dict(registered_plan_sha256=sha(O/'plan.json'),runner_sha256=sha(Path(__file__)),jobs=[],state='running')
if (O/'cli-progress.json').exists():raise RuntimeError('CLI check already started; inspect records')
assert read(O/'result.json')['all_driver_controls_passed']
plan=read(O/'plan.json')
for name,digest in plan['source_hashes'].items():assert sha(H/name)==digest,name
save(O/'cli-progress.json',dict(state='running',pid=os.getpid()))
try:
    jobs=read(O/'cli-jobs.json')['jobs'];assert len(jobs)==2
    for job in jobs:
        busy=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True,capture_output=True,check=True).stdout.strip()
        if busy:raise RuntimeError('Existing GPU task:'+busy)
        assert not Path(job['output']).exists()
        entry=H/'native_scene_package122_v2.py'
        code='import sys,runpy;sys.path[:0]=sys.argv[1:3];entry=sys.argv[3];sys.argv=sys.argv[3:];runpy.run_path(entry,run_name="__main__")'
        cmd=[sys.executable,'-s','-c',code,str(H),str(G),str(entry),'--scene',job['scene'],'--camera',job['camera'],'--projection',job['projection'],'--output',job['output']]
        start=time.monotonic()
        with (O/('cli-'+job['projection']+'.log')).open('x') as log:
            process=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=90)
        if process.returncode:raise RuntimeError('CLI failed '+job['projection']+' code '+str(process.returncode))
        actual=np.load(Path(job['output'])/'native-rgb-alpha.npz');expected=np.load(job['reference'])
        err={key:float(np.abs(actual[key]-expected[key]).max()) for key in ['rgb','alpha']}
        passed=max(err.values())<plan['limits']['small_roundtrip_max_abs']
        result['jobs'].append(dict(projection=job['projection'],camera=job['camera'],passed=passed,max_abs=err,seconds=time.monotonic()-start,command=cmd,
            image_sha256=sha(Path(job['output'])/'rgb.png'),receipt=read(Path(job['output'])/'render.json')))
        save(O/'cli-result.json',result)
        if not passed:raise AssertionError('CLI parity '+job['projection'])
        print(job['projection']+' CLI passed '+str(err),flush=True)
    result.update(state='complete',all_controls_passed=True,updated=datetime.now(timezone.utc).isoformat(),interactive_UI_tested=False)
except BaseException as error:
    result.update(state='failed',error=repr(error),traceback=traceback.format_exc());raise
finally:
    save(O/'cli-result.json',result);save(O/'cli-progress.json',dict(state=result['state'],pid=None))
