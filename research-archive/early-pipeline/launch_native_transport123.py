from pathlib import Path
import subprocess,json,sys,time
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
D=T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route/free-clothing-transport'
def lp(p):return Path('\\\\?\\'+str(Path(p).resolve()))
def linux(p):return '/mnt/'+str(p)[0].lower()+'/'+str(p)[3:].replace('\\','/')
assert lp(D/'native-plan.json').exists()
assert not lp(D/'native-progress.json').exists() and not lp(D/'native-result.json').exists()
cmd=['wsl','-d','Ubuntu-20.04','--','timeout','--signal=TERM','--kill-after=10s','240s',
     '/opt/gs-slam-lab/dynamic-human/.venv/bin/python','-B',linux(H/'probe_clothing_transport123.py'),'--output-dir',linux(D)]
with lp(D/'native-execution.log').open('xb') as log:
    start=time.monotonic();proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
rc=proc.returncode
lp(D/'native-exit.json').write_text(json.dumps(dict(returncode=rc,seconds=time.monotonic()-start,command=cmd),indent=2),encoding='utf-8')
if lp(D/'native-result.json').exists():
    r=json.loads(lp(D/'native-result.json').read_text(encoding='utf-8'))
    print(json.dumps({k:r.get(k) for k in ['passed','message','seconds','gpu_seconds','forward_count','backward_count','checks']},ensure_ascii=False,indent=2))
else:print('NO_RESULT: inspect native-execution.log and real PID before any further action')
sys.exit(rc)
