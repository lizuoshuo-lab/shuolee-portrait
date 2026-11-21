"""Mirror real worker milestones; no training, approval, retry, or scheduled task."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,msvcrt,os,subprocess,sys,time,traceback
HERE=Path(__file__).resolve().parent
REC=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
ROOT=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-local-surface-123')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def now():return datetime.now(timezone.utc).isoformat()
guard=(REC/'publication-worker.lock').open('a+b');guard.seek(0)
if guard.read(1)==b'':guard.write(b'0');guard.flush()
guard.seek(0);msvcrt.locking(guard.fileno(),msvcrt.LK_NBLCK,1)
def record(state,**kw):
    p=REC/'publication-worker.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(updated=now(),pid=os.getpid(),command=sys.argv,state=state,training_started_by_this_process=False,**kw),ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
record('running')
last=None
try:
    while True:
        q=read(REC/'queue-state.json');signature=[q['state'],q['current_stage'],q.get('active_subject')]
        for s in q['subjects']:
            p=ROOT/s['id']/'progress.json';r=read(p) if p.exists() else {}
            signature.append([s['id'],s['stage'],r.get('step',0)//1000,r.get('stage')])
        for p in sorted(REC.glob('*-review/actual-review.json')):signature.append([p.parent.name,p.stat().st_mtime_ns])
        for p in sorted(REC.glob('*-quality-review.json')):signature.append([p.name,p.stat().st_mtime_ns])
        digest=hashlib.sha256(json.dumps(signature).encode()).hexdigest()
        if digest!=last:
            subprocess.run([sys.executable,'-X','utf8',str(HERE/'publish_local_surface123.py')],check=True)
            record('running',last_publication=now(),signature=digest);last=digest
        if q['state']=='worker_finished_review_pending':break
        time.sleep(10)
    record('finished')
except Exception:
    record('failed',error=traceback.format_exc());raise
finally:
    guard.seek(0);msvcrt.locking(guard.fileno(),msvcrt.LK_UNLCK,1);guard.close()
