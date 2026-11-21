"""Stop only the verified idle round123 supervisor at grayT's review boundary."""
from pathlib import Path
from datetime import datetime,timezone
import os,signal,json,time
REC=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
ROOT=Path('/opt/gs-slam-lab/shuolee-local-surface-123')
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');q.replace(p)
q=read(REC/'queue-state.json');p=read(REC/'grayT-state.json')
assert q['active_subject']=='grayT' and q['current_stage']=='surface_review_pending'
assert p['training_running'] is False
assert not (ROOT/'grayT/reviews/surface-decision.json').exists()
assert not (ROOT/'grayT/gaussian-run').exists()
pid=q['worker_pid'];cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
assert 'shuolee-slide-update/local_surface123.py --run' in cmd
children=Path(f'/proc/{pid}/task/{pid}/children').read_text().strip()
assert not children,children
evidence=dict(updated=now(),reason='Latest user: smpl-x对不齐; nativeface residuals and fixed-camera relaxation fail alignment. Stop at existing idle boundary, not during optimization.',
    pid=pid,command=cmd,children=children,previous_queue=q,previous_subject_state=p,
    preserved_completed_subjects=['winterhat','whiteT'],grayT_gaussian_training_started=False,
    goal_stays_active=True,source_modified=False,original_fit_decision_preserved=True)
path=REC/'alignment-review-supervisor-stop.json'
assert not path.exists()
save(path,evidence)
os.kill(pid,signal.SIGTERM)
for _ in range(40):
    if not Path(f'/proc/{pid}').exists():break
    time.sleep(.25)
else:raise RuntimeError('Supervisor still exists; no further action or queue mutation.')
evidence.update(stopped_verified=now(),process_alive=False);save(path,evidence)
err='SMPL-X明显未对齐；冻结相机的独立三维点诊断仍有大残差。停在表面审阅节点，灰衣高斯训练尚未开始；目标继续排查。'
p.update(updated=now(),stage='failed_preserved',training_running=False,optimizer_waiting=False,pid=None,error=err,diagnostic='smplx-mapping-audit/summary.json')
save(REC/'grayT-state.json',p)
q.update(updated=now(),state='active_alignment_diagnosis',current_stage='failed_preserved',worker_pid=None,worker_command=None,
    reason=err,training_running=False,goal_stays_active=True)
for s in q['subjects']:
    if s['id']=='grayT':s.update(stage='failed_preserved',updated=now())
save(REC/'queue-state.json',q)
print(json.dumps(dict(stopped_pid=pid,process_alive=False,grayT_training_started=False,goal_active=True),ensure_ascii=False))
