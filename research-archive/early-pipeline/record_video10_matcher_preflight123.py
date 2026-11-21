from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';Q=R/'video10-correspondence-contrast'
C=H.parent/'shuolee-coordination'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
stamp=datetime.now(timezone.utc).isoformat();pre=read(Q/'preflight.json')
assert pre['state']=='preflight_passed_no_inference' and not (Q/'run').exists()
assert hashlib.sha256((Q/'plan.json').read_bytes()).hexdigest()==pre['plan_sha256']
save(Q/'progress.json',dict(updated=stamp,state='preflight_complete_runner_pending',pid=None,pairs=7,review_queries=72,
    new_inference_started=False,training_started=False,preflight=str(Q/'preflight.json'),
    next='Implement the registered seven-pair native-coordinate comparison runner. No source detection, cameras or training.',
    source_transport_repair='Pinned official archive has no .git; compare all73 archived source files. No model source changed.'))
save(Q/'preflight-failure-record.json',dict(updated=stamp,phase='preflight_only',reason='git rev-parse failed because official source was extracted from a pinned archive',
    evidence=str(Q/'preflight-terminal.log'),resolved_with='Exact archive SHA and all73 source file byte comparisons',
    prior_unobserved_launch='No result or active process remained before logged retry; previous outcome unknown.',
    weights_or_inference_executed=False))
for root in [R,T/'shuolee/current/local-clothed-surface-123']:
    q=read(root/'queue-state.json');q['updated']=stamp;q['running']=None
    q['next_plan']=str(Q/'plan.json');q['next_preflight']=str(Q/'preflight.json')
    q['local_GPU_task_active']=False;q['training_running']=False
    q['last_goal_turn']['preflight']='Seven-pair matcher contrast registered; 72 review queries frozen; runtime, weights, source archive and coordinates checked; no inference.'
    save(root/'queue-state.json',q)
c=read(C/'local-research-status.json');c['updated_at']=stamp
c['current_video10_matcher_contrast']=dict(state='preflight_complete_no_inference',plan=str(Q/'plan.json'),preflight=str(Q/'preflight.json'),pairs=7,review_queries=72,new_cameras=False,training_started=False)
save(C/'local-research-status.json',c)
print('Preflight recorded; no GPU task or training started.')
