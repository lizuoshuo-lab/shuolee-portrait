from pathlib import Path
from datetime import datetime,timezone
import json
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001');R=T/'shuolee/current/robust-portrait-122';Q=R/'video10-correspondence-contrast'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
p=read(Q/'progress.json');stamp=datetime.now(timezone.utc).isoformat()
assert p['pid']==306 and p['state'] in ['loading_existing_official_weights','matching','cycle_consistency']
for root in [R,T/'shuolee/current/local-clothed-surface-123']:
    q=read(root/'queue-state.json');q.update(updated=stamp,current_stage='video10_matcher_contrast',running=dict(pid=p['pid'],runtime='WSL',command='run_video10_matcher_contrast123.py',session_id=39598,verified_live_by='ps and nvidia-smi 2026-10-28T20:06Z'),local_GPU_task_active=True,training_running=False)
    save(root/'queue-state.json',q)
c=read(H.parent/'shuolee-coordination/local-research-status.json');c.update(updated_at=stamp,local_GPU_task_active=True)
c['current_video10_matcher_contrast'].update(state='running',pid=p['pid'],inference_started=True,training_started=False)
save(H.parent/'shuolee-coordination/local-research-status.json',c)
print('Recorded verified single GPU inference process306; not training.')
