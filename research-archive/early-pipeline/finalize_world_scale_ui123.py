from pathlib import Path
from datetime import datetime, timezone
import json, shutil, hashlib
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
S=T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route/training-interface-audit/saved-snapshot'
C=Path(__file__).parent.parent/'shuolee-coordination'
def lp(p):return Path('\\\\?\\'+str(Path(p).resolve()))
def read(p):return json.loads(lp(p).read_text(encoding='utf-8-sig'))
def save(p,v):lp(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
stamp=datetime.now(timezone.utc).isoformat()
ui=dict(updated=stamp,tab_id='32',url='http://127.0.0.1:8870/diagnostics/video10-global-initializer/index.html#world-scale-contract',
    actual_reload=True,AX_verified=True,
    observed=['new world-scale-contract heading','three real frame IDs and complete image decoding','known synthetic scale separated from real scale','19 CPU checks and checkpoint provenance','00 branch 3750cd7 and Windows29/WSL114/CI feedback','no actual new fitting or GS'],
    screenshot_verified=False,screenshot_error='Cannot take screenshot with 0 width.',
    layout_or_rendering_accepted=False,screenshot_workaround_attempted=False,historical_tab15_touched=False,
    new_PLY=False,new_model_quality_accepted=False,WPS_cloud_sync_verified=False)
save(S/'viewer-ui-review.json',ui)
p=S/'publication.json';v=read(p);v.update(viewer_UI_pending=False,viewer_AX_verified=True,viewer_visual_layout_verified=False,viewer_review=str(S/'viewer-ui-review.json'),
    world_unit_increment_handoff_sent=True,world_unit_increment_handoff_state='00_active_deployment_pending',
    handoff_thread_id='01a17c8c-e70f-7822-ad39-440290b3d1da',updated=stamp);save(p,v)
for base in [T/'shuolee/current/robust-portrait-122/video10-global-initializer',T/'viewer/diagnostics/video10-global-initializer',Path('D:/00/工作/查看器/diagnostics/video10-global-initializer')]:
    for name in ['viewer-ui-review.json','publication.json']:
        shutil.copy2(lp(S/name),lp(base/'training-interface-audit/saved-snapshot'/name))
p=C/'local-research-status.json';v=read(p);v.update(updated_at=stamp,latest_publication=str(S/'publication.json'),latest_viewer_review=str(S/'viewer-ui-review.json'))
v['world_unit_handoff'].update(state='sent_00_active_deployment_pending',target_thread_id='01a17c8c-e70f-7822-ad39-440290b3d1da',sent=True)
save(p,v)
print(json.dumps(dict(AX_verified=True,screenshot_verified=False,world_unit_patch_sent=True,goal_complete=False)))
