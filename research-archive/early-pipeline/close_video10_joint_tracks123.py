from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001');R=T/'shuolee/current/robust-portrait-122';G=R/'video10-global-initializer';O=G/'joint-track-candidate';REC=R.parent/'local-clothed-surface-123'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
stamp=datetime.now(timezone.utc).isoformat();deck=read(H/'video10-joint-tracks123-deck-receipt.json')
assert deck['canonical_replaced'] and deck['actual_last_page_visual_review'] and len(deck['renders'])==52
assert all(sha(Path(deck[k]))==deck['final_sha256'] for k in ['canonical_path','mirror_path'])
v=(O/'browser-viewer.txt').read_text(encoding='utf-8');g=(O/'browser-gallery.txt').read_text(encoding='utf-8');p=(O/'browser-report.txt').read_text(encoding='utf-8')
assert all(s in v for s in ['最新诊断 · 123 · 共享三维点联合求解仍未通过','309,523','93 个相机','质量尚未通过验收'])
assert all(s in g for s in ['128/161','414.54px','1126×677','鼻嘴投影落在脸颊'])
assert '共享三维点联合求解：人物对齐仍未通过' in p
ui=dict(updated=stamp,report_tab='32',viewer_tab='25',actual_report_DOM=True,actual_gallery_DOM=True,actual_model_metadata=True,
    screenshot_review=False,limitation='Previously recorded zero-width browser screenshot failure; not retried or bypassed. Actual native crops and final slide pixels reviewed separately. No new point-cloud render quality claim.',
    historical_model='local-surface-123-whiteT-s18000',gaussians=309523,cameras=93,model_changed=False,quality_accepted=False,tab15_untouched=True,company00_deployed=False,WPS_cloud_roundtrip_verified=False,
    source_sha256={n:sha(O/n) for n in ['browser-report.txt','browser-gallery.txt','browser-viewer.txt']})
save(O/'viewer-ui-review.json',ui);save(O/'ppt-receipt.json',deck)
pub=read(O/'publication.json');pub.update(updated=stamp,viewer_UI_checked=True,viewer_pixels_reviewed=False,PPT_updated=True,PPT_sha256=deck['final_sha256'],first51_slides_unchanged=True,actual_native_crop_review=True,tab15_untouched=True)
save(O/'publication.json',pub);save(G/'publication.json',dict(**pub,latest_stage=str(O/'publication.json')))
for root in [R,REC]:
    q=read(root/'queue-state.json');q['updated']=stamp;q['last_goal_turn'].update(publication=str(O/'publication.json'));q.update(running=None,training_running=False,local_GPU_task_active=False);save(root/'queue-state.json',q)
p=H.parent/'shuolee-coordination/local-research-status.json';x=read(p);x['updated_at']=stamp;x['current_video10_joint_candidate'].update(publication=str(O/'publication.json'),actual_viewer_DOM_verified=True,PPT_sha256=deck['final_sha256'],company00_deployed=False);save(p,x)
for dest in [T/'viewer/diagnostics/video10-global-initializer',Path('D:/00/工作/查看器/diagnostics/video10-global-initializer'),Path('D:/00/工作日志/2026-10-29/人物对齐核验123/相机多解与人物对齐')]:
    shutil.copy2(G/'publication.json',dest/'publication.json')
    for n in ['publication.json','viewer-ui-review.json','ppt-receipt.json']:shutil.copy2(O/n,dest/'joint-track-candidate'/n)
print(json.dumps(dict(publication=str(O/'publication.json'),PPT_sha256=deck['final_sha256'],UI_DOM_verified=True,model_quality_accepted=False,goal_complete=False),ensure_ascii=False))
