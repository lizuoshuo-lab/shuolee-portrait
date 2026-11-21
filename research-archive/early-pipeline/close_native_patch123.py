from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';REC=R.parent/'local-clothed-surface-123';G=R/'video10-global-initializer'
PA=R/'video10-correspondence-contrast/head-roi-contrast/native-affine-contrast'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
stamp=datetime.now(timezone.utc).isoformat();deck=read(H/'video10-native-patch123-deck-receipt.json')
assert deck['canonical_replaced'] and deck['actual_last_page_visual_review'] and len(deck['renders'])==52
assert all(sha(Path(deck[k]))==deck['final_sha256'] for k in ['canonical_path','mirror_path'])
texts={name:(PA/f'browser-{name}.txt').read_text(encoding='utf-8') for name in ['report','gallery','viewer']}
assert all(x in texts['viewer'] for x in ['309,523','93 个相机','质量尚未通过验收','SMPL-X对齐未通过，局部细化没有改善'])
assert all(x in texts['gallery'] for x in ['2.403px','99/491','43.99px','最大42.07px'])
assert 'SMPL-X对齐：局部细化未改善，旧门槛不能放行' in texts['report']
ui=dict(updated=stamp,actual_report_DOM=True,actual_gallery_DOM=True,actual_model_metadata=True,
    report_tab='32',viewer_tab='25',screenshot_review=False,viewport_loading_label_present=('载入中' in texts['viewer']),new_render_quality_claim=False,
    limitation='Known zero-width browser screenshot failure not retried or bypassed. New diagnosis and historical model metadata verified through actual UI. Loading label remains; no claim of fresh interactive render review. Native saved crops and PPT slide52 were separately visually inspected.',
    model='local-surface-123-whiteT-s18000',model_changed=False,gaussians=309523,cameras=93,quality_accepted=False,tab15_untouched=True,company00_deployed=False,WPS_cloud_roundtrip_verified=False,
    source_sha256={name:sha(PA/f'browser-{name}.txt') for name in texts})
save(PA/'viewer-ui-review.json',ui);save(PA/'ppt-receipt.json',deck)
pub=read(PA/'publication.json');pub.update(updated=stamp,viewer_UI_checked=True,viewer_pixels_reviewed=False,PPT_updated=True,PPT_sha256=deck['final_sha256'],first51_slides_unchanged=True,actual_native_crop_review=True,local_branch_closed=True,pattern_native_input_reviewed=True,tab15_untouched=True)
save(PA/'publication.json',pub);save(G/'publication.json',dict(**pub,latest_stage=str(PA/'publication.json')))
for root in [R,REC]:
    q=read(root/'queue-state.json');q['updated']=stamp;q['last_goal_turn'].update(publication=str(PA/'publication.json'));q.update(running=None,worker_pid=None,training_running=False,local_GPU_task_active=False);save(root/'queue-state.json',q)
path=H.parent/'shuolee-coordination/local-research-status.json';x=read(path);x['updated_at']=stamp;x['current_native_patch_review'].update(publication=str(PA/'publication.json'),actual_viewer_DOM_verified=True,PPT_sha256=deck['final_sha256'],company00_deployed=False);save(path,x)
for dest in [T/'viewer/diagnostics/video10-global-initializer',Path('D:/00/工作/查看器/diagnostics/video10-global-initializer'),Path('D:/00/工作日志/2026-10-29/人物对齐核验123/相机多解与人物对齐')]:
    shutil.copy2(G/'publication.json',dest/'publication.json')
    for n in ['publication.json','viewer-ui-review.json','ppt-receipt.json']:shutil.copy2(PA/n,dest/'head-roi-contrast/native-affine-contrast'/n)
print(json.dumps(dict(publication=str(PA/'publication.json'),PPT_sha256=deck['final_sha256'],goal_complete=False,training_started=False),ensure_ascii=False))
