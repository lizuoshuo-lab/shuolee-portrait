from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001');R=T/'shuolee/current/robust-portrait-122';REC=R.parent/'local-clothed-surface-123';G=R/'video10-global-initializer';CR=G/'joint-track-candidate/convergence-repair'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
stamp=datetime.now(timezone.utc).isoformat();deck=read(H/'video10-convergence123-deck-receipt.json')
assert deck['canonical_replaced'] and deck['actual_last_page_visual_review'] and len(deck['renders'])==52
assert all(sha(Path(deck[k]))==deck['final_sha256'] for k in ['canonical_path','mirror_path'])
v=(CR/'browser-viewer.txt').read_text(encoding='utf-8');g=(CR/'browser-gallery.txt').read_text(encoding='utf-8');p=(CR/'browser-report.txt').read_text(encoding='utf-8')
assert all(x in v for x in ['相机误差下降，SMPL-X对齐仍未通过','309,523','93 个相机','质量尚未通过验收'])
assert all(x in g for x in ['93个仍未达到','42.07px','62降到23','鼻嘴投影落在脸颊'])
assert 'SMPL-X对齐核验：相机误差下降，质量仍未通过' in p
ui=dict(updated=stamp,report_tab='32',viewer_tab='25',actual_report_DOM=True,actual_gallery_DOM=True,actual_model_metadata=True,
    screenshot_review=False,viewport_loading_label_present=('载入中' in v),new_render_quality_claim=False,
    limitation='Known zero-width browser screenshot failure not retried or bypassed. Canvas loading label remained in DOM; only new diagnostic text and old model/camera metadata verified. Native saved image crops and slide52 pixels separately reviewed.',
    historical_model='local-surface-123-whiteT-s18000',gaussians=309523,cameras=93,model_changed=False,quality_accepted=False,tab15_untouched=True,company00_deployed=False,WPS_cloud_roundtrip_verified=False,
    source_sha256={n:sha(CR/n) for n in ['browser-report.txt','browser-gallery.txt','browser-viewer.txt']})
save(CR/'viewer-ui-review.json',ui);save(CR/'ppt-receipt.json',deck)
pub=read(CR/'publication.json');pub.update(updated=stamp,viewer_UI_checked=True,viewer_pixels_reviewed=False,PPT_updated=True,PPT_sha256=deck['final_sha256'],first51_slides_unchanged=True,actual_native_crop_review=True,head_ROI_all24_reviewed=True,tab15_untouched=True)
save(CR/'publication.json',pub);save(G/'publication.json',dict(**pub,latest_stage=str(CR/'publication.json')))
for root in [R,REC]:
    q=read(root/'queue-state.json');q['updated']=stamp;q['last_goal_turn'].update(publication=str(CR/'publication.json'));q.update(running=None,training_running=False,local_GPU_task_active=False);save(root/'queue-state.json',q)
path=H.parent/'shuolee-coordination/local-research-status.json';x=read(path);x['updated_at']=stamp;x['current_video10_convergence_repair'].update(publication=str(CR/'publication.json'),actual_viewer_DOM_verified=True,PPT_sha256=deck['final_sha256'],company00_deployed=False);save(path,x)
for dest in [T/'viewer/diagnostics/video10-global-initializer',Path('D:/00/工作/查看器/diagnostics/video10-global-initializer'),Path('D:/00/工作日志/2026-10-29/人物对齐核验123/相机多解与人物对齐')]:
    shutil.copy2(G/'publication.json',dest/'publication.json')
    for n in ['publication.json','viewer-ui-review.json','ppt-receipt.json']:shutil.copy2(CR/n,dest/'joint-track-candidate/convergence-repair'/n)
print(json.dumps(dict(publication=str(CR/'publication.json'),PPT_sha256=deck['final_sha256'],UI_DOM_verified=True,goal_complete=False),ensure_ascii=False))
