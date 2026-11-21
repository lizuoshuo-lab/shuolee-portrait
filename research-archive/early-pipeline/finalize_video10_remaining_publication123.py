from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
U=T/'shuolee/current/robust-portrait-122/video10-remaining-material-review';C=H.parent/'shuolee-coordination'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat();rec=read(H/'video10-global-preflight123-deck-receipt.json')
assert rec['canonical_replaced'] and rec['actual_last_page_visual_review'] and len(rec['renders'])==52
assert all(not x['changed'] for x in rec['renders'][:51])
assert read(H/'video10-global-preflight123-exact-package.json')['other_members_byte_identical']
for k in ['canonical_path','mirror_path']:assert sha(Path(rec[k]))==rec['final_sha256']
viewer=(U/'browser-viewer-loaded.txt').read_text(encoding='utf-8')
assert all(s in viewer for s in ['smplx123-video10-remaining-support','309,523','93 个相机','质量尚未通过验收','尚非全局初始化'])
assert '全局初始化接口核验' in (U/'browser-gallery.txt').read_text(encoding='utf-8')
assert 'SMPL-X对齐：材料审阅' in (U/'browser-report.txt').read_text(encoding='utf-8')
assert '16组原生扩大上下文' in (U/'browser-contexts.txt').read_text(encoding='utf-8')
ui=dict(updated=stamp,verification='Actual existing local browser AX DOM and loaded model metadata',viewer_tab='25',report_tab='32',latest_diagnostic='smplx123-video10-remaining-support',
    actual_report_DOM=True,actual_gallery_DOM=True,actual_context_gallery_DOM=True,actual_viewer_loaded_DOM=True,
    historical_model='local-surface-123-whiteT-s18000',gaussians=309523,cameras=93,
    browser_pixel_render_review=False,limitation='Previously recorded zero-width browser screenshot failure. DOM confirms page state only; it is not a new model quality review.',
    actual_native_rows_reviewed=222,native_observations_reviewed=696,expanded_contexts_reviewed=16,
    tab15_untouched=True,new_model=False,quality_accepted=False,company00_deployed=False,WPS_cloud_roundtrip_verified=False,
    evidence={f:sha(U/f) for f in ['browser-report.txt','browser-gallery.txt','browser-contexts.txt','browser-viewer-loaded.txt']})
save(U/'viewer-ui-review.json',ui);save(U/'ppt-receipt.json',rec)
p=read(U/'publication.json');p.update(updated=stamp,browser_DOM_verified=True,browser_pixel_render_review=False,PPT_pending=False,
    PPT_canonical_updated=True,PPT_sha256=rec['final_sha256'],PPT_receipt=str(H/'video10-global-preflight123-deck-receipt.json'),first51_preserved=True,
    ui_receipt=str(U/'viewer-ui-review.json'),tab15_untouched=True,company00_deployed=False,WPS_cloud_roundtrip_verified=False)
save(U/'publication.json',p)
c=read(C/'local-research-status.json');c['updated_at']=stamp
c['current_video10_remaining_material'].update(PPT_canonical_updated=True,PPT_sha256=rec['final_sha256'],browser_DOM_verified=True,first51_preserved=True,actual_model_quality_accepted=False,publication=str(U/'publication.json'))
save(C/'local-research-status.json',c)
for dest in [T/'viewer/diagnostics/video10-remaining-material-review',Path('D:/00/工作/查看器/diagnostics/video10-remaining-material-review'),Path('D:/00/工作日志/2026-10-29/人物对齐核验123/视频10材料与相机初始化核验')]:
    for f in ['publication.json','viewer-ui-review.json','ppt-receipt.json']:shutil.copy2(U/f,dest/f)
print(json.dumps(p,ensure_ascii=False))
