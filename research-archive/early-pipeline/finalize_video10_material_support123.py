"""Close publication checks without claiming a reconstruction acceptance."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';N=R/'video10-semantic-role'
C=H.parent/'shuolee-coordination'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat()
receipt=read(H/'video10-support123-deck-receipt.json')
assert receipt['canonical_replaced'] and receipt['actual_last_page_visual_review']
assert len(receipt['renders'])==52 and all(not x['changed'] for x in receipt['renders'][:51])
for k in ['canonical_path','mirror_path']:assert sha(Path(receipt[k]))==receipt['final_sha256']
viewer=(N/'browser-material-viewer-loaded.txt').read_text(encoding='utf-8')
gallery=(N/'browser-material-gallery.txt').read_text(encoding='utf-8')
report=(N/'browser-material-report.txt').read_text(encoding='utf-8')
assert all(t in viewer for t in ['smplx123-video10-material-support','309,523','93 个相机','质量尚未通过验收'])
assert '材料点增加，整圈相机连接仍不足' in gallery
assert '背景对应还没有连成整圈' in report
ui=dict(updated=stamp,verification='Actual existing local browser AX DOM and loading state',
    viewer_tab='25',report_tab='32',latest_diagnostic='smplx123-video10-material-support',
    actual_report_DOM=True,actual_gallery_DOM=True,actual_viewer_loaded_DOM=True,
    historical_model='local-surface-123-whiteT-s18000',gaussians=309523,cameras=93,
    browser_pixel_render_review=False,limitation='Current browser screenshot capture has a previously recorded zero-width failure; DOM is not a fresh image-quality review.',
    source_material_images_reviewed=113,expanded_contexts_reviewed=16,tab15_untouched=True,
    new_model=False,quality_accepted=False,company00_deployed=False,WPS_cloud_roundtrip_verified=False,
    evidence={f:sha(N/f) for f in ['browser-material-report.txt','browser-material-gallery.txt','browser-material-viewer-loaded.txt']})
save(N/'material-viewer-ui-review.json',ui)
pub=read(N/'material-publication.json')
pub.update(updated=stamp,browser_DOM_verified=True,browser_pixel_render_review=False,PPT_pending=False,
    PPT_canonical_updated=True,PPT_sha256=receipt['final_sha256'],PPT_receipt=str(H/'video10-support123-deck-receipt.json'),
    first51_preserved=True,ui_receipt=str(N/'material-viewer-ui-review.json'),tab15_untouched=True,
    company00_deployed=False,WPS_cloud_roundtrip_verified=False)
save(N/'material-publication.json',pub)
save(N/'material-ppt-receipt.json',receipt)
c=read(C/'local-research-status.json');c['updated_at']=stamp
c['current_video10_semantic_role'].update(publication=str(N/'material-publication.json'),PPT_canonical_updated=True,
    PPT_sha256=receipt['final_sha256'],browser_DOM_verified=True,first51_preserved=True,actual_model_quality_accepted=False)
save(C/'local-research-status.json',c)
for dest in [Path('D:/00/工作日志/2026-10-29/人物对齐核验123/视频10像素语义核验'),T/'viewer/diagnostics/video10-semantic-role',Path('D:/00/工作/查看器/diagnostics/video10-semantic-role')]:
    for name in ['material-publication.json','material-viewer-ui-review.json','material-ppt-receipt.json']:
        shutil.copy2(N/name,dest/name)
print(json.dumps(pub,ensure_ascii=False))
