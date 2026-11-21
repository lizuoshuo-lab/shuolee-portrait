from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001');R=T/'shuolee/current/robust-portrait-122';N=R/'video10-semantic-role';O=R/'video10-temporal-bridge';C=H.parent/'shuolee-coordination';L=Path('D:/00/工作日志/2026-10-29/人物对齐核验123')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
stamp=datetime.now(timezone.utc).isoformat();receipt=read(H/'video10-semantic123-deck-receipt.json')
assert receipt['canonical_replaced'] and receipt['actual_last_page_visual_review']
assert sha(Path(receipt['canonical_path']))==receipt['final_sha256']==sha(Path(receipt['mirror_path']))
domfiles=['browser-report.txt','browser-gallery.txt','browser-expanded-mask.txt','browser-viewer-loaded.txt']
for f in domfiles:assert (N/f).stat().st_size>100
assert 'smplx123-video10-semantic' in (N/'browser-report.txt').read_text(encoding='utf-8')
vd=(N/'browser-viewer-loaded.txt').read_text(encoding='utf-8');assert 'smplx123-video10-semantic' in vd and '309,523' in vd and '93' in vd
p=read(N/'publication.json');p.update(updated=stamp,browser_DOM_verified=True,dom_evidence={f:sha(N/f) for f in domfiles},historical_model='local-surface-123-whiteT-s18000',historical_points=309523,historical_cameras=93,
    screenshot_verified=False,screenshot_limitation='Prior IAB capture reports zero width; not repeated. This is DOM verification, not new pixel rendering or00 UI acceptance.',PPT_pending=False,PPT_receipt=str(H/'video10-semantic123-deck-receipt.json'),PPT_only_slide52_changed=True,first51_byte_and_render_preserved=True,training_started=False)
save(N/'publication.json',p)
save(N/'viewer-ui-review.json',dict(updated=stamp,actual_report_navigation=True,actual_gallery_navigation=True,failed_mask_section_expanded=True,main_viewer_reloaded=True,latestDiagnostic_visible=True,historical_model_preserved=True,tab15_untouched=True,DOM_only=True,new_point_cloud_loaded=False,new_render_quality_reviewed=False,company00_deployed=False))
save(N/'serialization-repair.json',dict(updated=stamp,first_log_sha256=sha(N/'classification-first-failure.log'),first_source_sha256=sha(N/'classification-first-source.py'),repaired_source_sha256=sha(H/'classify_video10_semantics123.py'),cause='JSON encoder rejected np.int64 diagnostic count',change='int(sum(...)) only',same_inputs_verified=True,mask_generation_repeated=False,feature_extraction_repeated=False,classification_result=str(N/'classification-result.json')))
for dest in [L/'视频10像素语义核验',T/'viewer/diagnostics/video10-semantic-role',Path('D:/00/工作/查看器/diagnostics/video10-semantic-role')]:
    for f in ['publication.json','viewer-ui-review.json','serialization-repair.json']:shutil.copy2(N/f,dest/f)
for dest in [L/'视频10时间桥接核验',T/'viewer/diagnostics/video10-temporal-bridge',Path('D:/00/工作/查看器/diagnostics/video10-temporal-bridge')]:shutil.copy2(O/'publication.json',dest/'publication.json')
c=read(C/'local-research-status.json');c['updated_at']=stamp;c['current_video10_semantic_role'].update(publication=str(N/'publication.json'),PPT_canonical_updated=True,browser_DOM_verified=True,first51_preserved=True);save(C/'local-research-status.json',c)
print(json.dumps(dict(PPT_updated=True,first51_preserved=True,viewer_DOM_verified=True,historical_model_preserved=True,next='actual review of113 prepared native material families',training_started=False),ensure_ascii=False))
