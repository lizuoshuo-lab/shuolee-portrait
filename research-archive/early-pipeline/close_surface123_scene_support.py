from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/local-clothed-surface-123';C=H.parent/'shuolee-coordination'
L=Path('D:/00/工作日志/2026-10-29/人物对齐核验123')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat()
receipt=read(H/'surface123-scene-support-deck-receipt.json')
assert receipt['canonical_replaced'] and receipt['actual_last_page_visual_review']
main=(R/'scene-support-main-viewer-ax.txt').read_text(encoding='utf-8')
report=(R/'scene-support-report-ax.txt').read_text(encoding='utf-8')
assert '头部附近场景验证仍不足' in main and '309,523' in main and '93 个相机' in main
assert '全部节点进入BA' in report and 'frame_000054 / frame_000058 / frame_000061' in report
ui=dict(updated=stamp,report_reload_and_gallery_expand_verified=True,main_viewer_reload_verified=True,
 current_model_preserved='local-surface-123-whiteT-s18000',points=309523,cameras=93,
 actual_browser_pixel_screenshot=False,diagnostic_images_actually_reviewed_from_files=True,
 screenshot_limitation='Existing browser surface previously reported zero width; not retried and no lower-level workaround.',
 main_native_render_acceptance=False,tab15_untouched=True,
 evidence=[dict(path=str(p),sha256=sha(p)) for p in [R/'scene-support-main-viewer-ax.txt',R/'scene-support-report-ax.txt']])
save(R/'scene-support-cross-run-ui-review.json',ui)
d=read(R/'scene-support-cross-run-completion.json');d.update(updated=stamp,state='read_only_scene_support_audit_published',
 diagnostic_DOM_verified=True,browser_pixel_review=False,PPT_updated=True,PPT_sha256=receipt['final_sha256'],
 first51_slides_unchanged=True,company00_handoff_sent=True,goal_active=True,quality_accepted=False,training_started=False)
save(R/'scene-support-cross-run-completion.json',d)
n=read(R/'next-action.json');n.update(updated=stamp,diagnostic_DOM_verified=True,actual_UI_verified=False)
save(R/'next-action.json',n)
p=read(R/'publication.json');p.update(updated=stamp,report='http://127.0.0.1:8870/body-smplx.html#scene-support-cross-run',
 diagnostic_DOM_verified=True,UI_checked=False,quality_accepted=False);save(R/'publication.json',p)
c=read(C/'local-research-status.json');c.update(updated_at=stamp,current_local_scene_support_publication=d,
 latest_local_handoff=dict(updated=stamp,thread='01a17c8c-e70f-7822-ad39-440290b3d1da',
 evidence=str(R/'scene-support-cross-run-conclusion.json'),reply_required=False))
save(C/'local-research-status.json',c)
for name in ['scene-support-cross-run-conclusion.json','scene-support-cross-run-completion.json','scene-support-cross-run-ui-review.json','next-action.json']:
 shutil.copy2(R/name,L/name)
shutil.copy2(H/'surface123-scene-support-final-52.png',L/'报告末页_场景支持核查.png')
print(json.dumps(d,ensure_ascii=False))
