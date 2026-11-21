from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/local-clothed-surface-123';C=H.parent/'shuolee-coordination';L=Path('D:/00/工作日志/2026-10-28/人物自由表面训练123')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat();receipt=read(H/'surface123-articulation-deck-receipt.json')
assert receipt['canonical_replaced'] and receipt['actual_last_page_visual_review']
main=(R/'facial-articulation-main-viewer-ax.txt').read_text(encoding='utf-8');report=(R/'facial-articulation-report-ax.txt').read_text(encoding='utf-8')
assert '表情补充未修复跨视角错位' in main and '309,523' in main and '93 个相机' in main
assert '共享表情' in report and 'frame_000080' in report
ui=dict(updated=stamp,report_reload_and_gallery_expand_verified=True,main_viewer_reload_verified=True,
    current_model_preserved='local-surface-123-whiteT-s18000',points=309523,cameras=93,actual_browser_pixel_screenshot=False,
    screenshot_limitation='Existing hidden browser surface has zero width; no repeated screenshot or security bypass. Open existing tab was queued by app.',
    diagnostic_images_actually_reviewed_from_files=True,main_native_render_acceptance=False,tab15_untouched=True,
    proof=[dict(path=str(p),sha256=sha(p)) for p in [R/'facial-articulation-main-viewer-ax.txt',R/'facial-articulation-report-ax.txt']])
save(R/'facial-articulation-ui-review.json',ui)
d=read(R/'facial-articulation-completion.json');d.update(updated=stamp,state='control_and_cross_run_audit_published',
    diagnostic_DOM_verified=True,browser_pixel_review=False,PPT_updated=True,PPT_sha256=receipt['final_sha256'],
    first51_slides_unchanged=True,company00_handoff_sent=True,goal_active=True,quality_accepted=False,training_started=False)
save(R/'facial-articulation-completion.json',d)
n=read(R/'next-action.json');n.update(updated=stamp,diagnostic_DOM_verified=True,actual_UI_verified=False);save(R/'next-action.json',n)
q=read(C/'local-research-status.json');q.update(updated_at=stamp,current_local_facial_articulation_publication=d,
    latest_local_handoff=dict(updated=stamp,thread='01a17c8c-e70f-7822-ad39-440290b3d1da',evidence=str(R/'facial-articulation-conclusion.json'),reply_required=False));save(C/'local-research-status.json',q)
for name in ['facial-articulation-conclusion.json','facial-articulation-completion.json','facial-articulation-ui-review.json','next-action.json']:
    shutil.copy2(R/name,L/name)
shutil.copy2(H/'surface123-articulation-final-52.png',L/'报告末页_表情与跨轮核查.png')
print(json.dumps(d,ensure_ascii=False))
