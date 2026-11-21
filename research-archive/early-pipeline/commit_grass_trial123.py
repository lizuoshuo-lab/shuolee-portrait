from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
O=T/'shuolee/current/local-clothed-surface-123/grass-exploratory'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat()
r=read(H/'grass-trial123-deck-receipt.json')
assert not r['canonical_replaced'] and r['changed_slides']==[52]
assert all(not x['changed'] for x in r['renders'][:51])
src=Path(r['source']);final=Path(r['finalPath'])
assert sha(src)==r['source_sha256'] and sha(final)==r['final_sha256']
shutil.copy2(final,src);shutil.copy2(final,T/'shuolee/reports'/src.name)
r.update(canonical_replaced=True,updated=stamp,actual_last_page_visual_review=True)
save(H/'grass-trial123-deck-receipt.json',r)
dom=(O/'viewer-dom.txt').read_text(encoding='utf-8');gallery=(O/'viewer-gallery-dom.txt').read_text(encoding='utf-8')
assert '实际高斯训练步数： 0' in dom and '50396条' in dom and '59.97px' in dom
assert '(expanded)' in gallery and all(f in gallery for f in ['000001','000019','000038','000056','000075','000094'])
ui=dict(updated=stamp,url='http://127.0.0.1:8870/diagnostics/video10-global-initializer/index.html#grass-exploratory-training',
    actual_UI_loaded=True,reloaded=True,tab='32',gallery_expanded=True,tab15_untouched=True,
    DOM_sha256=sha(O/'viewer-dom.txt'),gallery_DOM_sha256=sha(O/'viewer-gallery-dom.txt'),
    source_images_reviewed_separately=True,rendered_screenshot_verified=False,
    screenshot_limitation='Previously established zero-width browser screenshot failure; not bypassed.',
    new_gaussian_model_published=False,old_model_preserved=True,company00_UI_verified=False,cloud_sync_verified=False)
save(O/'viewer-ui-review.json',ui)
p=O/'publication.json';q=read(p);q.update(actual_UI_review_pending=False,actual_UI_review=str(O/'viewer-ui-review.json'),
    deck=str(src),deck_sha256=sha(src),deck_only_slide52_changed=True,goal_complete=False);save(p,q)
p=H.parent/'shuolee-coordination/local-research-status.json';q=read(p)
q.update(updated_at=stamp,current_stage='grass_exploratory_stopped_geometry_incompatible',training_started=False,
    training_running=False,gaussian_steps=0,current_worker_pid_wsl=None,
    latest_evidence=str(O/'result.json'),latest_visual_review=str(O/'fit-visual-review.json'),
    next_action=read(O/'result.json')['next'],active_goal_remains=True)
save(p,q)
for d in [T/'shuolee/current/robust-portrait-122/video10-global-initializer/grass-exploratory',
          T/'viewer/diagnostics/video10-global-initializer/grass-exploratory',
          Path('D:/00/工作/查看器/diagnostics/video10-global-initializer/grass-exploratory'),
          Path('D:/00/工作日志/2026-10-29/人物对齐核验123/相机多解与人物对齐/grass-exploratory')]:
    for name in ['publication.json','viewer-ui-review.json']:shutil.copy2(O/name,d/name)
print(json.dumps(dict(gaussian_steps=0,UI_checked=True,deck_updated=True,goal_complete=False),ensure_ascii=False))
