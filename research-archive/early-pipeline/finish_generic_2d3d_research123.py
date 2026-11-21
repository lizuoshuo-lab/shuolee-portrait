"""Commit the visually reviewed last slide and record actual diagnostic DOM review."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';O=REC/'generic-2d3d-research'
G=T/'shuolee/current/robust-portrait-122/video10-global-initializer'
SYNC=Path('D:/00/工作/查看器');LOG=Path('D:/00/工作日志/2026-10-29/人物对齐核验123')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
receipt=read(H/'generic-2d3d123-deck-receipt.json');prior=read(H/'indoor-semantic-all123-deck-receipt.json')
source=Path(receipt['source']);final=Path(receipt['finalPath'])
assert not receipt['canonical_replaced'] and sha(source)==receipt['source_sha256'] and sha(final)==receipt['final_sha256']
assert sha(Path(prior['finalPath']))==receipt['source_sha256']
assert receipt['changed_slides']==[52] and len(receipt['renders'])==52
assert read(H/'generic-2d3d123-exact-package.json')['other_members_byte_identical']
assert (H/'generic-2d3d123-final-validation.json').is_file()
dom=(O/'viewer-dom.txt').read_text(encoding='utf-8')
for text in ['通用2D—SMPL-X工程','主入口与新候选分离','SMPLify-X','尚未执行的适应性测试','实际代码哈希/行号']:
    assert text in dom,text
stamp=datetime.now(timezone.utc).isoformat()
mirror=T/'shuolee/reports/02_有限视角重建优化路线.pptx'
shutil.copy2(final,source);shutil.copy2(final,mirror)
assert sha(source)==sha(mirror)==receipt['final_sha256']
receipt.update(canonical_replaced=True,canonical_path=str(source),mirror_path=str(mirror),actual_last_page_visual_review=True,updated=stamp)
save(H/'generic-2d3d123-deck-receipt.json',receipt)
ui=dict(updated=stamp,url='http://127.0.0.1:8870/diagnostics/video10-global-initializer/index.html#generic-2d3d-research',
    existing_tab=32,reloaded=True,DOM_verified=True,DOM_sha256=sha(O/'viewer-dom.txt'),
    visible_content=['method comparison','local source findings','adaptation phases','upstream links'],
    screenshot_verified=False,screenshot_limitation='Previously established zero-width screenshot failure; no bypass or repeated capture.',
    research_page_only=True,new_model_loaded=False,cloud_sync_verified=False,company00_verified=False)
save(O/'viewer-ui-review.json',ui)
p=O/'publication.json';q=read(p);q.update(updated=stamp,diagnostic_DOM_verified=True,canonical_PPT_updated=True,
    PPT_path=str(source),PPT_sha256=receipt['final_sha256'],PPT_changed_slides=[52],other_51_preserved=True);save(p,q)
p=REC/'next-action.json';q=read(p);q.update(actual_UI_verified=True,publication_next=None,latest_research_ui=str(O/'viewer-ui-review.json'));save(p,q)
for dest in [G,T/'viewer/diagnostics/video10-global-initializer',SYNC/'diagnostics/video10-global-initializer',LOG/'相机多解与人物对齐']:
    for name in ['publication.json','viewer-ui-review.json']:
        shutil.copy2(O/name,dest/'generic-2d3d-research'/name)
print(json.dumps(dict(research_published=True,viewer_DOM_verified=True,canonical_PPT_updated=True,other51_preserved=True,adaptation_executed=False,training_started=False),ensure_ascii=False))
