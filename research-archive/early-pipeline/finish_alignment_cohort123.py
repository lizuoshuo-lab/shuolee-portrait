from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,shutil
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';O=REC/'generic-2d3d-research/adapter-a0b';G=T/'shuolee/current/robust-portrait-122/video10-global-initializer'
SYNC=Path('D:/00/工作/查看器');LOG=Path('D:/00/工作日志/2026-10-29/人物对齐核验123')
P=Path('D:/GS-SLAM-Data/shuolee-portrait-new-capture-118')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
receipt=read(H/'alignment-cohort123-deck-receipt.json');prior=read(H/'alignment-adapter123-v2-deck-receipt.json')
source=Path(receipt['source']);final=Path(receipt['finalPath'])
assert not receipt['canonical_replaced'] and sha(source)==prior['final_sha256']==receipt['source_sha256']
assert sha(final)==receipt['final_sha256'] and len(receipt['renders'])==52 and receipt['changed_slides']==[52]
assert read(H/'alignment-cohort123-exact-package.json')['other_members_byte_identical']
assert (H/'alignment-cohort123-final-validation.json').is_file()
dom=(O/'viewer-dom.txt').read_text(encoding='utf-8')
for text in ['6人753帧、38403条面部观察','旧目标确实已接入6个脚点','未执行人体拟合、相机优化或高斯训练']:
    assert text in dom,text
result=read(O/'result.json');assert result['valid_existing_streams']==6 and not result['rejected']
for name,digest in result['one_source_hashes'].items():assert sha(P/'src/shuolee_portrait'/name)==digest,name
stamp=datetime.now(timezone.utc).isoformat()
shutil.copy2(final,source);shutil.copy2(final,T/'shuolee/reports/02_有限视角重建优化路线.pptx')
assert sha(source)==receipt['final_sha256']
receipt.update(canonical_replaced=True,actual_last_page_visual_review=True,updated=stamp)
save(H/'alignment-cohort123-deck-receipt.json',receipt)
ui=dict(updated=stamp,url='http://127.0.0.1:8870/diagnostics/video10-global-initializer/index.html#generic-2d3d-cohort',
    existing_tab=32,reloaded=True,DOM_verified=True,DOM_sha256=sha(O/'viewer-dom.txt'),
    screenshot_verified=False,screenshot_limitation='Previously established zero-width screenshot failure; no repeated capture or bypass.',
    diagnostic_only=True,new_point_cloud=False,historical_tab15_untouched=True,cloud_sync_verified=False,company00_verified=False)
save(O/'viewer-ui-review.json',ui)
p=O/'publication.json';q=read(p);q.update(updated=stamp,diagnostic_DOM_verified=True,canonical_PPT_updated=True,
    PPT_path=str(source),PPT_sha256=receipt['final_sha256'],PPT_changed_slides=[52],other_51_preserved=True,
    local_source_repository=str(P),production_source_unchanged_this_round=True,Git_pushed=False);save(p,q)
p=REC/'next-action.json';q=read(p);q.update(actual_UI_verified=True,publication_next=None,latest_cohort_ui=str(O/'viewer-ui-review.json'));save(p,q)
for dest in [G,T/'viewer/diagnostics/video10-global-initializer',SYNC/'diagnostics/video10-global-initializer',LOG/'相机多解与人物对齐']:
    for name in ['publication.json','viewer-ui-review.json']:
        shutil.copy2(O/name,dest/'generic-2d3d-research/adapter-a0b'/name)
    shutil.copy2(REC/'generic-2d3d-research/adaptation-plan.json',dest/'generic-2d3d-research/adaptation-plan.json')
print(json.dumps(dict(A0b_complete=True,subjects=7,valid_existing_streams=6,frames=753,actual_DOM_verified=True,
    canonical_PPT_updated=True,real_fit=False,training=False,goal_stays_active=True),ensure_ascii=False))
