from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,zipfile
H=Path(__file__).parent;C=H.parent/'shuolee-coordination'
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122'
D=T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route/free-clothing-transport'
O=D/'native-training-entry'
def lp(p):return Path('\\\\?\\'+str(Path(p).resolve()))
def read(p):return json.loads(lp(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(lp(p).read_bytes()).hexdigest()
def save(p,j):lp(p).write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
stamp=datetime.now(timezone.utc).isoformat()
receipt=read(H/'free-transport123-native-training-deck-receipt.json')
src=Path(receipt['source']);candidate=Path(receipt['finalPath'])
assert sha(src)==receipt['source_sha256'] and sha(candidate)==receipt['final_sha256']
with zipfile.ZipFile(src) as a,zipfile.ZipFile(candidate) as b:
    assert set(a.namelist())==set(b.namelist())
    changed=[n for n in a.namelist() if a.read(n)!=b.read(n)]
assert set(changed)=={'ppt/slides/slide52.xml','ppt/notesSlides/notesSlide52.xml'}
assert all(not x['changed'] for x in receipt['renders'] if x['slide']<=51)
backup=H/('before-native-training-entry123-'+receipt['source_sha256']+'.pptx')
if not backup.exists():shutil.copy2(src,backup)
shutil.copy2(candidate,src);shutil.copy2(src,T/'shuolee/reports'/src.name)
assert sha(src)==receipt['final_sha256']
receipt.update(canonical_replaced=True,promoted_at=stamp,root_visual_review=True,
    root_visual_review_scope='Actual final52 PNG reviewed: 16-step synthetic field,200k65x65 single step,168495/200k and31505 failed, real94 fit pending clearly legible; no overflow.')
save(H/'free-transport123-native-training-deck-receipt.json',receipt)
ui=dict(updated=stamp,url='http://127.0.0.1:8870/diagnostics/video10-global-initializer/index.html#native-training-entry',
    actual_reload=True,AX_verified=True,visible_evidence=['16steps mean loss0.081274 to0.017802','200k65x65single step','168495pass31505fail','Full94fit pending','no new person PLY'],
    screenshot_verified=False,screenshot_limitation='Existing zero-width screenshot limitation retained; this round verifies rendered accessibility content only.',
    tab15_untouched=True,company00_remote_verified=False,new_PLY=False)
save(O/'viewer-ui-review.json',ui)
pub=read(O/'publication.json');pub.update(updated=stamp,state='report_UI_and_same_PPT_published',PPT=str(src),PPT_sha256=sha(src),PPT_only_slide52_changed=True,
    viewer_UI=str(O/'viewer-ui-review.json'),viewer_UI_AX_verified=True,viewer_UI_screenshot_verified=False,new_real_training=False,goal_complete=False)
save(O/'publication.json',pub);save(D/'publication.json',pub)
j=read(C/'local-research-status.json');j.update(updated_at=stamp,current_report_sha256=sha(src),latest_publication=str(O/'publication.json'),latest_viewer_review=str(O/'viewer-ui-review.json'));save(C/'local-research-status.json',j)
for dest in [R/'video10-global-initializer',T/'viewer/diagnostics/video10-global-initializer',Path('D:/00/工作/查看器/diagnostics/video10-global-initializer')]:
    for name in ['publication.json','viewer-ui-review.json']:shutil.copy2(lp(O/name),lp(dest/'free-clothing-transport/native-training-entry'/name))
    shutil.copy2(lp(D/'publication.json'),lp(dest/'free-clothing-transport/publication.json'))
print(json.dumps(dict(PPT_sha256=sha(src),first51_unchanged=True,actual_UI_AX=True,real_training=False)))
