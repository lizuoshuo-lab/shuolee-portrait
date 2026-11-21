from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,zipfile,shutil
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';D=T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route/free-clothing-transport';N=D/'native-ready'
def lp(p): return Path('\\\\?\\'+str(Path(p).resolve()))
def read(p):return json.loads(lp(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(lp(p).read_bytes()).hexdigest()
def save(p,v):lp(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
receipt=read(H/'free-transport123-gpu-deck-receipt.json');source=Path(receipt['source']);candidate=Path(receipt['finalPath'])
assert sha(source)==receipt['source_sha256'] and sha(candidate)==receipt['final_sha256']
with zipfile.ZipFile(source) as a,zipfile.ZipFile(candidate) as b:
    assert set(a.namelist())==set(b.namelist())
    changed=[n for n in a.namelist() if a.read(n)!=b.read(n)]
assert set(changed)=={'ppt/slides/slide52.xml','ppt/notesSlides/notesSlide52.xml'}
assert receipt['first51_render_hashes_identical'] and receipt['actual_changed_slide_review']['all52_rendered']
backup=H/('before-native-pass123-'+receipt['source_sha256']+'.pptx')
if not backup.exists():shutil.copy2(source,backup)
shutil.copy2(candidate,source);assert sha(source)==receipt['final_sha256']
stamp=datetime.now(timezone.utc).isoformat()
receipt.update(canonical_replaced=True,root_visual_review=True,root_visual_review_scope='Actual slide52 image reviewed: legible table, no overlap; synthetic-only/no-real-fit/GS/PLY explicit.',promoted_at=stamp,canonical_sha256=sha(source))
save(H/'free-transport123-gpu-deck-receipt.json',receipt)
review=read(N/'review.json');assert review.get('passed',review.get('independent_review_passed',False)),review.keys()
ui=dict(updated=stamp,url='http://127.0.0.1:8870/diagnostics/video10-global-initializer/index.html#free-clothing-transport',
    actual_reload=True,AX_verified=True,visible_evidence=['14 checks','11 completed forwards','independent RGB maximum','synthetic scope','no new person PLY'],
    screenshot_verified=False,screenshot_error='Cannot take screenshot with 0 width.',new_PLY=False,
    tab15_untouched=True,official_SIBR_tested=False,company00_remote_tested=False,WPS_sync_verified=False)
save(N/'viewer-ui-review.json',ui)
v=read(N/'publication.json');v.update(viewer_UI_pending=False,viewer_UI_scope='Actual reload and AX text check; screenshot API unavailable',PPT_pending=False,PPT=str(source),PPT_sha256=sha(source),first51_unchanged=True,independent_review=str(N/'review.json'));save(N/'publication.json',v)
for dest in [R/'video10-global-initializer',T/'viewer/diagnostics/video10-global-initializer',Path('D:/00/工作/查看器/diagnostics/video10-global-initializer')]:
    for name in ['publication.json','review.json','reviewer-process-review.json','viewer-ui-review.json']:
        shutil.copy2(lp(N/name),lp(dest/'free-clothing-transport/native-ready'/name))
print(json.dumps(dict(PPT_sha256=sha(source),slide52_only=True,UI_AX_verified=True,UI_screenshot=False),ensure_ascii=False))
