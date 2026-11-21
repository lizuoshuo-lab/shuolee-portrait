from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,zipfile
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
D=T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route/free-clothing-transport'
C=H.parent/'shuolee-coordination'
def lp(p):return Path('\\\\?\\'+str(Path(p).resolve()))
def read(p):return json.loads(lp(p).read_text(encoding='utf-8-sig'))
def save(p,v):lp(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(lp(p).read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat()
receipt=read(H/'free-transport123-native-deck-receipt.json')
new,old=Path(receipt['finalPath']),Path(receipt['source'])
assert sha(new)==receipt['final_sha256'] and sha(old)==receipt['source_sha256']
assert receipt['first51_render_hashes_identical'] and receipt['actual_changed_slide_review']['reviewed']
with zipfile.ZipFile(new) as n,zipfile.ZipFile(old) as o:
    assert set(n.namelist())==set(o.namelist())
    changed=sorted(k for k in n.namelist() if n.read(k)!=o.read(k))
    assert changed==['ppt/notesSlides/notesSlide52.xml','ppt/slides/slide52.xml']
backup=H/('before-free-transport123-'+receipt['source_sha256']+'.pptx')
if not backup.exists():shutil.copy2(old,backup)
assert sha(backup)==receipt['source_sha256']
shutil.copy2(new,old);shutil.copy2(new,T/'shuolee/reports'/old.name)
receipt.update(canonical_replaced=True,canonical_replaced_at=stamp,root_actual_slide52_review=True,root_review='Native table readable, CPU synthetic scope and GPU incomplete status clear; no real-photo or quality claim.',previous_deck_backup=str(backup));save(H/'free-transport123-native-deck-receipt.json',receipt)
ui=dict(updated=stamp,tab_id='32',url='http://127.0.0.1:8870/diagnostics/video10-global-initializer/index.html#free-clothing-transport',actual_reload=True,AX_verified=True,
    observed=['free-clothing-transport section','18 CPU controls and full covariance/SH direction','relative scale cancellation boundary','two retained native environment failures','00 commit8d54497 and CI139','no real training or new point cloud'],
    screenshot_verified=False,screenshot_error='Cannot take screenshot with 0 width.',layout_or_rendering_accepted=False,screenshot_workaround_attempted=False,historical_tab15_touched=False,new_PLY=False,WPS_cloud_verified=False)
save(D/'viewer-ui-review.json',ui)
p=D/'publication.json';v=read(p);v.update(updated=stamp,viewer_UI_pending=False,viewer_AX_verified=True,viewer_visual_layout_verified=False,viewer_review=str(D/'viewer-ui-review.json'),PPT_pending=False,PPT=str(old),PPT_sha256=sha(old),PPT_first51_unchanged=True,PPT_changed_slides=[52]);save(p,v)
for base in [T/'shuolee/current/robust-portrait-122/video10-global-initializer',T/'viewer/diagnostics/video10-global-initializer',Path('D:/00/工作/查看器/diagnostics/video10-global-initializer')]:
    for name in ['viewer-ui-review.json','publication.json']:shutil.copy2(lp(D/name),lp(base/'free-clothing-transport'/name))
p=C/'local-research-status.json';v=read(p);v.update(updated_at=stamp,current_report_sha256=sha(old),latest_viewer_review=str(D/'viewer-ui-review.json'),latest_publication=str(D/'publication.json'));save(p,v)
print(json.dumps(dict(PPT_sha256=sha(old),first51_unchanged=True,AX_verified=True,screenshot_verified=False,goal_complete=False)))
