from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, shutil, zipfile

H=Path(__file__).parent
C=H.parent/'shuolee-coordination'
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122'
D=T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route/free-clothing-transport'
G=D/'torch-consumer-gpu'
def lp(p): return Path('\\\\?\\'+str(Path(p).resolve()))
def read(p): return json.loads(lp(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(lp(p).read_bytes()).hexdigest()
def save(p,v): lp(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
stamp=datetime.now(timezone.utc).isoformat()
receipt=read(H/'free-transport123-torch-consumer-deck-receipt.json')
source=Path(receipt['source']); candidate=Path(receipt['finalPath'])
assert sha(source)==receipt['source_sha256']
assert sha(candidate)==receipt['final_sha256']
with zipfile.ZipFile(source) as a,zipfile.ZipFile(candidate) as b:
    assert set(a.namelist())==set(b.namelist())
    changed=[n for n in a.namelist() if a.read(n)!=b.read(n)]
assert set(changed)=={'ppt/slides/slide52.xml','ppt/notesSlides/notesSlide52.xml'}
assert receipt['first51_render_hashes_identical'] and receipt['main_table_xml_identical']
backup=H/('before-torch-consumer123-'+receipt['source_sha256']+'.pptx')
if not backup.exists(): shutil.copy2(source,backup)
shutil.copy2(candidate,source)
assert sha(source)==receipt['final_sha256']
shutil.copy2(source,T/'shuolee/reports'/source.name)
receipt.update(canonical_replaced=True,root_visual_review=True,promoted_at=stamp,
    root_visual_review_scope='Actual slide52 PNG reviewed: legible, no overlap, unchanged native14 table; new tensor-only verification and pending real training explicit.')
save(H/'free-transport123-torch-consumer-deck-receipt.json',receipt)
review=read(G/'review.json')
assert review.get('passed',review.get('independent_review_passed',False)),review.keys()
ui=dict(updated=stamp,url='http://127.0.0.1:8870/diagnostics/video10-global-initializer/index.html#free-clothing-transport',
    actual_reload=True,AX_verified=True,visible_evidence=['00 d4a3d73 deployed25/164','Torch CPU14','CUDA11forwards1backward83assertions','No native trainer connection or real new PLY'],
    screenshot_verified=False,screenshot_limitation='Prior screenshot returned zero-width surface; not retried or bypassed.',
    tab15_untouched=True,company00_remote_verified=False,new_PLY=False)
save(G/'viewer-ui-review.json',ui)
pub=dict(updated=stamp,state='published_and_handoff_dispatched',GPU_review=str(G/'review.json'),GPU_review_sha256=sha(G/'review.json'),
    viewer_UI=str(G/'viewer-ui-review.json'),viewer_UI_AX_verified=True,viewer_UI_screenshot_verified=False,
    PPT=str(source),PPT_sha256=sha(source),PPT_only_slide52_changed=True,
    handoff=str(C/'handoffs/torch-clothing-consumer-123-20261029/manifest.json'),
    handoff_state='sent_00_deployment_pending',new_real_training=False,new_PLY=False,goal_complete=False)
save(D/'publication.json',pub)
j=read(C/'local-research-status.json')
j['torch_clothing_consumer_handoff'].update(state='sent_00_deployment_pending',dispatched_at=stamp,target_thread_id='01a17c8c-e70f-7822-ad39-440290b3d1da')
j['free_clothing_transport_handoff'].update(state='deployed_and_returned_patch_merged',commit='d4a3d73fdf8dbec8aed4f661eccd3bbf95f9210b',targeted_tests=25,full_CPU_tests=164)
j.update(updated_at=stamp,current_report_sha256=sha(source),latest_publication=str(D/'publication.json'),latest_viewer_review=str(G/'viewer-ui-review.json'))
save(C/'local-research-status.json',j)
for dest in [R/'video10-global-initializer',T/'viewer/diagnostics/video10-global-initializer',Path('D:/00/工作/查看器/diagnostics/video10-global-initializer')]:
    target=dest/'free-clothing-transport'
    for name in ['publication.json','production-connection-notes.json']:shutil.copy2(lp(D/name),lp(target/name))
    for name in ['review.json','viewer-ui-review.json']:shutil.copy2(lp(G/name),lp(target/'torch-consumer-gpu'/name))
print(json.dumps(dict(PPT_sha256=sha(source),only_slide52_changed=True,handoff_dispatched=True,UI_AX_verified=True,real_training=False),ensure_ascii=False))
