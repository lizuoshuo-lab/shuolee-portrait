from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';O=R/'video-update-20261027/background-geometry/scene-support-supplement';C=H.parent/'shuolee-coordination'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat();r=read(H/'support122-deck-receipt.json');assert not r['canonical_replaced']
source=Path(r['source']);canonical=T/'shuolee/reports/02_有限视角重建优化路线.pptx';new=Path(r['finalPath'])
assert sha(source)==sha(canonical)==r['source_sha256'];assert sha(new)==r['final_sha256']
pkg=read(H/'support122-exact-package.json');assert pkg['other_members_byte_identical'];assert set(pkg['changed'])=={'ppt/slides/slide52.xml','ppt/notesSlides/notesSlide52.xml','ppt/media/n1_material122.png'}
assert all(not row['changed'] for row in r['renders'][:51])
for f in ['report-ui.png','viewer-ui.png']:assert (O/f).stat().st_size>10000
assert (H/'support122-final-52.png').is_file()
shutil.copy2(new,source);shutil.copy2(new,canonical);assert sha(source)==sha(canonical)==r['final_sha256']
r.update(canonical_replaced=True,accepted=stamp,canonical=str(canonical),delivery=str(source),changed_zip_members=pkg['changed'],first51_members_and_render_hashes_identical=True,changed_slide_visually_reviewed=True,evidence_image_aspect_ratio=1,PowerPoint_or_WPS_runtime_verified=False)
save(H/'support122-deck-receipt.json',r)
save(O/'ppt-update.json',{k:r[k] for k in ['accepted','canonical','delivery','final_sha256','changed_zip_members','first51_members_and_render_hashes_identical','changed_slide_visually_reviewed','evidence_image_aspect_ratio','PowerPoint_or_WPS_runtime_verified']})
save(O/'viewer-ui-review.json',dict(updated=stamp,state='actual_local_report_and_viewer_reviewed',report_url='http://127.0.0.1:8870/body-smplx.html#robust122-n11-support',report_screenshot='report-ui.png',viewer_screenshot='viewer-ui.png',actual_checks=['N11table606families and42reviewrows visible','Originalpixelmaterial evidencevisible','Viewer latestDiagnostic updated','119sofa204649points and89cameras stillloaded','No newPLY orcamera published;tab15 preserved'],company00_verified=False,cloud_sync_verified=False,new_training=False))
p=read(O/'publication.json');p.update(updated=stamp,UI_verified=True,PPT_updated=True,UI_review='viewer-ui-review.json',PPT_review='ppt-update.json');save(O/'publication.json',p)
q=read(R/'queue-state.json');q['updated']=stamp;q['last_goal_turn']['publication_complete']=True;q['last_goal_turn']['remaining']='Preregister and inspect native static-object support of clip03 whiteT newcapture; no currenthatcamera ortraining admitted.';save(R/'queue-state.json',q)
e=read(R/'evidence-ledger.json');e['updated']=stamp
for n in ['viewer-ui-review.json','ppt-update.json']:e['artifacts'].append('video-update-20261027/background-geometry/scene-support-supplement/'+n)
save(R/'evidence-ledger.json',e)
s=read(C/'local-research-status.json');s.update(updated_at=stamp,source_queue_updated=stamp,N11_UI_verified=True,same_PPT_lastpage_updated=True);s['viewer']['current_122_ui_verified']=True;s['viewer']['current_122_ui_scope']='ActualN11supportreport/table/image andlatestDiagnostic reviewed;119sofa204649points/89cameras retained. No newcalibration/model/00test.';s['renderer_readiness']['stage']='scene_support_supplement_reviewed_not_admitted';save(C/'local-research-status.json',s)
print(json.dumps(dict(PPT_updated=True,UI_verified=True,first51_unchanged=True,new_camera=False,training=False)))
