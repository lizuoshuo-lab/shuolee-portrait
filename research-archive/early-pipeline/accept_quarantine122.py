from pathlib import Path
import hashlib,json,shutil
from datetime import datetime,timezone,timedelta

H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122'
O=R/'video-update-20261027/background-geometry/material-quarantine'
C=H.parent/'shuolee-coordination'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
now=datetime.now(timezone.utc).isoformat()
r=read(H/'quarantine122-deck-receipt.json')
if r['canonical_replaced']:raise RuntimeError('Already accepted, do not repeat')
delivery=Path(r['source']);canonical=T/'shuolee/reports/02_有限视角重建优化路线.pptx';new=Path(r['finalPath'])
assert sha(delivery)==r['source_sha256'] and sha(canonical)==r['source_sha256']
assert sha(new)==r['final_sha256']
package=read(H/'quarantine122-exact-package.json')
assert package['other_members_byte_identical']
assert set(package['changed'])=={'ppt/slides/slide52.xml','ppt/notesSlides/notesSlide52.xml','ppt/media/n1_material122.png'}
assert len(r['renders'])==52 and all(not x['changed'] for x in r['renders'][:51])
assert (H/'quarantine122-final-52.png').is_file()
for name in ['report-ui.png','viewer-ui-loaded.png']:assert (O/name).stat().st_size>10000
shutil.copy2(new,canonical);shutil.copy2(new,delivery)
assert sha(canonical)==sha(delivery)==r['final_sha256']
r.update(canonical_replaced=True,accepted=now,canonical=str(canonical),delivery=str(delivery),changed_zip_members=package['changed'],first51_members_and_render_hashes_identical=True,changed_slide_visually_reviewed=True,evidence_image_aspect_ratio=2,PowerPoint_or_WPS_runtime_verified=False)
write(H/'quarantine122-deck-receipt.json',r)
write(O/'ppt-update.json',{k:r[k] for k in ['accepted','canonical','delivery','final_sha256','changed_zip_members','first51_members_and_render_hashes_identical','changed_slide_visually_reviewed','evidence_image_aspect_ratio','PowerPoint_or_WPS_runtime_verified']})
ui=dict(updated=now,state='actual_local_report_and_existing_model_reviewed',report_url='http://127.0.0.1:8870/body-smplx.html#robust122-n10',report_screenshot='report-ui.png',viewer_screenshot='viewer-ui-loaded.png',actual_checks=['New N10 table and native before/after figure visible','Viewer latestDiagnostic records4000basis, family399 andnoimprovement','119sofa remainsrendered,204649points,89cameras','No newpersonPLY orcamera substituted','tab15 preserved'],new_training=False,new_model_import=False,company00_verified=False,cloud_sync_verified=False)
write(O/'viewer-ui-review.json',ui)
pub=read(O/'publication.json');pub.update(updated=now,UI_verified=True,PPT_updated=True,UI_review='viewer-ui-review.json',PPT_review='ppt-update.json');write(O/'publication.json',pub)
q=read(R/'queue-state.json');q['updated']=now;q['last_goal_turn']['remaining']='Preregister one native recovered-background supplement with frozen family isolation; inspect physical support before any fixedK freshcamera candidate. No training.';q['last_goal_turn']['publication_complete']=True;write(R/'queue-state.json',q)
e=read(R/'evidence-ledger.json');e['updated']=now
for item in ['video-update-20261027/background-geometry/material-quarantine/viewer-ui-review.json','video-update-20261027/background-geometry/material-quarantine/ppt-update.json']:
    if item not in e['artifacts']:e['artifacts'].append(item)
write(R/'evidence-ledger.json',e)
s=read(C/'local-research-status.json');s['updated_at']=now;s['as_of_date']=datetime.now(timezone(timedelta(hours=8))).date().isoformat();s['source_queue_updated']=now;s['N9_N10_UI_verified']=True;s['same_PPT_lastpage_updated']=True;s['viewer']['current_122_ui_verified']=True;s['viewer']['current_122_ui_scope']='Actual N10 report table andimage verified; mainviewer119sofa204649points/89cameras retained, latestDiagnostic updated. No new model or00verification.';s['renderer_readiness']['stage']='material_mismatch_ablation_reviewed_and_published';write(C/'local-research-status.json',s)
lt=T/'shuolee/long-term-plan.md';t=lt.read_text(encoding='utf-8-sig');marker='N9/N10 publication accepted'
if marker not in t:lt.write_text(t+'\n\n'+marker+' '+now+'：实际进展页与主查看器已核验，119历史模型及89相机保留；同一52页PPT仅末页与其专属图片/备注更新，其他51页内容及渲染哈希不变。N11只有7帧遮罩准备完成；下一步先登记一次恢复背景区域补提，未启动新相机或训练。\n',encoding='utf-8')
print(json.dumps(dict(publication_complete=True,UI_verified=True,PPT_updated=True,first51_unchanged=True,goal_complete=False,training_started=False),ensure_ascii=False))
