"""Record actual display acceptance, not human reconstruction or calibration acceptance."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';O=R/'residual-distortion-direction';C=H.parent/'shuolee-coordination'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'viewer-ui-review.json').exists()
prefix='residualLinesVerified-122';r=read(H/(prefix+'-deck-receipt.json'))
src=Path(r['source']);final=Path(r['finalPath']);assert sha(src)==r['source_sha256'] and sha(final)==r['final_sha256']
assert r['changed_slides']==[52] and len(r['renders'])==52
assert read(H/(prefix+'-exact-package.json'))['other_members_byte_identical']
assert read(H/(prefix+'-final-validation.json'))['finalSha256']==r['final_sha256']
for n in ['report-ui.png','report-lower-ui.png','viewer-ui.png']:assert (O/n).stat().st_size>10000
stamp=datetime.now(timezone.utc).isoformat()
shutil.copy2(final,src);mirror=T/'shuolee/reports/02_有限视角重建优化路线.pptx';shutil.copy2(final,mirror)
assert sha(src)==sha(mirror)==r['final_sha256']
r.update(canonical_replaced=True,updated=stamp,actual_slide52_reviewed=True,first51_render_and_package_unchanged=True)
save(H/(prefix+'-deck-receipt.json'),r)
save(O/'ppt-update.json',dict(updated=stamp,path=str(src),sha256=sha(src),slides=52,changed=[52],actual_slide52_reviewed=True,first51_unchanged=True,receipt=str(H/(prefix+'-deck-receipt.json'))))
save(O/'viewer-ui-review.json',dict(updated=stamp,actual_report_tab='40',actual_viewer_tab='31',
 report_url='http://127.0.0.1:8870/body-smplx.html#robust122-d1',
 verified=['D1 report actually refreshed and reviewed: raw architectural images,6FIT/5heldout andfailed compensation guard shown.',
 'T6 section actually reviewed withsameframe differentshortsmotifs image andunchanged original observations.',
 'Mainpreview actually retains119sofa with89cameras andnewD1diagnostic;no newhumanmodel claimed.',
 'Slide52 actuallyreviewed,originalimageaspectmaintained,other51slides byte/renderunchanged;tab15notoperated.'],
 screenshots={n:sha(O/n) for n in ['report-ui.png','report-lower-ui.png','viewer-ui.png']},
 new_human_training=False,new_human_pointcloud=False,compensation_applied=False,physical_calibration_accepted=False,company00_tested=False))
p=read(O/'publication.json');p.update(updated=stamp,report_UI_verified=True,UI_evidence='viewer-ui-review.json',PPT_pending=False,PPT='ppt-update.json');save(O/'publication.json',p)
q=read(R/'queue-state.json');q.update(updated=stamp,phase='D1_residual_direction_and_T6_reviewed',running=None)
q['last_goal_turn'].update(publication_complete=True,completed='T6 actualmaterialreview complete;D1new residualcompensation tested andnotpromoted. Report/retained119UI/slide52 actuallyreviewed;first51unchanged. Goalunachieved.')
save(R/'queue-state.json',q)
c=read(C/'local-research-status.json');c.update(updated_at=stamp,source_queue_updated=stamp,latest_residual_direction_PPT=str(O/'ppt-update.json'),local_GPU_task_active=False)
c['viewer'].update(current_122_ui_verified=True,current_122_ui_scope='D1/T6report andretained119mainpreview;no residualcandidate deployment ornewperson. Existingnativev2separate.')
save(C/'local-research-status.json',c)
save(C/'local-residual-direction-handoff.json',dict(updated=stamp,stage='D1/T6',evidence=str(O/'review.json'),
 result='Userauthorizedfurther residualdistortion trial executed onwhiteT. FixedK4parameter outputinverseBrown fitted6staticlines;5heldout1.773to1.783px,bound95.35pxfieldextrapolation rejected. No candidate applied. T6confirmed sameframe shorts family974 motifmismatch;raw/splitsretained.',
 company_action='Informationonly. AlllocalworkCPU;yourGPUownership unchanged. No new00package,113source,renderercontractortrainingrequest. Noackloop.',
 limits='No physicalfacecalibration ornewperson accepted;old13faceheldoutmax13.04875764pxunchanged. Goalactive.',next=q['next']))
print(json.dumps(dict(publication_complete=True,goal_active=True,compensation_applied=False,first51_unchanged=True,ppt=str(src)),ensure_ascii=False))
