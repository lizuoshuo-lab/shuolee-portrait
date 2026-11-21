"""Record actual report/preview/slide inspection and publish only the reviewed deck."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001');R=T/'shuolee/current/robust-portrait-122'
O=R/'directfisheye-research';N=R/'native-capacity';C=H.parent/'shuolee-coordination'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'viewer-ui-review.json').exists()
prefix='directFisheyeAudit-122';receipt=H/(prefix+'-deck-receipt.json');r=read(receipt)
source=Path(r['source']);final=Path(r['finalPath'])
assert sha(source)==r['source_sha256'] and sha(final)==r['final_sha256']
assert len(r['renders'])==52 and r['changed_slides']==[52]
assert read(H/(prefix+'-exact-package.json'))['other_members_byte_identical']
assert read(H/(prefix+'-final-validation.json'))['finalSha256']==r['final_sha256']
for name in ['report-ui.png','report-lower-ui.png','viewer-ui.png']:assert (O/name).stat().st_size>10000
assert (N/'report-ui.png').stat().st_size>10000
stamp=datetime.now(timezone.utc).isoformat()
shutil.copy2(final,source);mirror=T/'shuolee/reports/02_有限视角重建优化路线.pptx';shutil.copy2(final,mirror)
assert sha(source)==sha(mirror)==r['final_sha256']
r.update(canonical_replaced=True,updated=stamp,actual_slide52_reviewed=True,first51_render_and_package_unchanged=True);save(receipt,r)
ppt=dict(updated=stamp,path=str(source),sha256=sha(source),slides=52,changed=[52],first51_unchanged=True,actual_slide52_reviewed=True,receipt=str(receipt),scope='DirectFisheye source-adaptation summary plus clearly labeled local synthetic T7 image; T7/T7b details and official research sources retained in notes')
save(O/'ppt-update.json',ppt);save(N/'ppt-update.json',ppt)
ui=dict(updated=stamp,actual_report_tab='40',actual_viewer_tab='31',report_url='http://127.0.0.1:8870/body-smplx.html#robust122-directfisheye',
 verified=['Report upper and lower sections actually reviewed: input boundary,21officialsources,CPUcontrols,fullK/CVOadaptation and failure limits visible.',
 'Main viewer loaded119sofa and89cameras with newDirectFisheye latestDiagnostic. No newhumanPLY or modelpromotion.',
 'Onlyslide52 changed; allfirst51renderhashes and nonowned packageparts preserved. Newslideactually inspected. Historicaltab15 notoperated.'],
 screenshots={name:sha(O/name) for name in ['report-ui.png','report-lower-ui.png','viewer-ui.png']},training_executed=False,official_CUDA_reproduction=False,new_human_pointcloud=False,physical_quality_accepted=False,company00_tested=False)
save(O/'viewer-ui-review.json',ui)
save(N/'viewer-ui-review.json',dict(updated=stamp,report_url='http://127.0.0.1:8870/body-smplx.html#robust122-t7',actual_report_tab='40',actual_report_reviewed=True,report_screenshot_sha256=sha(N/'report-ui.png'),main_viewer_current_review=str(O/'viewer-ui-review.json'),historical119_preserved=True,physical_quality_accepted=False,T7_remains_failed=True))
for folder in [O,N]:
    p=read(folder/'publication.json');p.update(updated=stamp,report_UI_verified=True,UI_evidence='viewer-ui-review.json',PPT_pending=False,PPT='ppt-update.json');save(folder/'publication.json',p)
q=read(R/'queue-state.json');q.update(updated=stamp,phase='DirectFisheye_research_and_T7_publication_complete',running=None)
q['last_goal_turn'].update(publication_complete=True,completed='OfficialDirectFisheye source/CPU research and CVO adaptation design complete. T7/T7b failure evidence published; actualreport/mainpreview/slide52 reviewed. Broaderportraitgoal unachieved.')
save(R/'queue-state.json',q)
c=read(C/'local-research-status.json');c.update(updated_at=stamp,source_queue_updated=stamp,next=q['next'],local_GPU_task_active=False,latest_directfisheye_publication=str(O/'publication.json'))
c['viewer'].update(current_122_ui_verified=True,current_122_ui_scope='DirectFisheye research andT7report, retained119preview; no newhumantraining orphysicalacceptance.');save(C/'local-research-status.json',c)
save(C/'local-directfisheye-handoff.json',dict(updated=stamp,stage='DirectFisheye official source audit',evidence=str(O/'review.json'),
 result='CurrentofficialHEAD db9b3a25,21sources verified. CVO accumulates sharedGaussian photometric gradients, doesnotoptimizecamera ormaterialreprojection. Publicloadersdropcx/cy(potential5.985pxoffsetatourconditionalportraitK); directionR*zbreaksworldgauge; densificationanchor-only;RGBAheightcheckand scalarL1weight issues. CPU/sourcecontrols only, noofficialCUDA/persontraining.',
 engineering='T7nativeportrait200koneupdateworks butPLYfirstresponsegatefailed. T7b samepackageandNPZ repeatbitexact; secondPLYtiny; missingfirstpreexportarraysprecludesuniqueattribution. No rerun/thresholdrelaxation.',
 company_action='Informationonly: do notclaimDirectFisheye or CVO implemented incompanypipeline. No install/packageupdate requested. LocalGPUidle now; anynextGPUcontrol mustrespectsharedlock. Noackloop.',next=q['next']))
print(json.dumps(dict(research_publication_complete=True,T7_publication_complete=True,person_training=False,goal_active=True,ppt=str(source)),ensure_ascii=False))
