"""Record actual T5 display/slide review; do not promote diagnostic supervision."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';O=R/'native-supervision';C=H.parent/'shuolee-coordination'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'viewer-ui-review.json').exists()
r=read(H/'nativeSupervision-122-deck-receipt.json');src=Path(r['source']);final=Path(r['finalPath'])
assert sha(src)==r['source_sha256'] and sha(final)==r['final_sha256']
assert r['changed_slides']==[52] and len(r['renders'])==52
assert read(H/'nativeSupervision-122-exact-package.json')['other_members_byte_identical']
assert read(H/'nativeSupervision-122-final-validation.json')['finalSha256']==r['final_sha256']
for n in ['report-ui.png','report-lower-ui.png','viewer-ui.png']:assert (O/n).stat().st_size>10000
stamp=datetime.now(timezone.utc).isoformat()
shutil.copy2(final,src);mirror=T/'shuolee/reports/02_有限视角重建优化路线.pptx';shutil.copy2(final,mirror)
assert sha(src)==sha(mirror)==r['final_sha256']
r.update(canonical_replaced=True,updated=stamp,actual_slide52_reviewed=True,first51_render_and_package_unchanged=True)
save(H/'nativeSupervision-122-deck-receipt.json',r)
save(O/'ppt-update.json',dict(updated=stamp,path=str(src),sha256=sha(src),slides=52,changed=[52],actual_slide52_reviewed=True,first51_unchanged=True,receipt=str(H/'nativeSupervision-122-deck-receipt.json')))
ui=dict(updated=stamp,actual_report_tab='40',actual_viewer_tab='31',report_url='http://127.0.0.1:8870/body-smplx.html#robust122-t5',
 verified=['Actual report shows all464 nativeframes/actualclock, source identity and original/cutout images.',
 'Report states alpha remainsdraft, full464camerasunregistered and13.049pxfacemax unchanged.',
 'Mainpreview visibly retains119sofa andshowsupdatedT5latestDiagnostic; no newpointcloud registered.',
 'Slide52 actually viewed, originalaspectpreserved,no addedimageannotations; allother51slides package/renderidentical.',
 'Tab15 untouched; nativev1/v2 andold113 unchanged.'],
 screenshots={n:sha(O/n) for n in ['report-ui.png','report-lower-ui.png','viewer-ui.png']},
 actual_successful_person_fit=False,actual_successful_person_GPU_training=False,physical_calibration_accepted=False,new_human_pointcloud=False,company00_tested=False)
save(O/'viewer-ui-review.json',ui)
p=read(O/'publication.json');p.update(updated=stamp,report_UI_verified=True,UI_evidence='viewer-ui-review.json',PPT_pending=False,PPT='ppt-update.json');save(O/'publication.json',p)
q=read(R/'queue-state.json');q.update(updated=stamp,phase='T5_native_supervision_reviewed',running=None)
q['last_goal_turn'].update(publication_complete=True,completed='T5 all464 nativeframes/actualclock anddraftalphas complete;all464silhouettes and5nativehead/hands/feet setsreviewed. Report/mainpreview/slide52reviewed,first51unchanged. Fullcamera andphysicalsupervision remainunaccepted.')
save(R/'queue-state.json',q)
s=read(C/'local-research-status.json');s.update(updated_at=stamp,source_queue_updated=stamp,latest_native_supervision_PPT=str(O/'ppt-update.json'),T5_GPU_task_active=False)
s['viewer'].update(current_122_ui_verified=True,current_122_ui_scope='T5 report andretained119mainpreview actuallyreviewed; no newperson. ExistingT2nativev2registeredUI remains separatelyscoped.')
save(C/'local-research-status.json',s)
save(C/'local-native-supervision-handoff.json',dict(updated=stamp,stage='T5',evidence=str(O/'review.json'),
 result='All464 clip03native1520x2704 images andactualPTS fixed;464alpha proposals,7existingdiagnostic images bitexact;actualcoarseallmask/native5framesreview. No camera/fit/GS orphysicaledgeacceptance.',
 data_root='D:/GS-SLAM-Data/robust122-whiteT-native-supervision',
 company_action='Information only. T5GPUcomplete/released. Coordinateyourlocal GPUownqueue normally. Old113/00source untouched. No deployment/training request andnoackloop.',
 limits='Full464camerasunregistered; firstYAML4000basisremainsconditional;13faceheldoutmax13.04875764pxunchanged. Draftmaskapproval nottrainingadmission.',physical_admission=False,next=q['next']))
print(json.dumps(dict(T5_publication_complete=True,goal_active=True,physical_admission=False,ppt=str(src)),ensure_ascii=False))
