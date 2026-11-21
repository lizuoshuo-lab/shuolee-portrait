from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';O=R/'native-cvo-control';C=H.parent/'shuolee-coordination'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if (O/'viewer-ui-review.json').exists():raise RuntimeError('Already accepted; no duplicate publication')
prefix='nativeCVO-122';receipt=H/(prefix+'-deck-receipt.json');p=read(receipt)
source=Path(p['source']);final=Path(p['finalPath'])
assert sha(source)==p['source_sha256'] and sha(final)==p['final_sha256']
assert len(p['renders'])==52 and p['changed_slides']==[52]
assert read(H/(prefix+'-exact-package.json'))['other_members_byte_identical']
assert read(H/(prefix+'-final-validation.json'))['finalSha256']==p['final_sha256']
for n in ['report-ui.png','report-lower-ui.png','viewer-ui.png']:
    assert (O/n).stat().st_size>10000
stamp=datetime.now(timezone.utc).isoformat()
shutil.copy2(final,source);mirror=T/'shuolee/reports/02_有限视角重建优化路线.pptx';shutil.copy2(final,mirror)
assert sha(source)==sha(mirror)==p['final_sha256']
p.update(canonical_replaced=True,updated=stamp,actual_slide52_reviewed=True,first51_render_and_package_unchanged=True);save(receipt,p)
save(O/'ppt-update.json',dict(updated=stamp,path=str(source),sha256=sha(source),slides=52,changed=[52],
    first51_unchanged=True,actual_slide52_reviewed=True,receipt=str(receipt),scope='Known synthetic CVO/native archive controls; no human quality claim'))
ui=dict(updated=stamp,report_tab='40',main_preview_tab='31',
    url='http://127.0.0.1:8870/body-smplx.html#robust122-t8',
    actual_review=['CVO report image/table/control limits and native-vs-PLY response table visible.',
                   'Main preview retains119sofa204649points/89cameras; new37checkdiagnostic visible.',
                   'Fullportrait/landscape RGB, portraitalpha, nativecrops and slide52 actually reviewed; no text added on source images.'],
    screenshots={n:sha(O/n) for n in ['report-ui.png','report-lower-ui.png','viewer-ui.png']},
    tab15_untouched=True,native_v3_GUI_tested=False,new_person_model=False,physical_quality_accepted=False,company00_tested=False)
save(O/'viewer-ui-review.json',ui)
review=read(O/'review.json');review.update(actual_native_crop_review_pending=False,actual_native_crop_review=True,updated=stamp);save(O/'review.json',review)
for parent in [T/'viewer',Path('D:/00/工作/查看器'),Path('D:/00/工作日志/2026-10-27/内参文件解析'),
               T/'viewer/new-capture-118/calibration-note',Path('D:/00/工作/查看器/new-capture-118/calibration-note')]:
    shutil.copy2(O/'review.json',parent/'122-native-cvo/review.json')
publication=read(O/'publication.json');publication.update(updated=stamp,report_UI_verified=True,UI_evidence='viewer-ui-review.json',PPT_pending=False,PPT='ppt-update.json');save(O/'publication.json',publication)
q=read(R/'queue-state.json');q.update(updated=stamp,phase='T8_components_and_publication_complete',running=None)
q['last_goal_turn'].update(publication_complete=True,completed='CVO mean-gradient/all-view statistics and exact-native archive controls passed; report/main119preview/slide52 actually reviewed. Physical portrait goal remains unmet.')
save(R/'queue-state.json',q)
c=read(C/'local-research-status.json');c.update(updated_at=stamp,source_queue_updated=stamp,next=q['next'],local_GPU_task_active=False,latest_native_cvo_publication=str(O/'publication.json'))
c['viewer'].update(current_122_ui_verified=True,current_122_ui_scope='T8report and retained119preview only; v3exact package nativeCUDAreadback tested, nativev3GUIpending.');save(C/'local-research-status.json',c)
save(C/'local-native-cvo-handoff.json',dict(updated=stamp,stage='T8 CVO component and exact-native package',
    evidence=str(O/'gpu-result.json'),source_modules=[str(H/x) for x in ['native_covisibility122.py','native_linear_archive122.py','native_scene_package122_v3.py']],
    result='22CPU+15GPUcontrols pass; mean2viewgrad matches joint within3.73e-9; allviewWORLDXYZgrad/visibility/radius includes8exclusivepointseach.200konefreshupdate at1520x2704and2704x1520. ExactNPZparameters andfullfloatnativeRGB/alpha readback0; standardPLYmaxresponse.003922 separatelyfails.',
    limits='Noactualadaptiveclone/prune, no realCVO qualityablation, no persontraining; realface13.049px andallframe/materialgatesunchanged. OldT7failure preserved. v1/v2runningservices untouched; v3GUIpending.',
    company_action='Information only. Do not deploy or call this part ofcompany119/main yet. GPUprocess completeandidle; same sharedlock retained. Noackloop.',next=q['next']))
print(json.dumps(dict(publication_complete=True,goal_active=True,quality_accepted=False,ppt=str(source)),ensure_ascii=False))
