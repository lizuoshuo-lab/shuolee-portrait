"""Complete the inspected T3 report and slide52, preserving physical failure."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';O=R/'native-training-entry';C=H.parent/'shuolee-coordination'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'viewer-ui-review.json').exists()
receipt=read(H/'nativeEntry-122-deck-receipt.json');src=Path(receipt['source']);final=Path(receipt['finalPath'])
assert sha(src)==receipt['source_sha256'] and sha(final)==receipt['final_sha256']
assert receipt['changed_slides']==[52] and len(receipt['renders'])==52
assert read(H/'nativeEntry-122-exact-package.json')['other_members_byte_identical']
assert read(H/'nativeEntry-122-final-validation.json')['finalSha256']==receipt['final_sha256']
for n in ['report-ui.png','report-table.png','viewer-ui.png']:assert (O/n).stat().st_size>10000
stamp=datetime.now(timezone.utc).isoformat()
shutil.copy2(final,src);mirror=T/'shuolee/reports/02_有限视角重建优化路线.pptx';shutil.copy2(final,mirror)
assert sha(src)==sha(mirror)==receipt['final_sha256']
receipt.update(canonical_replaced=True,updated=stamp,actual_slide52_reviewed=True,first51_render_and_package_unchanged=True)
save(H/'nativeEntry-122-deck-receipt.json',receipt)
save(O/'ppt-update.json',dict(updated=stamp,path=str(src),sha256=sha(src),slides=52,changed=[52],actual_slide52_reviewed=True,first51_unchanged=True,receipt=str(H/'nativeEntry-122-deck-receipt.json')))
ui=dict(updated=stamp,actual_report_tab='40',actual_viewer_tab='31',
    report_url='http://127.0.0.1:8870/body-smplx.html#robust122-t3',
    verified=['Actual T3 report reloaded: input problems, no false completeframe plan,13.049pxfacefailure andnohumantraining visible.',
              'LatestDiagnostic shows T3; historical119sofa204649Gaussians/89cameras loaded andpreserved.',
              'Table layout actually inspected; report-table.png is a crop of actualUI with no addedtext orresampling.',
              'Tab15 untouched; nativev1/v2service sources unchanged; no newGPUrender for this publication.'],
    screenshots={n:sha(O/n) for n in ['report-ui.png','viewer-ui.png','report-table.png']},
    actual_training_entry_rejection_verified=True,actual_successful_person_GPU_training_verified=False,
    physical_calibration_accepted=False,new_human_pointcloud=False,company00_tested=False)
save(O/'viewer-ui-review.json',ui)
p=read(O/'publication.json');p.update(updated=stamp,report_UI_verified=True,UI_evidence='viewer-ui-review.json',PPT_pending=False,PPT='ppt-update.json');save(O/'publication.json',p)
q=read(R/'queue-state.json');q.update(updated=stamp,phase='T3_prepared_input_entry_complete_next_fresh_surface_executor',running=None)
q['last_goal_turn'].update(publication_complete=True,completed='T3 actual entry46CPUchecks and actualrealinputexit3beforemodel/GPU;102targets/errorsunchanged. Report/mainpreview/slide52actuallyreviewed,first51unchanged. No newperson orphysicalpass.')
save(R/'queue-state.json',q)
s=read(C/'local-research-status.json');s.update(updated_at=stamp,source_queue_updated=stamp,T3_prepared_entry_reviewed=True,latest_native_training_entry_PPT=str(O/'ppt-update.json'))
s['viewer'].update(current_122_ui_verified=True,current_122_ui_scope='T3 report andretained119mainpreview verified. No newhuman. T2native8875registeredGUI remains separately scoped.')
save(C/'local-research-status.json',s)
save(C/'local-native-training-entry-handoff.json',dict(updated=stamp,stage='T3',evidence=str(O/'review.json'),
    result='Prepared-input actual train entry enforcesN3/N13/fullframe/image-mask evidence beforemodel/GPU.46CPUchecks; actualclip03exit3,13facemax13.0488px and102rawtargets unchanged.',
    files=[str(H/n) for n in ['native_training_admission122.py','train_native_person122.py']],
    company_action='Information only. Old113/00installation entry untouched. No deployment or training request; no acknowledgement loop.',
    limits='Upstream rawvideo calibration/freshfit-surface executor and positive actualGPU path notverified.464 is sourceencodedcount, not a trainingframe plan.',
    physical_admission=False,next=q['next']))
print(json.dumps(dict(T3_publication_complete=True,goal_active=True,physical_admission=False,ppt=str(src)),ensure_ascii=False))
