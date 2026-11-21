from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib
import shutil

H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122'
O=R/'native-residual-fit'
C=H.parent/'shuolee-coordination'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

assert not (O/'publication-ui-review.json').exists()
receipt=read(H/'residualFit-122-deck-receipt.json')
source,final=Path(receipt['source']),Path(receipt['finalPath'])
assert sha(source)==receipt['source_sha256'] and sha(final)==receipt['final_sha256']
assert receipt['changed_slides']==[52] and len(receipt['renders'])==52
assert all(not x['changed'] for x in receipt['renders'][:51])
assert read(H/'residualFit-122-exact-package.json')['other_members_byte_identical']
assert read(H/'residualFit-122-final-validation.json')['finalSha256']==receipt['final_sha256']
result=read(O/'result-completed.json')
assert len(result['checks'])==29 and all(x['passed'] for x in result['checks'])
assert not result['person_fit'] and not result['person_training']
assert result['CPU_projection_max_px']==result['checks'][0]['max_coordinate_error_px']
assert read(O/'review.json')['actual_image_review']
screens=['report-ui.png','report-lower-ui.png','viewer-ui.png']
assert all((O/name).stat().st_size>10000 for name in screens)
q=read(R/'queue-state.json')
assert q['phase']=='R6_publication_pending'
stamp=datetime.now(timezone.utc).isoformat()
mirror=T/'shuolee/reports/02_有限视角重建优化路线.pptx'
assert sha(mirror)==receipt['source_sha256']
shutil.copy2(final,source);shutil.copy2(final,mirror)
assert sha(source)==sha(mirror)==receipt['final_sha256']
receipt.update(canonical_replaced=True,updated=stamp,actual_slide52_reviewed=True,first51_render_and_package_unchanged=True)
save(H/'residualFit-122-deck-receipt.json',receipt)
save(O/'ppt-update.json',dict(updated=stamp,path=str(source),sha256=sha(source),slides=52,changed=[52],first51_unchanged=True,actual_slide52_reviewed=True,native_PowerPoint_tested=False))
save(O/'publication-ui-review.json',dict(updated=stamp,report_tab='40',main_preview_tab='31',native_UI_tab_retained='44',
    screenshots={name:sha(O/name) for name in screens},
    actual_review=['New report heading,29checktable,knowngeometry depthtriptych,changedvisibility policy andremainingtraininggap visibly inspected.',
                   'Mainviewer reloaded and finishedloading:119sofa204649points/89cameras unchanged,new29check diagnostic visible.',
                   'PPT52 finaltitle,diagram,legend andlimits visually reviewed; prior51renders andunrelated packagebytes identical.'],
    native_UI_new_execution=False,tab15_untouched=True,person_fit=False,person_training=False,physical_quality_accepted=False,company00_tested=False))
p=read(O/'publication.json');p.update(updated=stamp,report_UI_verified=True,PPT_pending=False,UI_evidence='publication-ui-review.json',PPT='ppt-update.json');save(O/'publication.json',p)
p=read(O/'progress.json');p.update(state='completed_reviewed',updated=stamp,publication='publication.json');save(O/'progress.json',p)
q.update(updated=stamp,phase='R6_publication_complete',running=None)
q['last_goal_turn'].update(publication_complete=True,completed='R6 completeoutput-camera fitter/continuousmeshvisibility/clothingmask/overlay/preparation:29checks,one smallCUDAprojection/backwardmax1.58e-5px. Originalsummaryaliaserror preserved and corrected without rerun. Real102targets/13faces13.049px refusedpreGPU. Report/119preview/PPT52actually reviewed. No positiveSMPL-Xfit,surfaceinit orhumantraining.')
save(R/'queue-state.json',q)
e=read(R/'evidence-ledger.json');e['updated']=stamp;e['confirmed'].append('R6 publication actually reviewed in reporttab40 and main119previewtab31. Model204649points/89cameras retained. PPT52updated,first51package/renderhashes unchanged. No00orofficialvieweracceptance.');e['artifacts'] += [str(O/'publication-ui-review.json'),str(O/'ppt-update.json')];save(R/'evidence-ledger.json',e)
c=read(C/'local-research-status.json');c.update(updated_at=stamp,source_queue_updated=stamp,next=q['next'],local_GPU_task_active=False,latest_native_residual_fit_publication=str(O/'publication.json'))
c['viewer'].update(current_122_ui_verified=True,current_122_ui_scope='R6 report/table/nativepixel syntheticdepthtriptych/limitations verified; main119204649points/89cameras retained. NativeR4UI retained,not rerun. No personquality/companyacceptance.')
save(C/'local-research-status.json',c)
save(C/'local-residual-fit-handoff.json',dict(updated=stamp,stage='R6 completeoutput-camera fitter visibility silhouette andoverlay',
    evidence=[str(O/'result-completed.json'),str(O/'review.json'),str(O/'serialization-repair.json')],publication=str(O/'publication.json'),
    url='http://127.0.0.1:8870/body-smplx.html#robust122-residual-fit',
    result='29checks. CPUprojection2.27e-13px; singleCUDAprojection/backward1.58e-5px; analyticdepth1.33e-15. 10kmeshquery0.247s. Actualpersoninput refusespreGPU;102targets/13faces allretained13.049px. No positiveSMPL-Xfit,surfaceinit ortraining. Summaryvariablealias repaired fromsavedpercheckrecords only.',
    gap='Formal optimizerdispatch stillneeds residual trainer/freshfit/surface/sourcebinding. Changed visibilitypolicy is combinedengineeringchange,not singlefactorDexperiment. Selfocclusion doesnotcoverexternalobjects. Realcalibration/material admission remainsfailed.',
    company_action='Information only; keep stable113code. Researchfilesnotmerged intodeployment. No acknowledgement loop.',next=q['next']))
print(json.dumps(dict(publication_complete=True,ppt=str(source),goal_active=True,physical_quality_pass=False,next=q['next']),ensure_ascii=False))
