"""Record actual D4 display/deck checks and next registered engineering gap."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001');R=T/'shuolee/current/robust-portrait-122'
O=R/'native-edge-profile';C=H.parent/'shuolee-coordination'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'viewer-ui-review.json').exists()
prefix='nativeEdgeProfile-122';receipt=H/(prefix+'-deck-receipt.json');r=read(receipt);source=Path(r['source']);final=Path(r['finalPath'])
assert sha(source)==r['source_sha256'] and sha(final)==r['final_sha256']
assert r['changed_slides']==[52] and len(r['renders'])==52
assert read(H/(prefix+'-exact-package.json'))['other_members_byte_identical']
assert read(H/(prefix+'-final-validation.json'))['finalSha256']==r['final_sha256']
for n in ['report-ui.png','report-lower-ui.png','viewer-ui.png']:assert (O/n).stat().st_size>10000
stamp=datetime.now(timezone.utc).isoformat();shutil.copy2(final,source);mirror=T/'shuolee/reports/02_有限视角重建优化路线.pptx';shutil.copy2(final,mirror)
assert sha(source)==sha(mirror)==r['final_sha256']
r.update(canonical_replaced=True,updated=stamp,actual_slide52_reviewed=True,first51_render_and_package_unchanged=True);save(receipt,r)
save(O/'ppt-update.json',dict(updated=stamp,path=str(source),sha256=sha(source),slides=52,changed=[52],first51_unchanged=True,actual_slide52_reviewed=True,receipt=str(receipt)))
save(O/'viewer-ui-review.json',dict(updated=stamp,actual_report_tab='40',actual_viewer_tab='31',report_url='http://127.0.0.1:8870/body-smplx.html#robust122-d4',
 verified=['D4 actualreport reloadandreview: nativecrops, all11frozen-pointset line statistics and modelbiascontrols visible.',
 'Mainviewer completedrender of119sofa,89cameras andnewD4diagnostic. No fake newhumanmodel.',
 'Slide52 actuallyinspected and first51render/package identical. Tab15 notoperated.'],
 screenshots={n:sha(O/n) for n in ['report-ui.png','report-lower-ui.png','viewer-ui.png']},new_human_training=False,new_human_pointcloud=False,compensation_applied=False,physical_calibration_accepted=False,company00_tested=False))
p=read(O/'publication.json');p.update(updated=stamp,report_UI_verified=True,UI_evidence='viewer-ui-review.json',PPT_pending=False,PPT='ppt-update.json');save(O/'publication.json',p)
nextstep='T7 native-capacity/plan.json is registered, NOT run. Verify realGPUprocesses and00coordination first, then implement one bounded unchangedv2 200000point native1520x2704 and2704x1520 forward/backward/freshone-step/fullSHPLY-scene roundtrip control under sharedflock. Preservefullresolutionandpointcount;stop onfailure, no fallback/thresholdsweep. This fills a documented64x64-only engineering gap, not physicalalignment. Also audit remaining per-person physicaladmission evidence; D1-D4edge branch ended withoutadoption and cannot be rerun.'
assert read(R/'native-capacity/plan.json')['stage']=='T7_native_resolution_capacity'
q=read(R/'queue-state.json');q.update(updated=stamp,phase='D4_reviewed_T7_registered_not_running',running=None,next=nextstep)
q['last_goal_turn'].update(publication_complete=True,completed='D4 completedandactuallyreviewed; single-edge peakartifact and modelmismatchbias established. Report/119UI/slide52 verified. T7 native200kcapacity gap registered, not executed. Goalunachieved.',remaining=nextstep);save(R/'queue-state.json',q)
p=read(R/'protocol.json');p['next']=nextstep;save(R/'protocol.json',p)
c=read(C/'local-research-status.json');c.update(updated_at=stamp,source_queue_updated=stamp,next=nextstep,latest_native_edge_profile_PPT=str(O/'ppt-update.json'),local_GPU_task_active=False,native_capacity_plan=str(R/'native-capacity/plan.json'))
c['viewer'].update(current_122_ui_verified=True,current_122_ui_scope='D4 report and retained119mainpreview; no newcalibration or personmodel.');save(C/'local-research-status.json',c)
save(C/'local-native-edge-handoff.json',dict(updated=stamp,stage='D4',evidence=str(O/'review.json'),
 result='Nativepixel edge-spread1320fits completed. Known single edge generates3oldinterpolatedpeaks; newmodel reducesjitter onsomelines but943boundhits. Gamma/twoedge controls have.447/1.546pxbias despite tinyformaluncertainty. No physicalcompensationorpersontrainingadmitted.',
 company_action='Informationonly. D4CPUcomplete, noGPUjobnow. Next registeredT7 requires sharedGPUflock andrealprocesscheck beforelaunch; two native-size200k syntheticcapacitycontrols, not humantraining. No00package/119engineeringchange. Noackloop.',
 next=nextstep))
print(json.dumps(dict(D4_publication_complete=True,T7_registered_not_running=True,goal_active=True,ppt=str(source)),ensure_ascii=False))
