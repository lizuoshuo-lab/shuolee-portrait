"""Record actual T4 UI and slide review, with no physical quality promotion."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';O=R/'native-fresh-preparation';C=H.parent/'shuolee-coordination'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'viewer-ui-review.json').exists()
r=read(H/'freshPrep-122-deck-receipt.json');src=Path(r['source']);final=Path(r['finalPath'])
assert sha(src)==r['source_sha256'] and sha(final)==r['final_sha256']
assert r['changed_slides']==[52] and len(r['renders'])==52
assert read(H/'freshPrep-122-exact-package.json')['other_members_byte_identical']
assert read(H/'freshPrep-122-final-validation.json')['finalSha256']==r['final_sha256']
for n in ['report-ui.png','report-lower-ui.png','viewer-ui.png']:assert (O/n).stat().st_size>10000
stamp=datetime.now(timezone.utc).isoformat()
shutil.copy2(final,src);mirror=T/'shuolee/reports/02_有限视角重建优化路线.pptx';shutil.copy2(final,mirror)
assert sha(src)==sha(mirror)==r['final_sha256']
r.update(canonical_replaced=True,updated=stamp,actual_slide52_reviewed=True,first51_render_and_package_unchanged=True)
save(H/'freshPrep-122-deck-receipt.json',r)
save(O/'ppt-update.json',dict(updated=stamp,path=str(src),sha256=sha(src),slides=52,changed=[52],actual_slide52_reviewed=True,first51_unchanged=True,receipt=str(H/'freshPrep-122-deck-receipt.json'),
 initial_export_attempt='Missing RUNTIME_NODE_MODULES in first-party import subprocess; canonical untouched. Same authored content finalized after supplying the existing runtime path.'))
ui=dict(updated=stamp,actual_report_tab='40',actual_viewer_tab='31',report_url='http://127.0.0.1:8870/body-smplx.html#robust122-t4',
 verified=['Actual report shows4000basis withconditionalK distinction, observed-support table and known syntheticcenter images.',
 'Report lower section explicitly retains13.049pxfacefailure,no newperson/GPU andcoarse externalocclusion limitation.',
 'Mainpreview visibly retains119sofa andshowsupdatedT4 latestDiagnostic; no syntheticpointcloud registeredas person.',
 'Slide52 actually viewed,originalaspectpreserved,no addedimageannotations; allother51slides package/renderidentical.',
 'Tab15 untouched; nativev1/v2source andold113 unchanged.'],
 screenshots={n:sha(O/n) for n in ['report-ui.png','report-lower-ui.png','viewer-ui.png']},
 actual_successful_person_fit=False,actual_successful_person_GPU_training=False,physical_calibration_accepted=False,new_human_pointcloud=False,company00_tested=False)
save(O/'viewer-ui-review.json',ui)
p=read(O/'publication.json');p.update(updated=stamp,report_UI_verified=True,UI_evidence='viewer-ui-review.json',PPT_pending=False,PPT='ppt-update.json');save(O/'publication.json',p)
q=read(R/'queue-state.json');q.update(updated=stamp,phase='T4_preparation_complete_next_native_supervision',running=None)
q['last_goal_turn'].update(publication_complete=True,completed='T4 observed-support defect controlled withsame8views;26CPUchecks,200ksyntheticfreeworldseeds,newfreshfitinterfaceactualinputdenial. Report/mainpreview/slide52reviewed,first51unchanged. No physicalfit/trainingpass.')
save(R/'queue-state.json',q)
s=read(C/'local-research-status.json');s.update(updated_at=stamp,source_queue_updated=stamp,T4_preparation_CPU_reviewed=True,latest_native_preparation_PPT=str(O/'ppt-update.json'))
s['viewer'].update(current_122_ui_verified=True,current_122_ui_scope='T4report andretained119mainpreview actuallyreviewed; no newperson. ExistingT2nativev2registeredUI remains separatelyscoped.')
save(C/'local-research-status.json',s)
save(C/'local-native-preparation-handoff.json',dict(updated=stamp,stage='T4',evidence=str(O/'review.json'),
 result='Fresh-preparation interface and observed-valid silhouette denominator26CPUcontrols passed;6knownpositive2unknown gives6/6 vs6/8.200ksyntheticfreeworld seeds only. Actualperson entry rejects beforeprior/fitGPU, allfaceerrors unchanged.',
 files=[str(H/n) for n in ['prepare_native_person122.py','native_surface122.py']],
 company_action='Information only. Old113/00installation source untouched. No serverdeployment ortraining request andnoackloop.',
 limits='Successful actual officialprior/SMPLXfit and realGPUexecution unverified. Conditional4000basis alreadyusedpreviouscameras; no physicalcalibration or qualitypass.',
 physical_admission=False,next=q['next']))
print(json.dumps(dict(T4_publication_complete=True,goal_active=True,physical_admission=False,ppt=str(src)),ensure_ascii=False))
