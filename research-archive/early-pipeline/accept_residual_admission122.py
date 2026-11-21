from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib
import shutil

H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122'
O=R/'native-residual-admission'
C=H.parent/'shuolee-coordination'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

assert not (O/'publication-ui-review.json').exists()
receipt=read(H/'residualAdmission-122-deck-receipt.json')
source,final=Path(receipt['source']),Path(receipt['finalPath'])
assert sha(source)==receipt['source_sha256'] and sha(final)==receipt['final_sha256']
assert receipt['changed_slides']==[52] and len(receipt['renders'])==52
assert all(not x['changed'] for x in receipt['renders'][:51])
assert read(H/'residualAdmission-122-exact-package.json')['other_members_byte_identical']
assert read(H/'residualAdmission-122-final-validation-v2.json')['finalSha256']==receipt['final_sha256']
assert all(x['passed'] for x in read(O/'result.json')['checks'])
assert read(O/'review.json')['actual_image_review']
screens=['report-ui.png','report-lower-ui.png','viewer-ui.png']
assert all((O/name).stat().st_size>10000 for name in screens)
stamp=datetime.now(timezone.utc).isoformat()
mirror=T/'shuolee/reports/02_有限视角重建优化路线.pptx'
assert sha(mirror)==receipt['source_sha256']
shutil.copy2(final,source);shutil.copy2(final,mirror)
assert sha(source)==sha(mirror)==receipt['final_sha256']
receipt.update(canonical_replaced=True,updated=stamp,actual_slide52_reviewed=True,first51_render_and_package_unchanged=True)
save(H/'residualAdmission-122-deck-receipt.json',receipt)
save(O/'ppt-update.json',dict(updated=stamp,path=str(source),sha256=sha(source),slides=52,changed=[52],first51_unchanged=True,actual_slide52_reviewed=True,native_PowerPoint_tested=False,
    publication_repairs=['Finalizer first required its configured Node modules path; environment corrected.','Legend clarified; new private candidate and validation receipt used because finalizer preserves prior outputs. No change to experiment code or data.']))
save(O/'publication-ui-review.json',dict(updated=stamp,report_tab='40',main_preview_tab='31',native_UI_tab_retained='44',
    screenshots={name:sha(O/name) for name in screens},
    actual_review=['New report heading/table, all48target plot, legend and downstream projection limitations visibly inspected.',
                   'Mainviewer reloaded:119sofa204649points/89cameras unchanged, new39check diagnostic visible.',
                   'PPT52 final legend/layout visually reviewed; prior51renders andunrelated package bytes identical.'],
    native_UI_new_execution=False,tab15_untouched=True,person_training=False,physical_quality_accepted=False,company00_tested=False))
p=read(O/'publication.json');p.update(updated=stamp,report_UI_verified=True,PPT_pending=False,UI_evidence='publication-ui-review.json',PPT='ppt-update.json');save(O/'publication.json',p)
p=read(O/'progress.json');p.update(state='completed_reviewed',updated=stamp,publication='publication.json');save(O/'progress.json',p)
q=read(R/'queue-state.json');q.update(updated=stamp,phase='R5_publication_complete',running=None)
q['last_goal_turn'].update(publication_complete=True,completed='R5complete residual material/input admission:39CPUchecks; explicitD0 realdiagnostic retains13.049px and is refused preGPU. Upstreampoint/visibility/surface pinholeboundaries identified. Report/119preview/PPT52 actually reviewed. No human training or realcalibration success.')
save(R/'queue-state.json',q)
c=read(C/'local-research-status.json');c.update(updated_at=stamp,source_queue_updated=stamp,next=q['next'],local_GPU_task_active=False,latest_native_residual_admission_publication=str(O/'publication.json'))
c['viewer'].update(current_122_ui_verified=True,current_122_ui_scope='R5 report/table/nativepixel controlplot/limitations verified; main119204649points/89cameras retained. R4v4nativeUI is retained, not rerun. No personquality/companyacceptance.')
save(C/'local-research-status.json',c)
save(C/'local-residual-admission-handoff.json',dict(updated=stamp,stage='R5 CPU formal input andmaterial camera contract',
    evidence=[str(O/'result.json'),str(O/'review.json'),str(O/'upstream-projection-audit.json')],publication=str(O/'publication.json'),
    url='http://127.0.0.1:8870/body-smplx.html#robust122-residual-admission',
    result='39CPUchecks. Independentknown nonzeroD16face max2.33e-13px; same observations ignoringD6.705px. Old noisyD0regression1.04e-13px. Real102targets/13face allretained13.049px; newaudit refuses beforeGPU. Oldtrain rejectsnew schema. No training/camera/coefffit.',
    gap='Legacy preparation/fitter keypoint projection AND meshvisibility, clothing masksampling stillpinhole. Need versionedintegration withsameKED. This is not evidence ofhistoricalghostingcause.',
    company_action='Information only; keep stable113 code. Research files not merged into deployment. No acknowledgement loop.',next=q['next']))
print(json.dumps(dict(publication_complete=True,ppt=str(source),goal_active=True,physical_quality_pass=False),ensure_ascii=False))
