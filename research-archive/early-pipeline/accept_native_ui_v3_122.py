from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001');R=T/'shuolee/current/robust-portrait-122';O=R/'native-ui-v3';C=H.parent/'shuolee-coordination'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (O/'viewer-ui-review.json').exists()
receipt=H/'nativeUI3-122-deck-receipt.json';d=read(receipt);source=Path(d['source']);final=Path(d['finalPath'])
assert sha(source)==d['source_sha256'] and sha(final)==d['final_sha256'] and d['changed_slides']==[52] and len(d['renders'])==52
assert read(H/'nativeUI3-122-exact-package.json')['other_members_byte_identical']
assert read(H/'nativeUI3-122-final-validation.json')['finalSha256']==d['final_sha256']
screens=['report-ui.png','report-lower-ui.png','viewer-ui.png','portrait-rgb-ui.png','landscape-rgb-ui.png','preview-rejection-ui.png','cached-reload-ui.png']
assert all((O/n).stat().st_size>10000 for n in screens)
stamp=datetime.now(timezone.utc).isoformat()
shutil.copy2(final,source);mirror=T/'shuolee/reports/02_有限视角重建优化路线.pptx';shutil.copy2(final,mirror)
assert sha(source)==sha(mirror)==d['final_sha256']
d.update(canonical_replaced=True,updated=stamp,actual_slide52_reviewed=True,first51_render_and_package_unchanged=True,
 finalizer_recovery='First validation subprocess lacked RUNTIME_NODE_MODULES. Same staged deck finalized with bundled runtime path, without reauthoring.');save(receipt,d)
save(O/'ppt-update.json',dict(updated=stamp,path=str(source),sha256=sha(source),slides=52,changed=[52],first51_unchanged=True,actual_slide52_reviewed=True,receipt=str(receipt)))
ui=dict(updated=stamp,native_tab='43',report_tab='40',main_preview_tab='31',
 urls=['http://127.0.0.1:8876/','http://127.0.0.1:8870/body-smplx.html#robust122-t9'],
 actual_review=['File chooser imported the actual browser-downloaded native ZIP. Two recorded cameras and RGB/alpha displayed without aspect deformation.',
 'Native/preview ZIP and float downloads verified; preview ZIP rejected, existing native package retained; reload/camera cache only2GPUjobs.',
 'Main preview remains119sofa204649points/89cameras with latestdiagnostic updated; source images and all camera parameters preserved.',
 'Report table/scopes and PPT52 inspected; first51 rendering and package members unchanged.'],
 screenshots={n:sha(O/n) for n in screens},tab15_untouched=True,new_person_model=False,physical_quality_accepted=False,
 native_v3_GUI_tested=True,arbitrary_import_verified=False,continuous_orbit_verified=False,official_external_GUI=False,company00_tested=False)
save(O/'viewer-ui-review.json',ui)
p=read(O/'publication.json');p.update(updated=stamp,report_UI_verified=True,PPT_pending=False,UI_evidence='viewer-ui-review.json',PPT='ppt-update.json');save(O/'publication.json',p)
q=read(R/'queue-state.json');q.update(updated=stamp,phase='T9_UI_and_publication_complete',running=None)
q['last_goal_turn'].update(publication_complete=True,completed='Exact native v3 actual import, two full-size recorded-camera UI responses0/download/rejection/cache passed; report/main119preview andPPT52 reviewed. Physical portrait goal remains unmet.');save(R/'queue-state.json',q)
c=read(C/'local-research-status.json');c.update(updated_at=stamp,source_queue_updated=stamp,next=q['next'],local_GPU_task_active=False,latest_native_UI_v3_publication=str(O/'publication.json'))
c['viewer'].update(current_122_ui_verified=True,current_122_ui_scope='T9 actual isolatedv3 OpenFile/recorded-camera/native-float-download, exact synthetic response0. Main119preview preserved. No person orofficialexternalGUI acceptance.');save(C/'local-research-status.json',c)
save(C/'local-native-ui-v3-handoff.json',dict(updated=stamp,stage='T9 exact native UI',evidence=str(O/'review.json'),url='http://127.0.0.1:8876/',
 result='26CPU checks plus21savedchecks andactualGUI: downloadednativeZIP OpenFile,2completeK/E cameras,fullnativeRGB/alpha responses0,real float/ZIP downloads bitexact,previewtype rejection andcachedviews. Native model NPZ canonical; PLY standard preview separately labelled.',
 limits='Registered synthetic package only, not arbitrary import/continuous orbit/officialSIBR/independentSpark/00 deployment. Realface13.049px andallframes/materialgates unchanged. No person training orchanges tocompanysource.',
 company_action='Information only; do not deploy or merge these research modules as stable119. No acknowledgement loop.',next=q['next']))
print(json.dumps(dict(publication_complete=True,ppt=str(source),goal_active=True,physical_quality_pass=False),ensure_ascii=False))
