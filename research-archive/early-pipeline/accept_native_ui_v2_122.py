"""Finalize T2 report/UI evidence and the inspected last slide, retaining all limits."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';O=R/'native-ui-v2';C=H.parent/'shuolee-coordination'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'viewer-ui-review.json').exists()
stamp=datetime.now(timezone.utc).isoformat()
receipt=read(H/'nativeUIV2-122-deck-receipt.json')
src=Path(receipt['source']);final=Path(receipt['finalPath'])
assert sha(src)==receipt['source_sha256'] and sha(final)==receipt['final_sha256']
assert receipt['changed_slides']==[52] and len(receipt['renders'])==52
assert read(H/'nativeUIV2-122-exact-package.json')['other_members_byte_identical']
assert read(H/'nativeUIV2-122-final-validation.json')['finalSha256']==receipt['final_sha256']
for name in ['report-ui.png','viewer-ui.png','ui-completed.png']:
    assert (O/name).stat().st_size>10000
shutil.copy2(final,src)
mirror=T/'shuolee/reports/02_有限视角重建优化路线.pptx';shutil.copy2(final,mirror)
assert sha(src)==sha(mirror)==receipt['final_sha256']
receipt.update(canonical_replaced=True,updated=stamp,actual_final_slide52_reviewed=True,first51_render_and_package_unchanged=True)
save(H/'nativeUIV2-122-deck-receipt.json',receipt)
save(O/'ppt-update.json',dict(updated=stamp,path=str(src),sha256=sha(src),slides=52,changed=[52],actual_slide52_reviewed=True,first51_unchanged=True,receipt=str(H/'nativeUIV2-122-deck-receipt.json')))
save(O/'viewer-ui-review.json',dict(updated=stamp,actual_report_tab='40',actual_viewer_tab='31',actual_native_tab='42',
    report_url='http://127.0.0.1:8870/body-smplx.html#robust122-t2',native_url='http://127.0.0.1:8875/',
    verified=['Report actually reloaded and shows T2,4000basis,firstK/rectifiedK distinction and remainingphysicalfailure.',
              'Nativev2 actualscreen shows completed pinhole0,correctrecordedcamera and exhausted4jobbudget; no fifthjob.',
              'LatestDiagnostic visiblyupdated; historical119sofa204649Gaussians and89cameras preserved.',
              'Tab15 not operated; existingv1service preserved.'],
    screenshots={n:sha(O/n) for n in ['report-ui.png','viewer-ui.png','ui-completed.png','ui-pinhole0.png','ui-pinhole1.png','ui-kb4.png','ui-large.png','ui-alpha.png']},
    native_v2_GUI_verified=True,native_v2_GUI_scope=read(O/'review.json')['GUI_scope'],
    arbitrary_OpenFile_verified=False,continuous_orbit_verified=False,
    physical_calibration_accepted=False,new_human_pointcloud=False,company00_tested=False))
p=read(O/'publication.json');p.update(updated=stamp,report_UI_verified=True,UI_evidence='viewer-ui-review.json',PPT_pending=False,PPT='ppt-update.json');save(O/'publication.json',p)
q=read(R/'queue-state.json');q.update(updated=stamp,phase='T2_registered_native_UI_complete_next_training_admission',running=None)
q['last_goal_turn'].update(publication_complete=True,completed='T2 actualnativev2UI4controls/20CPUguards/downloadbytes pass; report/mainpreview/slide52actuallyreviewed,first51unchanged. Scopedrecordedviews only; nohumanqualityor00acceptance.')
save(R/'queue-state.json',q)
s=read(C/'local-research-status.json');s.update(updated_at=stamp,source_queue_updated=stamp,T2_UI_verified=True,latest_native_v2_UI_PPT=str(O/'ppt-update.json'))
s['viewer'].update(current_122_ui_verified=True,current_122_ui_scope='T2 report/mainpreviewretention andnative8875 registeredpackage/recordedcamera/RGBAlpha/download verified. NoarbitraryOpenFile/continuousorbit/humanquality/00acceptance.')
save(C/'local-research-status.json',s)
save(C/'local-native-v2-ui-handoff.json',dict(updated=stamp,stage='T2',evidence=str(O/'review.json'),
    result='4 actual recordedcamera nativev2 renders and pairedZIP/NPZ browserdownloads verified against frozenreference. Registeredsyntheticpackages only.',
    company_action='Information only; no deployment requested, no 119/113 replacement, no confirmation loop.',
    physical_admission=False,K_basis='FirstYAML4000 resize/center crop unchanged',next=q['next']))
print(json.dumps(dict(publication_complete=True,goal_active=True,native_v2_registered_UI=True,
    full_viewer_accepted=False,physical_admission=False,ppt=str(src)),ensure_ascii=False))
