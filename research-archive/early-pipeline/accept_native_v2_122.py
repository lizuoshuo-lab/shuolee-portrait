"""Persist actual report/UI review and publish verified slide52 only."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';O=R/'native-renderer-v2';C=H.parent/'shuolee-coordination'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'viewer-ui-review.json').exists()
stamp=datetime.now(timezone.utc).isoformat();receipt=read(H/'nativeV2-122-deck-receipt.json')
src=Path(receipt['source']);final=Path(receipt['finalPath'])
assert sha(src)==receipt['source_sha256'] and sha(final)==receipt['final_sha256']
assert receipt['changed_slides']==[52] and len(receipt['renders'])==52
assert read(H/'nativeV2-122-exact-package.json')['other_members_byte_identical']
assert read(H/'nativeV2-122-final-validation.json')['finalSha256']==receipt['final_sha256']
for name in ['report-ui.png','report-ui-detail.png','viewer-ui.png']:assert (O/name).stat().st_size>10000
shutil.copy2(final,src);mirror=T/'shuolee/reports/02_有限视角重建优化路线.pptx';shutil.copy2(final,mirror)
assert sha(src)==sha(mirror)==receipt['final_sha256']
receipt.update(canonical_replaced=True,updated=stamp,actual_final_slide52_reviewed=True,first51_render_and_package_unchanged=True);save(H/'nativeV2-122-deck-receipt.json',receipt)
save(O/'ppt-update.json',dict(updated=stamp,path=str(src),sha256=sha(src),slides=52,changed=[52],actual_slide52_reviewed=True,first51_unchanged=True,receipt=str(H/'nativeV2-122-deck-receipt.json')))
save(O/'viewer-ui-review.json',dict(updated=stamp,actual_report_tab='40',actual_viewer_tab='31',report_url='http://127.0.0.1:8870/body-smplx.html#robust122-t1',
    verified=['Actual report shows41driver+2CLI results,4000basis assumption and13.05px physical limitation.','All six synthetic normal-scale tiles and12step blurred ellipsoid visibly inspected.','LatestDiagnostic changed toT1; historical119sofa204649Gaussians/89cameras retained.','Tab15 not operated.'],
    screenshots={n:sha(O/n) for n in ['report-ui.png','report-ui-detail.png','viewer-ui.png']},
    synthetic_component_training_executed=True,live_v1_renderer_unchanged=True,native_v2_GUI_verified=False,
    physical_calibration_accepted=False,new_human_pointcloud=False,company00_tested=False,
    limitations='Actual report refresh and historical preview retention. v2package verified by driver/CLI only, not OpenFile Gaussian/points or native interactive GUI.'))
p=read(O/'publication.json');p.update(updated=stamp,UI_verified=True,UI_scope='report_and_historical_preview_only',UI_evidence='viewer-ui-review.json',PPT_pending=False,PPT='ppt-update.json');save(O/'publication.json',p)
q=read(R/'queue-state.json');q['updated']=stamp;q['last_goal_turn'].update(publication_complete=True,completed='T1 explicitv2renderer/package/trainingcomponent41controls and2independentCLI controls pass, syntheticimages/reportUI/PPT52 reviewed. Same200k30681updates,PLYresponse4.59e-6;physicalandv2GUIunaccepted.');save(R/'queue-state.json',q)
s=read(C/'local-research-status.json');s.update(updated_at=stamp,source_queue_updated=stamp,T1_UI_verified=True,latest_native_v2_PPT=str(O/'ppt-update.json'))
s['viewer'].update(current_122_ui_verified=True,current_122_ui_scope='T1 report andlatestDiagnostic visiblyverified;119sofa retained. This is not v2nativeinteractive,personquality or00verification.');save(C/'local-research-status.json',s)
print(json.dumps(dict(publication_complete=True,goal_active=True,native_v2_component_and_CLI=True,native_v2_GUI=False,physical_admission=False,ppt=str(src)),ensure_ascii=False))
