"""Record actual UI/image review and publish only the validated slide52 edit."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';A=R/'native-training-integration';C=H.parent/'shuolee-coordination'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (A/'viewer-ui-review.json').exists()
stamp=datetime.now(timezone.utc).isoformat()
receipt=read(H/'nativeTraining122-deck-receipt.json');src=Path(receipt['source']);final=Path(receipt['finalPath'])
assert sha(src)==receipt['source_sha256'] and sha(final)==receipt['final_sha256']
assert receipt['changed_slides']==[52] and len(receipt['renders'])==52
for n in ['report-ui.png','report-ui-detail.png','viewer-ui.png']:assert (A/n).stat().st_size>10000
shutil.copy2(final,src);mirror=T/'shuolee/reports/02_有限视角重建优化路线.pptx';shutil.copy2(final,mirror)
assert sha(src)==sha(mirror)==receipt['final_sha256']
receipt.update(canonical_replaced=True,updated=stamp,actual_final_slide52_reviewed=True,first51_render_and_package_unchanged=True);save(H/'nativeTraining122-deck-receipt.json',receipt)
save(A/'ppt-update.json',dict(updated=stamp,path=str(src),sha256=sha(src),slides=52,changed=[52],actual_slide52_reviewed=True,first51_unchanged=True,receipt=str(H/'nativeTraining122-deck-receipt.json')))
save(A/'viewer-ui-review.json',dict(updated=stamp,actual_report_tab='40',actual_viewer_tab='31',report_url='http://127.0.0.1:8870/body-smplx.html#robust122-t0',
    verified=['T0 report heading,4000basis unchanged,synthetic120step comparison and caveats visibly inspected.','Actual20万-point comparison table,zero/30681 counts and native culling image visible.','LatestDiagnostic is T0 and historical119sofa204649Gaussians/89cameras remains loaded.','No operation on tab15.'],
    screenshots={n:sha(A/n) for n in ['report-ui.png','report-ui-detail.png','viewer-ui.png']},new_human_pointcloud=False,new_camera_candidate=False,new_training=False,
    synthetic_training_executed=True,production_renderer_fixed=False,physical_calibration_passed=False,company00_tested=False,
    limitations='Report refresh and historical model retention only. Synthetic native export was tested through actual CLI roundtrip, not new OpenFile/points GUI acceptance.'))
p=read(A/'publication.json');p.update(updated=stamp,UI_verified=True,UI_evidence='viewer-ui-review.json',PPT_pending=False,PPT='ppt-update.json');save(A/'publication.json',p)
q=read(R/'queue-state.json');q['updated']=stamp;q['last_goal_turn'].update(publication_complete=True,completed='T0a96point120step native training component partially passes;200k suite fails. T0b5singlevariable controls confirm opacity-compensation bound error,30681points update with only that rule disabled;independent rayoracle2.24e-8. ActualUI/PPT52 updated,productionrepair pending,nohumanadmission.');save(R/'queue-state.json',q)
s=read(C/'local-research-status.json');s.update(updated_at=stamp,source_queue_updated=stamp,T0_UI_verified=True,latest_native_training_PPT=str(A/'ppt-update.json'))
s['viewer'].update(current_122_ui_verified=True,current_122_ui_scope='ActualT0 report andlatestDiagnostic inspected;119sofa remains unchanged. Productionrenderer,physicalcalibration and00deployment notaccepted.');save(C/'local-research-status.json',s)
print(json.dumps(dict(publication_complete=True,goal_active=True,human_training=False,production_fix=False,ppt=str(src)),ensure_ascii=False))
