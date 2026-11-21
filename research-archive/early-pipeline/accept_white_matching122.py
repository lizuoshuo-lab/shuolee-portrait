"""Publish the validated one-slide edit and actual local UI evidence."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';O=R/'video-update-20261027/clip03-native-matching';C=H.parent/'shuolee-coordination'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat()
assert not (O/'viewer-ui-review.json').exists()
receipt=read(H/'whiteMatching122-deck-receipt.json');source=Path(receipt['source']);final=Path(receipt['finalPath'])
assert sha(source)==receipt['source_sha256'] and sha(final)==receipt['final_sha256']
assert receipt['changed_slides']==[52] and len(receipt['renders'])==52
for name in ['report-ui.png','viewer-ui.png']:assert (O/name).stat().st_size>10000
shutil.copy2(final,source)
mirror=T/'shuolee/reports/02_有限视角重建优化路线.pptx';shutil.copy2(final,mirror)
assert sha(source)==sha(mirror)==receipt['final_sha256']
receipt.update(canonical_replaced=True,updated=stamp,actual_final_slide52_reviewed=True,first51_render_and_package_unchanged=True)
save(H/'whiteMatching122-deck-receipt.json',receipt)
save(O/'ppt-update.json',dict(updated=stamp,path=str(source),sha256=sha(source),slides=52,changed=[52],actual_slide52_reviewed=True,first51_unchanged=True,receipt=str(H/'whiteMatching122-deck-receipt.json'),
 finalizer_repair='Initial native import check needed RUNTIME_NODE_MODULES. Same authored candidate finalized with bundled path. This did not rerun any research or change authored deck contents.'))
ui=dict(updated=stamp,actual_report_tab='40',actual_viewer_tab='31',report_url='http://127.0.0.1:8870/body-smplx.html#robust122-n12b',
 verified=['N12b heading,4000basis explanation,104materialreview counts andnative evidence image visibly present','LatestDiagnostic reflectsN12b inactualviewer','Historical119sofa stillrendered inGaussianmode with204649points and89cameras;no modelreplacement','tab15 untouched'],
 screenshots={n:sha(O/n) for n in ['report-ui.png','viewer-ui.png']},new_pointcloud=False,new_camera=False,new_training=False,physical_calibration_passed=False,company00_tested=False,
 limitations='Material/report update only. No new OpenFile/Gaussian-points acceptance because no new model exists. Displayed native crops are centered on observations, not proof of3D alignment.')
save(O/'viewer-ui-review.json',ui)
p=read(O/'publication.json');p.update(updated=stamp,UI_verified=True,UI_evidence='viewer-ui-review.json',PPT_pending=False,PPT='ppt-update.json');save(O/'publication.json',p)
q=read(R/'queue-state.json');q['updated']=stamp;q['last_goal_turn']['publication_complete']=True;q['last_goal_turn']['completed']='N12b frozennative matching2740tracks/1584families,104materialrows+16contexts actuallyreviewed, report/viewer andsamePPTslide52 updated. No newcamera/training.';save(R/'queue-state.json',q)
s=read(C/'local-research-status.json');s.update(updated_at=stamp,source_queue_updated=stamp,N12b_UI_verified=True,latest_white_material_PPT=str(O/'ppt-update.json'))
s['viewer']['current_122_ui_verified']=True;s['viewer']['current_122_ui_scope']='ActualN12b materialreview report/latestDiagnostic checked;119sofa unchanged. No physicalcalibration/00deploymentpass.'
save(C/'local-research-status.json',s)
print(json.dumps(dict(publication_complete=True,goal_active=True,new_camera=False,training=False,ppt=str(source)),ensure_ascii=False))
