"""Publish validated slide52 and actual local UI review, with no model promotion."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';A=R/'video-update-20261027/clip03-grid-coverage/array-cache-repair';C=H.parent/'shuolee-coordination'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat()
assert not (A/'viewer-ui-review.json').exists()
receipt=read(H/'whiteGrid122-deck-receipt.json');src=Path(receipt['source']);final=Path(receipt['finalPath'])
assert sha(src)==receipt['source_sha256'] and sha(final)==receipt['final_sha256']
assert receipt['changed_slides']==[52] and len(receipt['renders'])==52
for n in ['report-ui.png','viewer-ui.png']:assert (A/n).stat().st_size>10000
shutil.copy2(final,src);mirror=T/'shuolee/reports/02_有限视角重建优化路线.pptx';shutil.copy2(final,mirror)
assert sha(src)==sha(mirror)==receipt['final_sha256']
receipt.update(canonical_replaced=True,updated=stamp,actual_final_slide52_reviewed=True,first51_render_and_package_unchanged=True);save(H/'whiteGrid122-deck-receipt.json',receipt)
save(A/'ppt-update.json',dict(updated=stamp,path=str(src),sha256=sha(src),slides=52,changed=[52],actual_slide52_reviewed=True,first51_unchanged=True,receipt=str(H/'whiteGrid122-deck-receipt.json')))
save(A/'viewer-ui-review.json',dict(updated=stamp,actual_report_tab='40',actual_viewer_tab='31',report_url='http://127.0.0.1:8870/body-smplx.html#robust122-n12d',
 verified=['N12d heading,4000basis,equal-budget countsand3nativecrop rows actuallyvisible','latestDiagnostic updatedtoN12d inviewer','Historical119sofa204649Gaussians/89cameras loaded, unchangedmodel','tab15 untouched'],
 screenshots={n:sha(A/n) for n in ['report-ui.png','viewer-ui.png']},new_pointcloud=False,new_camera=False,new_training=False,physical_calibration_passed=False,company00_tested=False,
 limitations='Only local research evidence publication. No new OpenFile/Gaussian-points acceptance because no new model exists. Nativepoint-centeredcrops are not3D alignment proof.'))
p=read(A/'publication.json');p.update(updated=stamp,UI_verified=True,UI_evidence='viewer-ui-review.json',PPT_pending=False,PPT='ppt-update.json');save(A/'publication.json',p)
q=read(R/'queue-state.json');q['updated']=stamp;q['last_goal_turn'].update(publication_complete=True,completed='N12d equal-budget grid selection comparison+83materialrows+16contexts,actualUI andsamePPTslide52 completed. Roofcoverage improved;7FIT/2heldout nonground fails6/3;no newcamera/training.');save(R/'queue-state.json',q)
s=read(C/'local-research-status.json');s.update(updated_at=stamp,source_queue_updated=stamp,N12d_UI_verified=True,latest_white_grid_PPT=str(A/'ppt-update.json'))
s['viewer']['current_122_ui_verified']=True;s['viewer']['current_122_ui_scope']='Actual N12d controlled selection/material report and latestDiagnostic checked;119sofa unchanged. No physicalcalibration/00deployment pass.';save(C/'local-research-status.json',s)
print(json.dumps(dict(publication_complete=True,goal_active=True,new_camera=False,training=False,ppt=str(src)),ensure_ascii=False))
