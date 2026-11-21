from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
T=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';D=R/'camera-document-review-20261028'
C=Path(__file__).parent.parent/'shuolee-coordination'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
stamp=datetime.now(timezone.utc).isoformat()
shots=[D/'report-ui.png',D/'viewer-ui.png'];assert all(p.exists() for p in shots)
review=dict(updated=stamp,report_tab='40',viewer_tab='31',actual_UI_reviewed=True,
    report_url='http://127.0.0.1:8870/body-smplx.html#robust122-camera-document',
    verified=['Report shows4000x3000primaryhypothesis andnewcamera-teamevidence','Mainviewer latestDiagnostic updated; old119sofa model204649points/89cameras retained'],
    screenshots=[dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in shots],
    historical_tab15_untouched=True,training_started=False,new_PLY=False,quality_accepted=False,remote00_verified=False,WPS_cloud_verified=False)
save(D/'viewer-ui-review.json',review)
p=read(D/'publication.json');p.update(updated=stamp,UI_verified=True);save(D/'publication.json',p)
q=read(R/'queue-state.json');q.update(updated=stamp,phase='camera_document_sources_and_primary4000_reviewed');q['last_goal_turn'].update(completed='N5/N6 actualreportUI andsamePPTpublished; twoPDFs andcamera-team/diagram/4000steering recorded withnativeUIverification',remaining='Residual spatial/time/depth/material diagnosis; full registration andface<1px admission remain unaccepted');q['last_goal_turn']['evidence']+=['camera-document-review-20261028/result.json','camera-document-review-20261028/primary4000-update.json','camera-document-review-20261028/viewer-ui-review.json'];save(R/'queue-state.json',q)
e=read(R/'evidence-ledger.json');e['updated']=stamp;e['confirmed'].append('Latest4000x3000primaryhypothesis/document review displayed inexisting report tab40 andmainviewer tab31, retaining119model. UpdatedexistingPPT first51pages remainunchanged. No new training or physicalcalibration claim.');e['artifacts'].append('camera-document-review-20261028/viewer-ui-review.json');save(R/'evidence-ledger.json',e)
s=read(C/'local-research-status.json');s.update(updated_at=stamp,source_queue_updated=stamp,camera_document_UI_verified=True);s['viewer']['current_122_ui_scope']='N5/N6 andcamera-source/4000primaryassumption report+latestDiagnostic verified;119 retained. R1e synthetic nativeUI separatelyverified.';save(C/'local-research-status.json',s)
p=read(R/'publication.json');p.update(updated=stamp,camera_document_UI_verified=True,camera_document_review='camera-document-review-20261028/viewer-ui-review.json');save(R/'publication.json',p)
print(json.dumps(dict(verified=True,model119_unchanged=True,training=False),ensure_ascii=False))
