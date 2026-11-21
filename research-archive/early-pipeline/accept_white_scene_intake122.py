"""Record completed actual UI review, without claiming new model/calibration acceptance."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';O=R/'video-update-20261027/clip03-scene-support';C=H.parent/'shuolee-coordination'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat()
assert not (O/'viewer-ui-review.json').exists()
for name in ['report-ui.png','viewer-ui.png']:assert (O/name).stat().st_size>10000
ui=dict(updated=stamp,actual_report_tab='40',actual_viewer_tab='31',report_url='http://127.0.0.1:8870/body-smplx.html#robust122-n12a',
    verified=['N12a heading,4000basis explanation andnativewhiteTimage actuallyvisible','LatestDiagnostic updated bypolling in actualviewer','Historical119sofa stillrendered inGaussianmode;no modelreplacement','tab15 untouched'],
    screenshots={name:sha(O/name) for name in ['report-ui.png','viewer-ui.png']},
    new_pointcloud=False,new_camera=False,new_training=False,physical_calibration_passed=False,company00_tested=False,
    limitations='This is a preparation/report review; no new OpenFile/Gaussian-points acceptance because no new model was produced.')
save(O/'viewer-ui-review.json',ui)
p=read(O/'publication.json');p.update(updated=stamp,UI_verified=True,UI_evidence='viewer-ui-review.json');save(O/'publication.json',p)
q=read(R/'queue-state.json');q['updated']=stamp;q['last_goal_turn']['publication_complete']=True;q['last_goal_turn']['completed']='N11once-only nativebackground supplement and42+10materialreview; N12a newwhiteTsevennativeframes/peoplexclusions actuallyreviewed andpublished.';save(R/'queue-state.json',q)
s=read(C/'local-research-status.json');s.update(updated_at=stamp,source_queue_updated=stamp,N12a_UI_verified=True);s['viewer']['current_122_ui_verified']=True;s['viewer']['current_122_ui_scope']='ActualN12a nativewhiteTpreparationreport andlatestDiagnostic reviewed;119sofaGaussian remains. No newcalibration/model/00test.';save(C/'local-research-status.json',s)
print(json.dumps(dict(UI_verified=True,goal_active=True,new_camera=False,training=False),ensure_ascii=False))
