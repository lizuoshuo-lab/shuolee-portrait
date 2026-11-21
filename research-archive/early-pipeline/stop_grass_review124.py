from pathlib import Path
import json
from datetime import datetime,timezone
r=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124/grass-temporal')
p=json.loads((r/'review-013/result.json').read_text())
assert p['frames']==13
v=dict(updated=datetime.now(timezone.utc).isoformat(),continue_same_candidate=False,actual_visual_review=True,quality_accepted=False,gaussian_admitted=False,reviewed_frames=[x['frame'] for x in p['records']],reason='All13 native body/face/feet comparisons and common synthetic side images reviewed. Persistent backwards raised arm; new32->33 right-elbow jump52.7667deg and frame33 elbow residual37.3199->82.5238px fail pose protection. Stop at13, do not run27/94 or Gaussians. No change to weights or historical models.',review_files=['review-013/result.json','review-013/page-01.png','review-013/page-02.png','review-013/page-03.png','review-013/page-04.png'])
path=r/'review-decision.json'
assert not path.exists();path.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
print('GRASS_STOP_DECISION_SAVED')