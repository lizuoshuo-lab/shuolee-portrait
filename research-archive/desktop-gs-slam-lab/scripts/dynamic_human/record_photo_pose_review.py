from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-hand-photo-pose-105'
files=[f'review/hands-{i:02d}.png' for i in range(7)]+['comparison/camera06-frame000732-full-person.png','comparison/camera12-frame000729-worst-full-person.png','comparison/camera12-frame000731-full-person.png','temporal/camera04-comparison.png','temporal/camera13-comparison.png']
rows=[dict(file=f,sha256=hashlib.sha256((O/f).read_bytes()).hexdigest()) for f in files]
out=dict(status='21handcontexts6sampledcontinuousframes3wholeperson_comparisons_visually_reviewed',recorded_utc=datetime.now(timezone.utc).isoformat(),files=rows,native_hand_contexts=21,temporal_sampled_contexts=6,wholeperson_contexts=3,clarity_accepted=False,adopted=False,goal_complete=False,
observations=['Open-hand source views show distinct fingers; original GS mixes them, and fixed/refined surface results retain dark patches and broken contours.','Refined pose and fixed pose look very similar across the reviewed contexts. No convincing restoration of finger separation.','Grasping fingers and held-object boundaries remain mixed in both refined and fixed models.','Whole-person silhouettes and large clothing regions remain similar; this local correction did not resolve the visible hand defects.'],
scope='All seven hand sheets were viewed at their native640-pixel width. Three full-person comparison files were displayed with tool resizing; original saved panels retain native pixels. Temporal contact sheets display721,727,731for04and13. Videos were encoded/probed/decoded but not visually played; other continuous frames were numerically checked, not all visually inspected.',
decision='Do not adopt105; retain079baseline. The pose-only mechanism trial is finished. Independent scene and full-background reconstruction remain unfinished.')
(O/'visual-review.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf8')
print('Recorded actual12file visual inspection scope.')
