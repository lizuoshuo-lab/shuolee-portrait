from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2]/'outputs/dynamic-human-enerf-fine-graph-071'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
k=read(R/'paired-comparison/summary.json');h=read(R/'heldout/paired-comparison/summary.json')
d={region:float(np.mean([x['delta_db'] for x in k['details'] if x['region']==region])) for region in sorted(set(x['region'] for x in k['details']))}
summary=dict(status='metrics_complete_visual_review_pending',known=k['validation_foreground'],heldout=h['validation_foreground'],known_per_camera=k['per_camera_validation_delta_db'],heldout_per_camera=h['per_camera'],details_delta_db=d,known_worst={key:k['worst_change'][key] for key in ['camera','frame','foreground_delta_db']},heldout_worst=h['worst_change'],training=read(R/'residual_graph/completion.json'),goal_complete=False)
(R/'metrics-summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
print(json.dumps(summary))
