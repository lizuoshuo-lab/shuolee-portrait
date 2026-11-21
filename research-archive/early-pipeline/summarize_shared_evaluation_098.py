"""Summarize the fixed comparison without replacing the full per-view evidence."""
from pathlib import Path
import json
import numpy as np

R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/person-static-shared-face-fresh-098')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
e=read(R/'evaluation/result.json')
groups={'all_available':(17,68),'eyebrows':(17,27),'nose':(27,36),'eyes':(36,48),'mouth':(48,68)}
consistency={}
for name,(lo,hi) in groups.items():
    rows=[r for r in e['same_feature_3d_consistency'] if lo<=r['id']<hi and r['common_visible_cameras']>=6]
    record={'features':len(rows),'minimum_common_cameras':min(r['common_visible_cameras'] for r in rows)}
    for metric in ('scatter_rms_world','shared_target_rms_world'):
        record[metric]={arm:float(np.mean([r[arm][metric] for r in rows])) for arm in ('control','shared')}
        record[metric]['relative_change_percent']=100*(record[metric]['shared']/record[metric]['control']-1)
    consistency[name]=record
out=dict(summary=e['summary'],same_feature_3d_consistency=consistency,
         independent_transfer={name:{'common':r['common'],'control':r['errors_full_px']['control'],
                                      'shared':r['errors_full_px']['shared']} for name,r in e['independent_transfer'].items()},
         boundaries=e['boundaries'],
         interpretation='Shared target residuals are training-target diagnostics. Independent texture correspondences and visible body/face rendering must be checked separately. World units are the reconstruction scale, not certified metric units.',
         quality_accepted=False)
(R/'evaluation/compact-summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,ensure_ascii=False))
