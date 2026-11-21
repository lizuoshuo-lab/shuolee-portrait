"""Summarize paired changes without converting aggregate scores into acceptance."""
from pathlib import Path
import json, hashlib
import numpy as np
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-actor23-geodesic-107'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
audit=read(O/'matched-training-audit.json');assert audit['all_training_complete'] and audit['initial_and_sampling_match']
c=read(O/'comparison/summary.json');t=read(O/'continuity/summary.json');lo=read(O/'lower-body-review/summary.json');assert c['PNG_checks']==1584 and t['PNG_checks']==300
rows=read(O/'comparison/records.json')['records'];detail=read(O/'comparison/detail-metrics.json')['records'];B=['path_rigidity','path_rigidity_binding'];changes={}
for split in sorted({r['split'] for r in rows}):
    rr=[r for r in rows if r['split']==split];changes[split]={}
    for b in B:
        values=np.array([r['change_vs_baseline_db'][b] for r in rr]);worst=sorted([dict(camera=r['camera'],frame=r['frame'],change_db=r['change_vs_baseline_db'][b]) for r in rr],key=lambda x:x['change_db'])[:3]
        changes[split][b]=dict(mean_change_db=float(values.mean()),median_change_db=float(np.median(values)),improved=int((values>1e-6).sum()),regressed=int((values< -1e-6).sum()),images=len(rr),worst=worst)
out=dict(status='training_and_native_numeric_audits_complete_visual_review_pending',training=audit,comparison={k:v for k,v in c.items() if k!='panels'},change_vs_baseline=changes,continuity={k:v for k,v in t.items() if k!='panels'},lower_body_mean_psnr_db=lo['mean_psnr_db'],visual_review_required=True,clarity_accepted=False,adopted_as_clear_human_result=False,scope='One seed per configuration on inspected development views; the source-defined native windows and all baseline observations are unchanged. Initial graph connectivity is from a silhouette shell, not true anatomical mesh connectivity. No independent geographic/long-motion confirmation and no joint background.',remaining=['Inspect fingers, raised foot and regression cases at native scale.','Longer motion and occlusion confirmation.','Static background/person joint reconstruction and OmniRe integration.'],goal_complete=False)
save(O/'numeric-summary.json',out)
print(json.dumps(dict(changes=changes,details=c['detail_summary'],lower_body=lo['mean_psnr_db'],continuity=t['temporal_difference_residual']),indent=2))
