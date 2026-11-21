"""Independently recompute stored normal components and matched experiment boundaries."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-enerf-open-hand-flow-083';M=O/'model-support'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out={};reports={}
for branch in ['photo_control','normal_flow']:
    R=O/branch;completion=read(R/'completion.json');assert completion['additional_graph_steps']==1500 and sha(R/'resume.pt')==completion['checkpoint_sha256']
    audit=read(R/'frozen-state-audit.json');assert audit['active_nodes']==27 and audit['inactive_nodes']==357 and audit['points']==252984
    errors=[];checks=[];maxdifference=0.
    for r in read(M/'completion.json')['records']:
        if not r['model_visible']:continue
        z=np.load(M/r['filename']);p=np.load(R/'flow-evaluation'/r['filename'])
        n=(p['flow'].astype(np.float64)*z['normal'].astype(np.float64)).sum(1)
        e=np.abs(n-z['normal_target'].astype(np.float64))
        delta=max(np.max(np.abs(n-p['normal']),initial=0),np.max(np.abs(e-p['absolute_error']),initial=0))
        assert delta<3e-6;maxdifference=max(maxdifference,float(delta));errors.extend(e.tolist())
        assert r['camera'] not in ['00','06','12','14','16'] and r['source_frame']%2==0 and r['target_frame']%2==0
        checks.append(dict(file=r['filename'],count=len(e),source_sha256=sha(M/r['filename']),prediction_sha256=sha(R/'flow-evaluation'/r['filename'])))
    assert len(errors)==1527 and len(checks)==43
    report=read(R/'normal-flow-evaluation.json');values=dict(mean_absolute_error_px=float(np.mean(errors)),median_px=float(np.median(errors)),p90_px=float(np.quantile(errors,.9)))
    for k,v in values.items():assert abs(v-report[k])<3e-6
    out[branch]=dict(**values,count=len(errors),pairs=len(checks),maximum_recomputation_difference_px=maxdifference,checks=checks,checkpoint_sha256=completion['checkpoint_sha256'])
    reports[branch]=read(R/'native-evaluation/metrics.json')
assert {(r['camera'],r['frame'],r['input_sha256']) for r in reports['photo_control']['records']}=={(r['camera'],r['frame'],r['input_sha256']) for r in reports['normal_flow']['records']}
a,b=[dict(np.load(O/br/'schedule.npz')) for br in out];assert a.keys()==b.keys() and all(np.array_equal(a[k],b[k]) for k in a)
c=read(O/'comparison/summary.json');assert c['PNG_checks']==480 and c['max_recomputed_metric_difference']<1e-9
detail=read(O/'comparison/detail-metrics.json')['records'];detailmeans={name:dict(count=sum(r['name']==name for r in detail),mean_psnr_change_db=float(np.mean([r['change_db'] for r in detail if r['name']==name]))) for name in sorted({r['name'] for r in detail})}
save(O/'numerical-audit.json',dict(status='matched_schedules_frozen_state_1527normal_residuals_120native_pairs_480PNGs_verified',branches=out,normal_error_reduction_fraction=1-out['normal_flow']['mean_absolute_error_px']/out['photo_control']['mean_absolute_error_px'],detail_windows=detailmeans,comparison_sha256=sha(O/'comparison/summary.json'),flow_scope='Training estimated normal-component consistency; not ground-truth flow or unseen-motion accuracy.',goal_complete=False))
shutil.copy2(__file__,O/Path(__file__).name)
print(json.dumps(dict(branches={b:{k:v for k,v in r.items() if k!='checks'} for b,r in out.items()},detail=detailmeans)))
