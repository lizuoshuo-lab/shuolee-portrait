from pathlib import Path
from datetime import datetime,timezone
import json
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';REF=REC/'smplifyx-reference-control';stamp=datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
d=read(REF/'solver-conditioning/result.json')
summary=[]
for v in d['cases']:
    f64=v['results']['torch.float64'];fd=max(x['samples'][1]['relative_difference'] for x in f64['directional_checks'])
    summary.append({'case':v['case'],'fp64_directional_relative_max_h1e5':fd,'projection_precision_difference_max_px':v['fp32_fp64_projection_max_px'],'gradient_precision_difference_max':v['fp32_fp64_gradient_max_abs_difference'],'loss_formula_abs_difference':f64['formula_abs_difference']})
save(REF/'solver-conditioning/assessment.json',{'updated':stamp,'state':'read_only_numerical_diagnosis_complete','cases':summary,'prior_curvature_ratio_jawmax_to_shape':3264320000/5000,
 'findings':['FP64 directional derivatives agree with central differences. FP32 differences at these small steps are affected by objective rounding; that discrepancy is not proof that autograd is wrong.','Jaw/shape prior diagonal curvature differs by652864; this is block-prior disparity, not the full Hessian condition number.','Output projection difference is below0.00027px; precision alone does not directly explain many-pixel image misalignment.','Exact same physical objective has verified linear scaling chain rule.'],
 'historical_ghosting_uniquely_attributed':False,'old_saved_states_used_only_for_diagnosis':True,'gaussian_steps':0})
current=read(REF/'conditioned-reference/progress.json')
for folder in [REC,T/'shuolee/current/robust-portrait-122']:
    q=read(folder/'queue-state.json');q.update(updated=stamp,current_stage='smplify_conditioned_real_diagnostic_running',worker_pid=current.get('pid'),worker_command='fit_smplify_conditioned123.py run',training_running=False,training_started=False,gaussian_steps=0,next_plan=str(REF/'conditioned-reference/plan.json'),next='Review the once-only three-view numerical repair. Synthetic all-group nonregression remains failed; no Gaussian admission.');save(folder/'queue-state.json',q)
print(json.dumps({'assessment_saved':True,'pid':current.get('pid')},ensure_ascii=False))
