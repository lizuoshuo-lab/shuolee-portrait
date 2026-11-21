from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001');REC=T/'shuolee/current/local-clothed-surface-123';R=T/'shuolee/current/robust-portrait-122';O=REC/'smplifyx-reference-control'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat();result=read(O/'result.json')
assert sha(O/'runner-v1-invalid-score-contract.py')==result['source_sha256']
assessment={'updated':stamp,'original_numeric_result_preserved':True,'original_result_sha256':sha(O/'result.json'),'original_runner_sha256':result['source_sha256'],'valid_SMPLifyX_reference':False,'method_quality_accepted':False,'confirmed_defect':'Raw RTMW scores clipped as OpenPose probabilities, all116 mapped points weighted1. This new-adapter bug is not a claim about historical coarse-fit root cause.','repair_evidence':'score-contract-repair.json','corrected_real_fit_executed':False,'unknown':'Whether correcting score contract and optimizer convergence suffices for real images. Anatomical visibility remains unverified.','gaussian_steps':0,'goal_complete':False}
save(O/'assessment.json',assessment)
nextplan={'registered':stamp,'state':'registered_not_executed','purpose':'Isolate the optimizer repetition/convergence difference on known full SMPL-X geometry before any retry of invalid real reference.',
 'input_code':{str(H/'run_smplify_reference123.py'):sha(H/'run_smplify_reference123.py'),str(H/'smplify_score_contract123.py'):sha(H/'smplify_score_contract123.py')},
 'no_old_real_attempt_repeat':True,'score_policy':'Corrected explicit relative weight adapter; reject raw scores without conversion. Synthetic controls use known confidence and do not estimate visibility.',
 'causal_variable':'Only number/stopping semantics of optimizer.step calls perstage. Both arms keep same torchLBFGS implementation, fixedK, target, fresh synthetic initialization, loss and five upstream stage weights. This does not test vendorLBFGS implementation differences.',
 'cases':['One optimizer.step perstage as in invalid finite reference','Pinned upstream FittingMonitor repeated-step convergence logic with maxiters30 and innermaxiter30'],
 'synthetic_geometry':{'seed':12302,'views':1,'K':'Complete recorded fixedK2704x1520, no fit','ground_truth':'Neutral SMPL-X; VPoser latent0.8*sin(index+1), shape0.1*cos(index+1), handPCA0.2*sin(index+1) left and negatedright, jaw[0.02,0,0], expression0.05*cos(index+1), root[pi,0,0], translation[0,0,3.5]','initial_perturbation':'Latent+0.15*cos(index+1), root+[0.015,-0.02,0.01], translation+[0.03,-0.04,0.1]; hand,shape,jaw,expression startzero. Same fresh initialstate for both arms.'},
 'budget':{'arms':2,'max_gpu_seconds_per_arm':180,'max_closure_evaluations_per_arm':4000,'max_real_image_fits':0,'gaussian_steps':0,'new_downloads':0},
 'evidence':['Exact input hashes and identical fresh startingparameter hashes','Each stage parameters, loss components, stoppingreason and pergroup projected errors','Final gradient statistics, fullK preserved, finite depth','Rawpoint projections and objective independent formula check'],
 'stop':['Nonfinite loss/gradient or negative depth','Budget exhausted: report not converged, no automatic expansion','No retry with changed weights/thresholds/seed'],
 'admission_boundary':'Synthetic geometry tests implementation only. Before a corrected real reference, register separate one-time threeview repair with explicit visibility and no global-world averaging. No Gaussian training or camera changes from this control.'}
if not (O/'convergence-control-plan.json').exists():save(O/'convergence-control-plan.json',nextplan)
for folder in [REC,R]:
    p=folder/'queue-state.json';x=read(p);x.update(updated=stamp,current_stage='smplify_reference_invalid_score_contract_repaired_pending_convergence_control',worker_pid=None,worker_command=None,training_running=False,gaussian_steps=0,latest_diagnosis=str(O/'assessment.json'),next_plan=str(O/'convergence-control-plan.json'),valid_method_reference=False);save(p,x)
p=H.parent/'shuolee-coordination/local-research-status.json';x=read(p);x.update(updated_at=stamp,current_stage='smplify_reference_invalid_score_contract_repaired_pending_convergence_control',latest_evidence=str(O/'assessment.json'),next_plan=str(O/'convergence-control-plan.json'),valid_method_reference=False);save(p,x)
save(O/'viewer-ui-review.json',{'updated':stamp,'browser':'iab','tab':'32','url':'http://127.0.0.1:8870/diagnostics/video10-global-initializer/index.html#smplify-reference-control','actual_UI_refreshed':True,'AX_verified':['Score-interface failure explicitly shown','All three numeric rows visible','Invalid trial and no correction rerun stated','Threeviews body/face/bothhand figures exposed','No Gaussian training or newPLY'],'screenshot':'Unavailable: tab screenshot reports0width; no alternate capture used. Raw scientific images separately opened and reviewed.','tab15_untouched':True,'old_point_cloud_preserved':True,'company00_verified':False})
p=O/'publication.json';x=read(p);x.update(actual_UI_review_pending=False,actual_UI_AX_reviewed=True,invalid_reference_flagged=True,corrected_real_fit_run=False);save(p,x)
for dest in [R/'video10-global-initializer',T/'viewer/diagnostics/video10-global-initializer',Path('D:/00/工作/查看器/diagnostics/video10-global-initializer'),Path('D:/00/工作日志/2026-10-29/人物对齐核验123/相机多解与人物对齐')]:
    out=dest/'smplifyx-reference-control'
    for fn in ['assessment.json','score-contract-repair.json','convergence-control-plan.json','viewer-ui-review.json','publication.json']:shutil.copy2(O/fn,out/fn)
print(json.dumps({'old_invalid_result_preserved':True,'score_guard_verified':True,'next_plan_registered':True,'gaussian_steps':0}))
