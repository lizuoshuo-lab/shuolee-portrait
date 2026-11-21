from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,re
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001');REC=T/'shuolee/current/local-clothed-surface-123';R=T/'shuolee/current/robust-portrait-122';REF=REC/'smplifyx-reference-control';O=REF/'corrected-reference';G=R/'video10-global-initializer'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat()
plan=REF/'solver-conditioning-plan.json'
if not plan.exists():
    inputs=[O/'slot031/stage0.npz',O/'slot062/stage0.npz',REF/'convergence-control/ground-truth.npz']
    save(plan,{'registered':stamp,'state':'registered_not_executed','purpose':'Explain observed poor line-search conditioning without a new real fit or calibration candidate.',
      'evidence':str(O/'failure-replay/result.json'),'input_hashes':{str(p):sha(p) for p in inputs},
      'cases':['Exact saved view31 stage0 endpoint,stage1 objective','Exact saved view62 stage0 endpoint,stage1 objective','Known synthetic target and stage1 objective'],
      'hypotheses':['Large prior-induced curvature disparity across jaw/body/shape variables destabilizes a common unscaled step','Float32 objective rounding produces apparent zero progress before a large next trial','Wrong derivative would be a separate implementation defect, not a camera error'],
      'fixed':['completeK','semantic mapping','all raw target coordinates and explicit confidence','prior/data weights','original visibility caveats'],
      'read_only_checks':['Evaluate identical parameters and objective in FP32/FP64; verify values and finite gradients','Independent loss terms and analytical diagonal prior curvature per block','Directional derivative checks with predeclared steps1e-4,1e-5,1e-6; these are numerical checks, not fitted learning rates','Evaluate linear variable scaling derived from objective curvature; verify physical parameter/target identity and chain-rule equivalence before proposing any solver repair'],
      'budget':{'saved_states':3,'seconds_per_state':120,'objective_evaluations_per_state':200,'new_real_fits':0,'new_Gaussian_steps':0,'downloads':0},
      'stop':['No optimizer.step or parameter fitting in this diagnostic','Do not retune weights/visibility per view','Nonfinite values or exhausted numerical budget: retain failure, no automatic scan'],
      'old_geometry_use':'Saved endpoints are solely local diagnostic objects, not new fit or training initialization.',
      'next_if_evidence':'A generic feasible solver/reparameterization must first pass known-geometry controls; any further real reference requires a separate explicit bounded repair registration. No cross-experiment continuation.'})
save(O/'viewer-ui-review.json',{'updated':stamp,'tab_id':'32','url':'http://127.0.0.1:8870/diagnostics/video10-global-initializer/index.html#smplify-corrected-control',
 'actual_reload_and_AX_review':True,'visible_summary_and_tables_match_saved_outputs':True,'all12_raw_comparison_images_opened_separately':True,'tab15_untouched':True,
 'screenshot_obtained':False,'screenshot_limitation':'Browser screenshot reports Cannot take screenshot with 0 width; no bypass attempted. AX and local evidence-image inspection are separate.',
 'new_point_cloud':False,'OpenFile_gaussian_points_test_applicable':False,'quality_accepted':False,'company00_verified':False})
p=O/'publication.json';v=read(p);v.update(updated=stamp,actual_UI_review_pending=False,actual_UI_AX_verified=True,browser_screenshot_available=False,next_plan=str(plan));save(p,v)
for folder in [REC,R]:
    p=folder/'queue-state.json';v=read(p);v.update(next_plan=str(plan),worker_pid=None,worker_command=None,training_running=False);save(p,v)
p=REC/'next-action.json';v=read(p);v['next_plan']=str(plan);save(p,v)
notice='<p><strong>历史记录：</strong>以下为修复前的无效试跑；其中“尚未重跑”是当时状态。修复后结果见上方“SMPLify-X：对应关系能用，拟合流程仍需修复”。</p>'
p=G/'index.html';s=p.read_text(encoding='utf-8');needle='<section id="smplify-reference-control">'
if notice not in s:s=s.replace(needle,needle+notice,1);p.write_text(s,encoding='utf-8')
for dest in [G,T/'viewer/diagnostics/video10-global-initializer',Path('D:/00/工作/查看器/diagnostics/video10-global-initializer'),Path('D:/00/工作日志/2026-10-29/人物对齐核验123/相机多解与人物对齐')]:
    if dest!=G:shutil.copy2(p,dest/'index.html')
    shutil.copy2(plan,dest/'smplifyx-reference-control'/plan.name)
    for f in ['viewer-ui-review.json','publication.json']:shutil.copy2(O/f,dest/'smplifyx-reference-control/corrected-reference'/f)
print(json.dumps({'next_plan':str(plan),'goal_complete':False,'new_training':False},ensure_ascii=False))
