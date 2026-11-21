"""Publish engineering progress and a bounded next causal diagnostic."""
from pathlib import Path
from datetime import datetime,timezone
import json,shutil
from audit_grayT123_camera_support import REC,ROOT,read,save,sha

H=Path(__file__).parent;T=REC.parents[2];C=H.parent/'shuolee-coordination'
L=Path('D:/00/工作日志/2026-10-28/人物自由表面训练123');stamp=datetime.now(timezone.utc).isoformat()
out=REC/'alignment-admission-engineering'
test=read(out/'test-result-v4.json');native_result=read(out/'native-rejection-final.json')
assert test['all_contract_cases_passed'] and test['contract_cases']==27
assert len(test['entry_boundary_tests'])==4 and len(native_result['results'])==6
assert sha(H/'local_surface123_alignment_gate.py')==test['guard_source_sha256']==native_result['source_sha256']
assert sha(H/'local_surface123.py')==test['orchestrator_source_sha256']
assert not (ROOT/'gaussian-run').exists()
summary=('本地对齐准入已与粗SMPL-X初始化分开：拟合前检查相机与独立材料证据，表面初始化前、训练前和1000步复核检查当前拟合与同一相机/数据。'
    '检查绑定预登记留出名册、原生图像及mask、相机、拟合文件和实际审阅记录，重新计算全部面部留出误差。任一点达到1像素、名册遗漏或事后改变、FIT/HELD泄漏、缺少眼鼻嘴/视角或全身背景保护、仅有光流往返或亮度Hessian精度声明，均不能放行。'
    '27项合成合同检查与4个入口拦截测试通过；Windows与WSL现有环境均实际拒绝grayT、whiteT、winterhat缺失的精细准入。合成证据永远不能授权真实拟合或训练。'
    '这是错误放行路径的修复，不是重影修复。当前灰衣35个皮肤留出视角中位仍18.83像素，三个已拟合人物都未获新训练准入。原数据、相机、拟合和已发布点云没有改动，未启动训练。')
inventory_summary=('已实际审阅预登记40/54/66/80四帧的8幅原生头部和躯干裁片。眼外侧色素点可辨；额头、面颊和颈部还有少量候选斑点，但未确认它们在全部41帧的同一物理身份或刚性关系。'
    '灰色上衣以重复细织纹、领口和褶皱为主，没有在这8幅裁片中建立可作为刚体真值的多点躯干组；不同视角的手势和褶皱外观变化不能单独证明运动。'
    '头旁栏杆重复结构明显，40帧与后续视角的背景覆盖不同，未建立跨四帧唯一的近头部非共面场景控制。这个结论只覆盖已查看裁片，不宣称全视频不存在此类点。'
    '因此暂不能从现有单个皮肤材料点唯一估计真实头部六自由度运动。下一步仅做模型预测能力诊断：已有鼻部语义点作FIT代理，皮肤点保持独立验证；结果不能被解释为真实头部运动或正式相机标定。')
save(REC/'grayT-relative-motion-support-review/actual-review.json',dict(created=stamp,actual_visual_review=True,
    all8_native_crops_actually_reviewed=True,summary=inventory_summary,
    new_physical_rigid_material_groups_certified=0,global_video_absence_claim=False,
    unique_head_motion_identifiable=False,quality_accepted=False))
plan=dict(created=stamp,status='registered_not_executed',name='Predictive head-rigid nuisance diagnostic with independent skin validation',
    question='With original sceneK/E and fixed fitted head shape, can per-frame rigid head nuisance transforms estimated only from existing nose proxies predict the independent skin mark substantially better? A positive result still does not uniquely distinguish real motion from local calibration/template error.',
    causal_variable='A single shared rigid head correction versus one rigid correction per frame. Same frozen head geometry, same9nose proxy observations, same original weights, K/E,41frames, objective and optimizer settings. No body/skin/scene observation enters pose estimation.',
    frames=list(range(40,81)),landmark_mapping=dict(IBUG=list(range(27,36)),face51_slots=list(range(10,19)),RTMW_slots=list(range(50,59))),
    sources=dict(fit=str(ROOT/'fit/fitted.npz'),weights=str(ROOT/'fit/observation-weights.npz'),observations=str(ROOT/'data/observations.npz'),
        inputs=str(ROOT/'data/training-inputs.json'),scene_cameras=str(ROOT/'camera/solver-output.npz'),
        skin_observations=str(REC/'grayT-skin-localization-review/result.json'),
        static_skin_baseline=str(REC/'grayT-skin-localization-review/geometry-result.json'),
        existing_scene_controls=str(REC/'grayT-near-scene-support-review/result.json')),
    preparation=['Verify frame/slot identities and reconstructed SMPL-X barycentric nose projection against saved projection(<0.001px). Old geometry is only the frozen diagnostic object, never a newtraining initializer.',
        'Show all41native nose crops with their proxy observations and fixed existing weights before optimizing. Flag occluded/inferred semantic points without filtering by error. They are FITproxies, not physical truth.',
        'Report3Dnose rank, per-frame positive-weight count and Jacobian singular values. If fewer than6positive-weight observations or numerical rank<6 in anyframe, stop the complete41frame pose comparison; preserve the deficiency and do not interpolate.'],
    fit=dict(shape_fixed=True,camera_fixed=True,pose_parameterization='3rotation-vector +3translation around fixed nose centroid; translation normalized by fixed noseRMSradius',
        initialization='identity world correction relative to the existing originalscene camera for eachframe; one deterministic start',
        objective='sqrt(original face weight)*native2Dresidual; soft_l1 f_scale6px, unchanged acrossstatic/perframeconditions',
        static_global_max_nfev=150,per_frame_max_nfev=100,skin_point_max_nfev=100,
        synthetic_test='Known per-frame rigid head transforms must be recovered predictively before touching actual optimization; test projection/SE3 composition and fixedreference40gauge.'),
    held_validation=dict(skin_anchors=[40,48,56,64,72,80],skin_checks=[i for i in range(40,81) if i not in [40,48,56,64,72,80]],
        method='Only afterFITposeoutputsarefrozen, triangulate one canonical skin point from the same6anchors and evaluate all35checks. Originalstatic skinbaseline18.832509px median is reused, not reoptimized. No skin coordinate participates in pose fitting.',
        background_counterfactual='For unchanged originalHELD3Dscene controls inframes58/60/62/65, report oldprojection versus E_i H_i inv(H_40). This anchors camera40 and tests treating the head-local correction as a whole-scene camera correction; it is not a calibrated camera candidate. Keep vegetation labeledunverified and everyerror.'),
    budget=dict(runs=1,CPU_seconds=120,shared_rigid_fits=1,per_frame_pose_fits=41,new_feature_extractions=0,new_camera_candidates=0,training_runs=0),
    stop_conditions=['No weight/landmark/window/threshold sweep, no alternateinitializationretry, no errorbasedremoval.',
        'Any nonfinite/negative-depth finalfit, incompleteframecoverage or numericalfailure means no completehypothesis acceptance; retain alloutputs.',
        'If heldskin remains>=1px anywhere, or physicaluncertainty/coverage remainsunverified, formaltraining remainsdenied. Even all35passing onepoint is not wholeface acceptance.',
        'Do not apply transforms to trainingimages, models, cameras or publishedPLY. This is not a dynamicnetwork or a newmotionmainline.'],
    interpretation_limits=['Nose semantic proxies may be occluded/imputed or deforming; fixedSMPL-Xnose shape is approximate.',
        'A single physicalskintrack cannot uniquely identify6DoFheadmotion. Improvement would support predictiveusefulness of localizedrigidfreedom, not a uniquecause or productionfix.',
        'Allbackgroundcounterfactuals keep old3Dpoints fixed and onlytestthatparticularcamera interpretation; they cannot ruleout allK/Eorresidualdistortion alternatives.'],quality_accepted=False)
save(REC/'grayT-head-rigid-diagnostic-plan.json',plan)
con=dict(updated=stamp,summary=summary,engineering_completed=True,physical_alignment_fixed=False,
    tests=str(out/'test-result-v4.json'),native_runtime_validation=str(out/'native-rejection-final.json'),
    guard_source=str(H/'local_surface123_alignment_gate.py'),guard_sha256=test['guard_source_sha256'],
    orchestrator_source=str(H/'local_surface123.py'),orchestrator_sha256=test['orchestrator_source_sha256'],
    original_orchestrator_preserved=str(out/'orchestrator-before-guard.py'),
    current_subjects_rejected=['grayT','whiteT','winterhat'],training_started=False,quality_accepted=False,
    contract_does_not_prove_truth='Hashes and booleans cannot manufacture physical identity or staticity. The tool recomputes errors and verifies completeness; the registered measurements and actualindependentreview remain necessary.',
    inventory_summary=inventory_summary,next_plan=str(REC/'grayT-head-rigid-diagnostic-plan.json'))
save(REC/'alignment-admission-conclusion.json',con)
save(REC/'alignment-admission-implementation-completion.json',dict(completed=stamp,status='completed_no_training',
    original_plan=str(REC/'alignment-admission-implementation-plan.json'),result=str(REC/'alignment-admission-conclusion.json'),quality_accepted=False))
ledger=read(REC/'evidence-ledger.json');kind='alignment_admission_implemented_and_native_denials'
assert not any(e.get('kind')==kind for e in ledger['entries'])
ledger['entries'].append(dict(kind=kind,date=stamp,evidence=con));ledger['updated']=stamp;save(REC/'evidence-ledger.json',ledger)
n=read(REC/'next-action.json');n.update(updated=stamp,latest_diagnosis=str(REC/'alignment-admission-conclusion.json'),
    next_plan=con['next_plan'],actual_UI_verified=False,next=['Read and execute the registered head-rigid diagnostic preparation once. This is a diagnosticproxy test, not a camera ortrainingcandidate.',
        'Currentguardchecksandnear-scene/localizationcontrolsarecomplete; do not repeat them orrestart the oldqueue.'],current_worker_pid_wsl=None,publication_worker_pid_windows=None)
save(REC/'next-action.json',n)
q=read(REC/'queue-state.json');q.update(updated=stamp,state='active_alignment_diagnosis',training_running=False,worker_pid=None,reason=summary);save(REC/'queue-state.json',q)
g=read(REC.parent/'robust-portrait-122/queue-state.json');g.update(updated=stamp,phase='LOCAL_SURFACE123_alignment_admission_implemented',running=None,
    last_goal_turn=dict(classification='progress',current=summary,quality_accepted=False),next=plan['question']);save(REC.parent/'robust-portrait-122/queue-state.json',g)
state=read(C/'local-research-status.json');state.update(updated_at=stamp,current_local_alignment_admission=con,local_GPU_task_active=False,
    training_running=False,next_plan=con['next_plan']);save(C/'local-research-status.json',state)
with (T/'shuolee/long-term-plan.md').open('a',encoding='utf-8') as f:
    f.write('\n\n2026-10-28：精细对齐准入与相对运动支持核查\n\n'+summary+'\n\n'+inventory_summary+'\n\n下一步：'+plan['question']+'\n')
for name in ['alignment-admission-conclusion.json','alignment-admission-implementation-completion.json','grayT-head-rigid-diagnostic-plan.json']:
    shutil.copy2(REC/name,L/name)
for name in ['test-result-v4.json','native-rejection-final.json']:shutil.copy2(out/name,L/name)
print(json.dumps(dict(guard_completed=True,physical_alignment_fixed=False,training_started=False,next_plan=con['next_plan']),ensure_ascii=False))
