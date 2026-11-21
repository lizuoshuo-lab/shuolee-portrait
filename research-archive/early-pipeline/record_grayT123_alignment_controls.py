"""Record completed near-scene and localization controls without model promotion."""
from pathlib import Path
from datetime import datetime, timezone
import shutil, json
from audit_grayT123_camera_support import REC, ROOT, read, save, sha

H=Path(__file__).parent;T=REC.parents[2];C=H.parent/'shuolee-coordination'
L=Path('D:/00/工作日志/2026-10-28/人物自由表面训练123')
stamp=datetime.now(timezone.utc).isoformat()
near=read(REC/'grayT-near-scene-support-review/result.json')
photo=read(REC/'grayT-skin-localization-review/result.json')
geom=read(REC/'grayT-skin-localization-review/geometry-result.json')
assert near['road_candidate_target_checks']['n']==18 and geom['held_check']['n']==35
assert geom['held_check']['ge10px']==35 and not photo['failed_frames']
assert all(sha(Path(s['path']))==s['sha256'] for s in geom['sources'])
assert not (ROOT/'gaussian-run').exists()
summary=('SMPL-X仍未对齐：灰衣参与拟合的面部检测残差中位43.99像素。原几何门槛仅检查全身，允许身体中位误差约50–53像素，不能用于精细人脸验收。'
    '本轮补查6条原留出场景轨迹的全部110个节点：5条路面候选在同帧58/60/62/65共18次检查，中位0.687、最大2.446像素，2次超过1像素；另一条为远处树叶，不认证静止。'
    '最接近皮肤点估计深度的路面轨迹也有2次超过1像素，而且全部可信候选仍在地面，没有靠近头部高度的非共面固定结构，不能宣布相机亚像素通过。'
    '另外固定全部相机和原41帧，仅将皮肤色素点定位改为13px原生亮度模型。定位相对原光流最大修正2.910像素，全部41点实际审阅未见跨材料跳接。'
    '仍用原6锚帧拟合一个静态三维点后，35个原留出视角中位18.833、最大33.098像素，全部大于10像素；原光流对应中位18.925像素。局部定位修正未消除大错位。'
    '结论：当前相机、材料观测与静态人像假设仍不一致，不能只调整SMPL-X体型或继续高斯迭代。精确K/外参误差与人物局部运动尚未唯一分离。亮度模型的条件置信区间不是物理真值。灰衣高斯未启动，原点云和相机保留。')
con=dict(updated=stamp,subject='grayT',summary=summary,
    near_scene_result=str(REC/'grayT-near-scene-support-review/result.json'),
    localization_result=str(REC/'grayT-skin-localization-review/result.json'),
    localized_static_result=str(REC/'grayT-skin-localization-review/geometry-result.json'),
    near_scene_target_checks=near['road_candidate_target_checks'],skin_held_check=geom['held_check'],
    same_frame_controls=geom['matched_frame_scene_controls'],same_frame_road_errors=near['same_frame_road_errors'],
    original_skin_held_median_px=18.925094,localization_max_change_px=max(o['displacement_px'] for o in photo['observations']),
    all_near_scene110_nodes_and_skin41_nodes_actually_reviewed=True,all_four_new_projection_panels_actually_reviewed=True,
    parameter_changes=['Only one registered local photometric observation model; one3-variable diagnostic point refit with original6anchors'],
    original_cameras_and_SMPLX_unchanged=True,no_held_point_used_for_fitting=True,
    physical_subpixel_truth_certified=False,unique_physical_root_cause_identified=False,
    training_started=False,new_model_published=False,quality_accepted=False)
save(REC/'alignment-control-conclusion.json',con)
save(REC/'grayT-near-scene-support-review/post-geometry-review.json',dict(created=stamp,actual_visual_review=True,
    summary='6条候选全110节点已实际审阅。5条路面候选18个同帧检查：中位0.687px，最大2.446px，2次≥1px。另一条为远处树叶，保留其误差但不认证静止。路面近似共面，仍缺头部高度附近非共面固定结构。',
    no_error_based_deletion=True,camera_precision_certified=False,quality_accepted=False))
save(REC/'grayT-skin-localization-review/post-geometry-review.json',dict(created=stamp,actual_visual_review=True,
    all41_localizations_and_four_new_projection_panels_reviewed=True,summary=summary,
    synthetic_test_only_validates_implementation=True,physical_precision_certified=False,quality_accepted=False))
for name,output in [('grayT-near-scene-support',REC/'grayT-near-scene-support-review/result.json'),
                    ('grayT-skin-localization',REC/'grayT-skin-localization-review/geometry-result.json')]:
    save(REC/(name+'-completion.json'),dict(completed=stamp,status='completed_no_repeat',
        immutable_plan=str(REC/(name+'-plan.json')),result=str(output),quality_accepted=False))
nextplan=dict(created=stamp,status='planned_not_executed',name='Separate fine alignment admission from the coarse SMPL-X initialization gate',
    scope='Local123 control code only; no camera, fit, data or company113 installer change. No training launch.',
    defect='The existing coarse body gate explicitly reports quality=false, but subsequent manual finite-research continuation can enter surface initialization/training without a separate fine alignment receipt. This does not provide the strict usergoal admission.',
    change='Require a separately hash-bound fine alignment decision before any new surface initialization or training in the future successor queue. Keep coarse geometry report and historical research decisions unchanged. Missing/failed/incomplete physical material validation must deny formal training.',
    required_evidence=['All preregistered training frames registered, with exact current camera and input hashes.',
        'A frozen complete independent physical-material HELD roster; per-observation native Euclidean errors strictly<1px, localization uncertainty and eye/nose/mouth/view coverage actually reviewed. Detector errors and photometric Hessians alone cannot supply this evidence.',
        'Current fit hash plus actual fullbody, feet/occlusion and background protection review, with no silently dropped failures.'],
    validation='Read-only rejection against current grayT, whiteT, winterhat artifacts; synthetic positive and negative contract tests only, explicitly not realdata admission. Preserve original files and ensure no launch occurs.',
    bounds=dict(code_change_sets=1,new_training_runs=0,new_camera_candidates=0,GPU_tasks=0),
    next_geometry_question='Near-head noncoplanar scene support and multiple independent rigid skin materials are not yet established. A camera-versus-head-motion intervention must be separately specified before fitting; do not rerun completed controls.')
save(REC/'alignment-admission-implementation-plan.json',nextplan)
ledger=read(REC/'evidence-ledger.json');kind='grayT_near_scene_and_localization_controls_once'
assert not any(e.get('kind')==kind for e in ledger['entries'])
ledger['entries'].append(dict(kind=kind,date=stamp,evidence=con));ledger['updated']=stamp;save(REC/'evidence-ledger.json',ledger)
q=read(REC/'queue-state.json');q.update(updated=stamp,state='active_alignment_diagnosis',training_running=False,worker_pid=None,reason=summary)
save(REC/'queue-state.json',q)
n=read(REC/'next-action.json');n.update(updated=stamp,latest_diagnosis=str(REC/'alignment-control-conclusion.json'),
    latest_actual_checkpoint_review=str(REC/'grayT-skin-localization-review/post-geometry-review.json'),
    next_plan=str(REC/'alignment-admission-implementation-plan.json'),actual_UI_verified=False,
    next=['Do not repeat completed near-scene or skin-localization controls.','Implement the registered separate alignment-admission guard without starting the oldqueue or changing oldmodels.'],
    current_worker_pid_wsl=None,publication_worker_pid_windows=None)
save(REC/'next-action.json',n)
g=read(REC.parent/'robust-portrait-122/queue-state.json');g.update(updated=stamp,
    phase='LOCAL_SURFACE123_near_scene_and_observation_controls_complete',running=None,
    last_goal_turn=dict(classification='progress',current=summary,quality_accepted=False),next=nextplan['change'])
save(REC.parent/'robust-portrait-122/queue-state.json',g)
state=read(C/'local-research-status.json');state.update(updated_at=stamp,current_local_alignment_control_diagnosis=con,
    local_GPU_task_active=False,training_running=False,next_plan=str(REC/'alignment-admission-implementation-plan.json'))
save(C/'local-research-status.json',state)
with (T/'shuolee/long-term-plan.md').open('a',encoding='utf-8') as f:
    f.write('\n\n2026-10-28：同帧近景与皮肤定位独立控制\n\n'+summary+'\n\n下一步：'+nextplan['change']+'\n')
L.mkdir(parents=True,exist_ok=True)
for name in ['alignment-control-conclusion.json','alignment-admission-implementation-plan.json',
             'grayT-near-scene-support-completion.json','grayT-skin-localization-completion.json']:
    shutil.copy2(REC/name,L/name)
print(json.dumps(dict(conclusion=str(REC/'alignment-control-conclusion.json'),next_plan=str(REC/'alignment-admission-implementation-plan.json'),new_model=False),ensure_ascii=False))
