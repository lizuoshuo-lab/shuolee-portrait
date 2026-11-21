"""Publish reviewed material diagnostics, with no model or camera replacement."""
from pathlib import Path
from datetime import datetime,timezone
import json,shutil
from audit_grayT123_camera_support import REC,ROOT,read,save,sha

stamp=datetime.now(timezone.utc).isoformat();H=Path(__file__).parent;T=REC.parents[2];C=H.parent/'shuolee-coordination'
skin=REC/'grayT-skin-texture-probe';old=REC/'grayT-material-static-check'
r=read(skin/'geometry-result.json');s=read(old/'result.json');vr=read(REC/'grayT-material-identity-review/actual-review.json')
assert len(r['matched_frame_scene_controls'])==4 and r['held_check']['n']==35 and r['held_check']['ge10px']==35
summary=('灰衣SMPL-X面部拟合残差中位约44像素，原粗门槛仅检查全身，未完成精细人脸验收。'
    '25条FIT和3条HELD脸区候选的全部279个观测已逐一查看，均为眼睑/眼裂、唇齿嘴角或轮廓附近，未认证为静止皮肤材料。'
    '原3条留出候选12次检查有11次超过1像素，最大27.88像素。'
    '新增一次皮肤色素点诊断：在原生帧40–80追踪同一可见深色皮肤标记，冻结全部相机，用预登记6帧只拟合一个静态三维点。'
    '其余35帧全部超过10像素，中位18.93、最大32.35像素。'
    '同一帧58/60/62/65中，皮肤点分别17.12/13.43/11.74/13.69像素，现有地面留出点分别0.96/0.16/0.44/0.69像素。'
    '这表明当前观测、相机与静态人像假设存在不一致，不能只靠调整SMPL-X体型解决。'
    '边界必须保留：皮肤跟踪仍可能有数像素中心漂移，0.452像素光流往返误差不等于物理定位精度；地面控制只有一个材料候选、4帧，位置与深度不同。'
    '当前证据不能唯一排除K/外参/残余畸变，也不能直接判定全部由头部微动导致。灰衣高斯仍未开始，旧模型保留。')
save(old/'actual-review.json',dict(created=stamp,actual_visual_review=True,all12_projection_checks_actually_reviewed=True,
    summary='原3条全脸区留出候选共12次检查，11次≥1px，最大27.88px；全部冻结相机与既有三维点投影。候选为嘴角/唇缘与鼻梁轮廓，物理静止性未认证，不能作为亚像素真值。',
    fit_or_held_changes=0,quality_accepted=False))
save(skin/'post-geometry-review.json',dict(created=stamp,actual_visual_review=True,
    skin_all41_native_and_locator_reviewed=True,skin_four_same_frame_projection_panels_reviewed=True,
    asphalt_all9_anchor_and_check_crops_reviewed=True,
    asphalt_note='All9markers remain in a similar asphalt aggregate pattern. The track is plausible, not a surveyed uniquely identified reference. First2crops include black out-of-frame padding; original pixels were not warped.',
    summary=summary,camera_accuracy_certified=False,motion_uniquely_identified=False,quality_accepted=False))
idx=read(skin/'index.json');idx['images'].append(dict(frame='frame57,58,59,60,61,62,63,65,69',
    region='同帧地面控制的全部原始节点，含原锚点与原留出；首两幅黑区为画面外',file='asphalt_all_nodes_locator.png',raw='asphalt_all_nodes_native.png',columns=['native128px, chronological']))
save(skin/'index.json',idx)
con=dict(updated=stamp,subject='grayT',summary=summary,previous= str(REC/'camera-support-conclusion.json'),
    material_identity_review=str(REC/'grayT-material-identity-review/actual-review.json'),
    original_held_result=str(old/'result.json'),skin_result=str(skin/'geometry-result.json'),
    physical_subpixel_truth_certified=False,unique_physical_root_cause_identified=False,
    fit_face_median_px=43.992,skin_held_median_px=r['held_check']['median'],skin_held_max_px=r['held_check']['maximum'],
    skin_held_ge10px=35,skin_held_observations=35,same_frame_controls=r['matched_frame_scene_controls'],
    original_camera_and_geometry_unchanged=True,training_started=False,quality_accepted=False,
    diagnosis='A static shared human point is inconsistent under current frozen cameras/measurements. SMPL-X-only parameter correction cannot establish coherent multi-view geometry.',
    no_full_face_precision_or_unseen_video_claim=True)
save(REC/'material-consistency-conclusion.json',con)
next_plan=dict(created=stamp,status='planned_not_executed',name='Same-frame near-person scene support audit',
    causal_question='The current same-frame control is asphalt at about twice the diagnostic face-point depth and near the image edge. Is there physically plausible closer static-scene support at the face frames, without changing cameras or deleting high residuals?',
    inputs='Existing grayT appearance-tracks, frozen held-evaluation, native images, current point diagnostic. No new extraction or new camera candidate.',
    selection='At registered frames58/60/62/65, find HELD components whose whole track remains outside the original person box+20px. Rank by frame coverage, then closeness of saved positive depth to diagnostic skin depth, then lexicalID; no use of reprojection residual. Review at most6whole components with all nodes. Save candidate eligibility and selection before reading errors.',
    restrictions=['Saved depth is estimated and not physical survey truth. No same-frame available depth match is an explicit coverage gap.',
        'Never promote HELD into FIT or use error to delete a component. No geometric re-triangulation for controls.',
        'If physical identity/staticity cannot be supported, stop this bounded audit and report remaining ambiguity.',
        'Do not repeat skin tracking, change its thresholds or seed, or fit time-varying points to erase errors.',
        'Do not restart old123orchestrator or any grayT Gaussian training. Any future camera or head-alignment intervention needs a separately specified causal variable and independent protection.'],
    budget=dict(components_max=6,CPU_seconds=120,feature_extractions=0,optimizations=0,training_runs=0),
    goal_mode='active_not_timer')
save(REC/'grayT-near-scene-support-plan.json',next_plan)
follow=read(REC/'grayT-material-followup-plan.json');follow.update(status='completed_review_and_saved_projection_check',
    completed=stamp,physical_subpixel_acceptance=False,result=str(REC/'material-consistency-conclusion.json'))
save(REC/'grayT-material-followup-plan.json',follow)
ledger=read(REC/'evidence-ledger.json');kind='grayT_skin_material_static_consistency_once'
assert not any(e.get('kind')==kind for e in ledger['entries'])
ledger['entries'].append(dict(kind=kind,date=stamp,summary=summary,evidence=con));ledger['updated']=stamp;save(REC/'evidence-ledger.json',ledger)
n=read(REC/'next-action.json');n.update(updated=stamp,latest_diagnosis=str(REC/'material-consistency-conclusion.json'),
    latest_actual_checkpoint_review=str(skin/'post-geometry-review.json'),next_plan=str(REC/'grayT-near-scene-support-plan.json'),
    next=['Read grayT-near-scene-support-plan.json; completed skin/material checks must not be repeated.','Preserve all original models and cameras; grayT has no Gaussian training.'],
    current_worker_pid_wsl=None,publication_worker_pid_windows=None);save(REC/'next-action.json',n)
q=read(REC/'queue-state.json');q.update(updated=stamp,training_running=False,worker_pid=None,state='active_alignment_diagnosis',reason=summary)
for person in q['subjects']:
    if person['id'] in ['winterhat','whiteT']:person['stage']='published_research_quality_failed'
save(REC/'queue-state.json',q)
state=read(C/'local-research-status.json');state.update(updated_at=stamp,current_local_material_consistency_diagnosis=con,
    local_GPU_task_active=False,training_running=False,next_plan=str(REC/'grayT-near-scene-support-plan.json'));save(C/'local-research-status.json',state)
goal=read(REC.parent/'robust-portrait-122/queue-state.json');goal.update(updated=stamp,phase='LOCAL_SURFACE123_skin_material_static_inconsistency_measured',running=None,
    last_goal_turn=dict(classification='progress',current=summary,quality_accepted=False),next=next_plan['causal_question']);save(REC.parent/'robust-portrait-122/queue-state.json',goal)
with (T/'shuolee/long-term-plan.md').open('a',encoding='utf-8') as f:
    f.write('\n\n2026-10-28：灰衣真实皮肤标记静态一致性诊断\n\n'+summary+'\n\n下一步：'+next_plan['causal_question']+'\n')
L=Path('D:/00/工作日志/2026-10-28/人物自由表面训练123');L.mkdir(parents=True,exist_ok=True)
for name in ['material-consistency-conclusion.json','grayT-near-scene-support-plan.json']:shutil.copy2(REC/name,L/name)
print(json.dumps(con,ensure_ascii=False))
