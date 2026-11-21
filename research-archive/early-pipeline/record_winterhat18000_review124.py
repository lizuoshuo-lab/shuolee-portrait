from pathlib import Path
from datetime import datetime,timezone
import json,shutil
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
ROOT=T/'shuolee/current/local-clothed-surface-123/cross-subject-sh0-124'
OUT=ROOT/'winterhat-selected-sh0-pipeline124'
R=T/'shuolee/current/robust-portrait-122';C=H.parent/'shuolee-coordination'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
now=datetime.now(timezone.utc).isoformat();snap=OUT/'gaussian-run/step-018000'
run=read(OUT/'gaussian-run/result.json');geometry=read(OUT/'gaussian-run/geometry-progress.json')
assert read(OUT/'process-exit.json')['exit_code']==0 and geometry['step']==18000
review=dict(updated=now,step=18000,actual_visual_review=True,
    overview_body_face_feet_reviewed=[13,16,41,56,57,71,96,109],
    native_crops_reviewed=['frame_000071 face','frame_000013 body'],
    novel_views_reviewed=list(range(36)),
    finding='已审阅全部8个训练视角的原图／当前mesh／高斯，以及36个连续旋转视角。18,000步时脸部、帽子花纹和衣物比1,000步清楚，裙摆和靴子仍偏糊；未拍背面出现明显破碎、空洞，不能当成可靠表面。当前是SH0，背面破碎不能归因于高阶球谐方向颜色。网格共享偏移最大仅0.000512米，服装像素拟合不等于网格已外扩到衣物；源拟合的手臂深度与脚部对齐问题仍保留。本轮可作为完整流程实验查看，质量未通过。',
    geometry=geometry,source_pose_quality_accepted=False,quality_accepted=False,
    training_complete=True,automatic_retry=False,novel_view_ground_truth=False,
    source_and_export_black_background_parity=True,gray_background_native_parity_tested=False)
save(snap/'visual-review.json',review)
save(OUT/'delivery/step-018000/visual-review.json',review)
for name in ['orbit_00-11.png','orbit_12-23.png','orbit_24-35.png']:
    dest=Path('D:/00/记录/001')/('分析_冬帽SH0流程124_018000_'+name)
    assert not dest.exists();shutil.copy2(OUT/'delivery/step-018000/novel'/name,dest)
entry=dict(root=str(OUT),reference_frame='frame_000071',training_frames=8,step_reviewed=18000,
    final_points=24861,training_seconds=run['seconds'],training_complete=True,
    actual_visual_review=True,novel_views_reviewed=36,quality_accepted=False,
    source_fit_quality_accepted=False,original_roster_and_pose_preserved=True,
    gaussian_from_scratch=True,face_protected=True,vhap_used=False,
    final_mesh_offset_max_m=geometry['max_added_offset_m'],
    bounded_budget=18000,original_scheduler_iterations=600000,
    cold_start_15min_accepted=False,point_count_200k_met=False,viewer_UI_verified=False)
grass=read(ROOT/'grass-selected-sh0-pipeline124/launch.json')
for p in [R/'protocol.json',R/'queue-state.json',R/'evidence-ledger.json',C/'local-research-status.json']:
    d=read(p);d.update(updated=now,current_stage='grass_selected_sh0_pipeline124_training_winterhat_delivery',
        winterhat_sh0_pipeline124=entry,training_running=True,own_gpu_task_running=True,
        current_worker_pid_wsl=grass['pid'],quality_accepted=False,goal_complete=False)
    if 'updated_at' in d:d['updated_at']=now
    for key in ['worker_pid','active_pid','pid']:
        if key in d:d[key]=grass['pid']
    for key in ['gaussian_training_running','local_GPU_task_active','GPU_task_running','running']:
        if key in d:d[key]=True
    if 'state' in d:d['state']='grass_selected_sh0_pipeline124_training'
    d['grass_selected_sh0_pipeline124']=dict(root=str(ROOT/'grass-selected-sh0-pipeline124'),
        run_pid=grass['pid'],proc_start_identity=grass['proc_start_identity'],training_frames=7,
        reference_frame='frame_000046',replay_max_abs_m=8.344650268554688e-7,
        quality_accepted=False,pose_refit=False,maximum_step=18000,
        record_image_scope='共享七帧124 filename prefix preserves previous grass evidence')
    d['next_action']='Review same grass process at1000 before continuing to18000; complete winterhat paired-model UI delivery. No repeated pose repair.'
    d['previous_goal_turn_audit']=dict(classification='progress',blocked_streak=0,
        evidence='Winterhat completed18000 with actual8-view and36-orbit review; paired native export passed. Grass fresh seven-view training launched; Remy00 failure scope corrected.')
    save(p,d)
print('WINTERHAT18000_ACTUAL_REVIEW_RECORDED_GRASS_RUNNING',grass['pid'])
