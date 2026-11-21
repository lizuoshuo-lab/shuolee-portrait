from pathlib import Path
from datetime import datetime, timezone
import json

H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122'
OUT=T/'shuolee/current/local-clothed-surface-123/cross-subject-sh0-124/winterhat-selected-sh0-pipeline124'
snap=OUT/'gaussian-run/step-001000'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
now=datetime.now(timezone.utc).isoformat()
assert not (OUT/'review-1000-decision.json').exists()
request=read(OUT/'review-1000-request.json')
geometry=read(OUT/'gaussian-run/geometry-progress.json')
assert geometry['step']==1000 and geometry['new_relative_normal_reversals']==0
review=dict(updated=now,step=1000,actual_visual_review=True,
    overview_body_face_feet_reviewed=[13,16,41,56,57,71,96,109],
    native_crops_reviewed=['frame_000071 face','frame_000013 body','frame_000096 feet'],
    finding='All eight trained views show the hat pattern, blue jacket and yellow skirt. Face remains speckled, skirt and dark boots are coarse. The source-fit arm-depth and foot-alignment uncertainties remain; no new mesh fold or gross training divergence appears. Continue this same optimizer to the registered18000, not a pose refit or quality promotion.',
    geometry=geometry,protected_face_offset_zero=True,quality_accepted=False,
    source_pose_quality_accepted=False,novel_views_reviewed=False,
    decision='continue_same_live_optimizer_to_18000')
save(snap/'visual-review.json',review)
save(OUT/'review-1000-decision.json',dict(updated=now,
    world_ply_sha256=request['world_ply_sha256'],decision='continue',
    review=str(snap/'visual-review.json'),same_live_optimizer=True,maximum_step=18000))
entry=dict(root=str(OUT),reference_frame='frame_000071',training_frames=8,
    prepared_interface_original_forward_max_m=7.152557373046875e-7,
    step_reviewed=1000,points_at_1000=20908,seconds_to_1000=66.85458975,
    actual_visual_review=True,quality_accepted=False,source_fit_quality_accepted=False,
    original_roster_and_pose_preserved=True,gaussian_from_scratch=True,
    run_pid=288,proc_start_identity='2532',comparison_budget=18000,
    current_scope='Latest user authorized reproducible exploratory GS before further pose repairs. Historical pose rejection remains unchanged.',
    observation_reused=True,cold_start_15min_accepted=False)
for p in [R/'protocol.json',R/'queue-state.json',R/'evidence-ledger.json',H.parent/'shuolee-coordination/local-research-status.json']:
    d=read(p);d['updated']=now
    if 'updated_at' in d:d['updated_at']=now
    d['current_stage']='winterhat_sh0_pipeline124_training_after1000_review'
    d['winterhat_sh0_pipeline124']=entry
    d['current_worker_pid_wsl']=288
    d['training_running']=True;d['own_gpu_task_running']=True;d['quality_accepted']=False;d['goal_complete']=False
    for k in ['worker_pid','active_pid','pid']:
        if k in d:d[k]=288
    for k in ['gaussian_training_running','local_GPU_task_active','GPU_task_running','running']:
        if k in d:d[k]=True
    if 'state' in d:d['state']='winterhat_sh0_pipeline124_training'
    d['previous_goal_turn_audit']=dict(classification='progress',evidence='Completed sofa18000 paired-model UI delivery, Remy local import, and reproduced/fixed WindowsZIP404. No external blocker.',blocked_streak=0)
    d['next_action']='Observe the same live winterhat process, review10000/final, then export paired model/camera. Prepare indoor selected input without concurrent GPU use. No automatic repeat or pose-only repair loop.'
    save(p,d)
print('WINTERHAT_1000_ACTUAL_REVIEW_CONTINUE_SAME_OPTIMIZER')
