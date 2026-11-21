"""Record the one completed head5 contrast and its actual visual rejection."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib

CURRENT=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current')
ROOT=CURRENT/'local-clothed-surface-123/cross-subject-sh0-124'
OUT=ROOT/'head-absence-input124';R=CURRENT/'robust-portrait-122'
def read(p):return json.loads(p.read_text('utf-8-sig'))
def save(p,d):
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n','utf-8');tmp.replace(p)
def local(p):
    if p.startswith('/mnt/'):return Path(p[5].upper()+':/'+p[7:])
    if p.startswith('/opt/'):return Path('//wsl.localhost/Ubuntu-20.04'+p)
    return Path(p)

assert not (OUT/'conclusion.json').exists()
r=read(OUT/'result.json');v=read(OUT/'visual-review.json');p=read(OUT/'plan.json')
assert r['fitting_calls']==1 and r['completed']==['frame_000040']
assert r['skipped_frames']==['frame_000041'] and read(OUT/'progress.json')['pid'] is None
for name,digest in p['frozen'].items():assert hashlib.sha256(local(name).read_bytes()).hexdigest()==digest
images=v['frames']['frame_000040']['images'];assert len(images)==4
for item in images.values():assert hashlib.sha256(local(item['path']).read_bytes()).hexdigest()==item['sha256']
stamp=datetime.now(timezone.utc).isoformat();record=r['records'][0]
summary='鼻眼耳5点的追加停用对照已完成并拒绝：第40帧只运行一次25.6秒，同初值、相机、体型和时间约束均核验一致。相对仅停用内脸51点，身体平均检测误差28.32→27.65px，但左肩19.01→22.58px、右肩10.30→15.26px，原生投影与四方向mesh仍不正确。按登记条件跳过41帧，不推广整段、不启动高斯。'
findings={
    'head':'四列原生头部对照已实看：停用完整头部观测后仍未贴合真实后脑，头部偏高，原有体型轮廓异常保留。没有独立真值可证明新朝向准确。',
    'body_and_hands':'原生身体、腰部手臂裁片已实看；可见肩部对齐没有通过保护，手在腰部的投影/深度仍不可靠。均值下降不能抵消逐关节退化。',
    'common_views':'原输入、内脸停用、头部停用三组均以相同四个合成相机审阅。头/身体比例与姿态问题仍存在；相机相对坐标中的倾斜本身不能当作重力或解剖错误。',
    'scope':'无脸检测支持不代表所有耳点均不可见；本次方案也移除了可见耳部信息。因此只能拒绝该追加停用政策，不能推出原始后脑特征点可信。',
    'metadata':'原fit_one报告描述上游基线；当前完整目标额外包含已登记时间项，不能把继承的upstream_loss_monitor_optimizer_unchanged字段当成无时间项。原求解器、基线损失、优化器和当前时间权重均未改变。'}
next_action=('The one head5 absence control completed and failed actual native/body/side review;1fit25.6s,41 skipped by registered protection. Do not rerun it, expand another face/head threshold policy, start indoor full111 or train GS. '
             'Preserve the temporal adapter and all rejected input controls. Indoor shared beta and unseen lower legs remain unaccepted; an additional head ablation is not a route to full-body acceptance. '
             'Return to the requested grass white-dress cross-person validation: its existing94-frame independent SMPLify-X baseline has a known44->45 depth flip, but no grass temporal candidate exists in cross-subject-sh0-124. '
             'Read grass/training-admission.json and grass/pose-cause-audit/temporal-gauge.json, plus original full94 source metadata and upstream runner. Verify chronological frame IDs, timestamps, original118 observations, fullK, reference selection and camera-relative versus world gauge before registering one fresh bounded temporal candidate. '
             'Reuse existing original solver and already tested temporal/seed interfaces; no detector rerun, accepted-looking old-pose stitching, custom full-sequence solver, or retuning per person. First preregister original settings and native pose/depth review stops, and do not launch a GS run until actual pose admission. '
             'The historical shared-shape63 paired control and front protection already failed; read its decision rather than repeating it or silently changing reference beta. Keep demo120k SH0 unchanged and the four-person goal active.')
v.update(updated=stamp,state='actually_reviewed_quality_failed',actual_visual_review=True,reviewed_images=4,
    findings=findings,quality_accepted=False,default_applied=False)
save(OUT/'visual-review.json',v)
conclusion=dict(updated=stamp,state='completed_rejected',summary=summary,next_action=next_action,
    images=images,findings=findings,actual_visual_review=True,fit_calls=1,
    fit_seconds=record['seconds'],total_seconds=r['seconds'],skipped_frames=r['skipped_frames'],
    comparisons=record['comparisons'],exact_frame40_seed_and_camera_verified=True,
    quality_accepted=False,accepted=False,default_applied=False,gaussian_admitted=False,
    original_sources_unchanged=True,original_solver_baseline_preserved=True,temporal_loss_added=True,
    followup_readonly_checks=dict(camera_init_confidence_ignored_in_upstream=True,
        camera_seed_zero_confidence_count=dict(sofa=0,indoor=0),frames_checked=167,
        implication='No zero-confidence torso seed in these inputs; do not claim this original camera-init behavior caused the present failure.',
        prior_shape_control_decision=str(CURRENT/'local-clothed-surface-123/smplifyx-reference-control/official-first-cycle/shared-shape-control/decision.json'),
        prior_shape_promoted=False))
save(OUT/'conclusion.json',conclusion)
for path in [R/'protocol.json',R/'queue-state.json',R/'evidence-ledger.json',Path('C:/Users/Administrator/Documents/New project/shuolee-coordination/local-research-status.json')]:
    d=read(path);d.update(updated=stamp,current_stage='cross_subject_temporal124_head_absence_rejected',summary=summary,
        next_action=next_action,own_gpu_task_running=False,current_worker_pid_wsl=None,current_active_experiment=None,
        gaussian_training_running=False,training_allowed=False)
    d['head_absence_control124']=dict(plan=str(OUT/'plan.json'),result=str(OUT/'result.json'),conclusion=str(OUT/'conclusion.json'),
        actual_visual_review=True,fit_calls=1,skipped41=True,quality_accepted=False,default_applied=False)
    d['last_goal_turn']=dict(classification='progress',evidence=summary,goal_complete=False)
    save(path,d)
print(summary)
