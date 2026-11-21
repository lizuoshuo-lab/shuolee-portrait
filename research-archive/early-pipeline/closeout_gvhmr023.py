from pathlib import Path
from datetime import datetime
import json,hashlib,re
import numpy as np

lab=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab')
base=lab/'outputs/mini-splatting2-mip-depth-001/shuolee'
r22=base/'current/smplx-gvhmr-surface-contact-fresh-022'
r23=base/'current/smplx-gvhmr-shoe-motion-fresh-023'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),'utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now().astimezone().isoformat(timespec='seconds')
state=read(r23/'run-state.json');assert state['step']==1200 and not state['running']
assert state['output_sha256']==sha(r23/'fitted.npz')
with np.load(r22/'initialization.npz') as a,np.load(r23/'initialization.npz') as b:
    assert a.files==b.files
    changed=[key for key in a.files if not np.array_equal(a[key],b[key])]
    assert not changed
review=read(r23/'review/review-manifest.json');motion=read(r23/'motion-review/motion-audit.json')
baseline=motion['statistics']['scene']['shoe_transfer_split_px']['unoptimized_diagnostic']
candidate=motion['statistics']['refine']['shoe_transfer_split_px']['unoptimized_diagnostic']
body0=review['statistics']['scene']['visible_body_median_px'];body1=review['statistics']['refine']['visible_body_median_px']
component=candidate['median']<baseline['median']*.5 and body1-body0<2
result=dict(created=stamp,state='component_improved_body_geometry_not_accepted',
    candidate_sha256=sha(r23/'fitted.npz'),control_sha256=sha(r22/'fitted.npz'),
    independent_fresh_initialization=True,initial_arrays_identical_to_control=True,
    old_personalized_state_used=False,new_body_updates=1200,new_gaussian_updates=0,
    completed_updates_this_goal_turn={'022':1200,'023':1200,'gaussian':0},
    optimized_feature_points=33,unoptimized_diagnostic_feature_points=33,
    unoptimized_diagnostic_observations=623,diagnostic_before=baseline,diagnostic_after=candidate,
    training_body_median_px_before=body0,training_body_median_px_after=body1,
    diagnostic_component_gate_passed=bool(component),whole_body_accepted=False,gaussian_quality_accepted=False,
    findings=[
        'Relative motion of stable foot-region features improves strongly, including33 unoptimized points.',
        'Feature selection includes shoes and some trouser cuffs; no validated semantic shoe segmentation.',
        'Source-body projection worsens slightly8.13 to8.71px; wrists remain near3.59/4.52px; elbows30.40/24.45px.',
        'Frame23 knee bends12.38/6.89deg, but smaller angle is not accuracy evidence.',
        'Fixed-world side views retain questionable torso shape/lean and occluded-limb placement.',
        'The former knee-ankle-toe frame partly conflated shank motion with foot motion. Actual SMPL-X kinematics are verified and used in023.',
        'Static-consistency selection and source-camera reuse mean the diagnostic split is not independent novel-view GT.'
    ],
    visual_review=dict(source_contact_sheets=list(range(59)),full_source_frames=[0,11,23,33,45,58],
        fixed_world_views=['front','left45','right45','left90','right90','above25','below25'],
        source_feature_closeups=[11,45],time_snapshot_views=['front','left90'],
        time_snapshots=[0,11,23,33,45,58],full_continuous_video_accepted=False),
    next='Inspect observed torso/lower-limb motion and uncertainty; compare a shared torso/lower-body pose hypothesis with the current unconstrained per-frame prior, starting fresh. Do not increase optimization budget by default.',
    gaussian_next_gate='Correct observable body alignment and temporal geometry before the20万 fresh Gaussian/18000-step full run.',
    viewer_replaced=False,remote_deployed=False,retention='022 required matched control;023 current component candidate; official raw prior and original observations protected.')
save(r23/'phase-conclusion.json',result)
state.update(state=result['state'],accepted=False,component_motion_improved=True)
save(r23/'run-state.json',state)
audit22=read(r22/'foot-frame-audit/audit.json')
save(r22/'phase-conclusion.json',dict(created=stamp,state='pointwise_contact_gain_insufficient_control_retained',
    new_body_updates=1200,new_gaussian_updates=0,accepted=False,
    verified_kinematic_transfer_median_px_before=audit22['statistics']['021']['kinematic']['shoe_feature_transfer_px']['median'],
    verified_kinematic_transfer_median_px_after=audit22['statistics']['022']['kinematic']['shoe_feature_transfer_px']['median'],
    kinematic_audit='foot-frame-audit/audit.json',matched_control_for='023',
    decision='Do not promote; retain parameter arrays only as the necessary matched control for the texture experiment.'))
path=base/'work-status.json';status=read(path)
status.update(updated=stamp,current_phase='gvhmr_shoe_motion_fresh_023',local_training_running=False,
    current_decision='023从头1200步：未参与约束的33条足部附近纹理残差30.65→1.36px，身体拟合残差8.13→8.71px。足部运动一致性局部改善；侧面体形、肘及遮挡肢体仍未通过。保留当前候选及022匹配控制，无新高斯训练、不部署00。')
status['latest_local_execution']=dict(kind='gvhmr_shoe_motion_fresh_023',state=result['state'],running=False,
    directory=str(r23),planned_steps=1200,new_body_fit_updates=1200,gaussian_updates=0,
    old_person_fit_loaded=False,component_motion_improved=True,accepted=False)
status['current_experiment'].update(current_candidate=str(r23/'fitted.npz'),candidate_kind='SMPL-X body fitting, not Gaussian model',
    latest_fitting_trial='023',latest_fitting_steps=1200,latest_fitting_gaussian_steps=0,
    status=result['state'],current_training_running=False,accepted=False,candidate_accepted=False)
status['gvhmr_motion_023']=dict(conclusion=str(r23/'phase-conclusion.json'),state=result['state'],accepted=False)
save(path,status)
plan=base/'long-term-plan.md';text=plan.read_text('utf-8-sig')
start='下一步先检查肘/袖口观测的不确定性与身体根变换耦合'
end='尚未替换查看器、不部署00，SMPLest-X仍为未下载后备。'
a=text.index(start);b=text.index(end,a)+len(end)
replacement='''022从头1200步将脚底中心稳定改为对应顶点稳定，未带来整体突破。补充核对发现“膝—踝—脚尖”构造坐标系会混入小腿转动；改用SMPL-X真实髋—膝—踝旋转链，与官方关节变换核对最大差1.52×10⁻⁷。按正确运动链，021/022足部纹理传递中位残差仍为31.71/30.83px。

023重新从原图、相机、官方原始预测及模板初始化，1200步加入稳定足部附近纹理约束：当前踝坐标系相对时刻23传递原始三维特征，再与真实像素比较。33点参与损失、33点仅诊断，按脚均衡并按轨迹观测次数归一化，权重0.3、尺度10像素、100–300步渐增。与022初始化数组完全一致；未读旧个体状态作起点。

未优化的33点共623次观测，残差中位数30.65→1.36px、90%分位39.67→4.22px。身体可见点残差8.13→8.71px，肘部仍30.40/24.45px。局部运动一致性改善，但原图侧面及固定世界相机中的胸腹形态、倾斜和遮挡肢体仍未通过；时刻23膝弯曲12.38/6.89度不是测量真值。实际检查59帧联系表、6张完整叠图、7个世界方向、两条视频的6时刻快照，以及两张纹理投影近景。完整连续视频未验收。

特征按人物框下部与空间邻近选择，实际含鞋和裤脚，不能宣称全部是语义鞋面。源相机用过这些特征，稳定子集也预先筛选，因此33点诊断不是独立未知视角测试。022/023本轮共2400次身体更新，0次高斯更新。

下一步先检验躯干和下肢是否被逐帧自由度误解释为运动：核对实拍纹理，再对共享躯干/下身姿态做一组从头有界对照。旧017的共享下身对照未通过；本轮GVHMR先验和真实纹理约束不同，须重新比较，不能沿用旧结果。保留023当前候选、022匹配控制、019官方独立预测；旧学习状态不得初始化新实验。不重复加步数或调整展示相机来掩盖错误。尚未替换查看器、不部署00，SMPLest-X仍为未下载后备。'''
text=text[:a]+replacement+text[b:]
plan.write_text(text,'utf-8')
print(json.dumps({k:result[k] for k in ['state','diagnostic_before','diagnostic_after','diagnostic_component_gate_passed','whole_body_accepted']},ensure_ascii=False))
