from pathlib import Path
from datetime import datetime
import json
import hashlib
import numpy as np
from scipy.spatial.transform import Rotation

BASE = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
R20 = BASE/'current/smplx-gvhmr-registration-fresh-020'
R21 = BASE/'current/smplx-gvhmr-articulation-fresh-021'
now = datetime.now().astimezone().isoformat(timespec='seconds')
read = lambda p: json.loads(p.read_text('utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), 'utf-8')
def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}

st = read(R21/'run-state.json')
assert st['step']==1200 and not st['running']
assert sha(R21/'fitted.npz')==st['output_sha256']
init, fit = load(R21/'initialization.npz'), load(R21/'fitted.npz')
review = read(R21/'review/review-manifest.json')
assert np.array_equal(init['frame_ids'], fit['frame_ids']) and len(fit['frame_ids'])==59
floor = read(BASE/'current/smplx-observation-support-014/floor-reference.json')
normal, offset = np.array(floor['floor_normal_world']), float(floor['offset'])
scale = float(fit['world_scale'])
height = {}
for side in ['left','right']:
    h = (fit['vertices_world'][:,init['sole_'+side]]@normal+offset)/scale
    height[side] = {'min_model_m':float(h.min()), 'median_model_m':float(np.median(h)),
                    'max_model_m':float(h.max())}
r0 = Rotation.from_rotvec(init['body_pose'].reshape(-1,3)).as_matrix()
r1 = Rotation.from_rotvec(fit['body_pose'].reshape(-1,3)).as_matrix()
delta = Rotation.from_matrix(r1@r0.transpose(0,2,1)).as_rotvec().reshape(59,21,3)
beta_delta = fit['betas']-np.median(init['betas'],axis=0)
diagnostic = {
    'created':now,
    'fitted_sha256':sha(R21/'fitted.npz'),
    'correction_max_component_rad':float(np.abs(delta).max()),
    'correction_rotation_median_degrees':float(np.rad2deg(np.linalg.norm(delta,axis=-1)).mean()),
    'shape_max_component_change':float(np.abs(beta_delta).max()),
    'shape_rms_change':float(np.sqrt(np.mean(beta_delta**2))),
    'sole_height_above_estimated_floor':height,
    'metric_boundary':'Distances use template-scale meters and the estimated SfM floor; no measured 3D GT.',
    'bounds_from_protocol':read(R21/'protocol.json'),
}
# Correct the statistic name: this is the mean, not the median.
diagnostic['correction_rotation_mean_degrees']=diagnostic.pop('correction_rotation_median_degrees')
save(R21/'parameter-audit.json',diagnostic)

files = ['source-'+f'{i:06d}'+'.jpg' for i in [0,11,23,33,45,58]]
files += ['fixed-'+v+'.jpg' for v in ['front','left45','right45','left90','right90','above25','below25']]
files += ['all-views-00.jpg','all-views-20.jpg','all-views-40.jpg']
reviewed = [{'path':f, 'sha256':sha(R21/'review'/f)} for f in files]
conclusion = {
    'created':now, 'state':'staged_fitting_reviewed_partial_improvement_not_accepted',
    'body_geometry_accepted':False, 'viewer_model_replaced':False, 'remote_deployed':False,
    'fresh_body_updates':{'registration_image':800,'registration_scene':800,'staged_articulation':1200,'total':2800},
    'new_gaussian_updates':0, 'old_person_fit_used_for_initialization':False,
    'current_candidate':'021 retained for diagnostics; not a final body or deployment model',
    'input_contract':'All59 real calibration views; raw GVHMR prediction, original observations, licensed SMPL-X, raw local HaMeR articulation. Independent initialization and optimizers.',
    'statistics':review['statistics'],
    'improvements':[
        'Visible body observation residual median30.1466 ->22.7908 ->8.1265px. All values are fitting residuals, not independent accuracy.',
        'Visible wrist placement improves after local hand and bounded body articulation fitting.',
        'Frame23 geometric knee bend13.96/10.06deg remains close to independent prior13.69/10.53deg, unlike previous fit about20/21deg. Actual joint angles unknown.'
    ],
    'unresolved':[
        'Fixed-world side views still show body inclination and limb/shape differences; complete geometry not accepted.',
        'Elbow observation medians remain30.36/24.88px; the left elbow is worse than rigid scene registration24.83px.',
        'Some projected hand/foot outlines still disagree. Far-side limbs are weakly observed and excluded where occluded.',
        'Bare neutral template shape is not the clothing surface. Loose garments do not reveal true torso thickness or knee position.',
        'The silhouette containment ratio does not improve in every view; frame11 rises1.18% to2.42% from scene to articulation fit.',
        'Contact sheets cover all59 observations and7 fixed-world cameras at frame23. Continuous temporal playback has not been visually accepted.'
    ],
    'visual_review':{'source_overlays_full_size':[0,11,23,33,45,58], 'contact_sheet_frames':'00..58',
        'fixed_frame':23, 'common_world_cameras':['front','left45','right45','left90','right90','above25','below25'],
        'same_world_transform_for_all_models':True, 'per_model_display_normalization':False,
        'continuous_temporal_playback_reviewed':False,'files':reviewed},
    'method_findings':[
        'COCO17 surface regressor and same-name SMPL-X skeleton joints are different landmarks. Official regressor replay max5.4e-7m; shoulder differences about16-20px. Not the sole proven cause of earlier failure.',
        'Shared pose corrections and bounded frame corrections are composed on SO(3), with raw GVHMR local joints/rotations providing a soft prior.',
        'HaMeR contributes local finger articulation; crop-camera absolute wrist orientation is not transplanted into the scene.'
    ],
    'retention':'Preserve019 official predictions/weights,020 scene-registration control and021 current candidate, source and compact review. No deployment or viewer replacement.',
    'next':[
        'Audit elbow/cuff observation uncertainty and head/torso registration using real image evidence before adding more fitting steps.',
        'Review fixed-camera temporal deformation and foot height/contact. Lower2D residual alone cannot approve this candidate.',
        'Only after mesh acceptance start a new20万-point Gaussian initialization and18000-step bounded run; never resume an old personalized checkpoint.'
    ],
    'parameter_audit':'parameter-audit.json', 'independent_3d_gt':False
}
save(R21/'phase-conclusion.json',conclusion)
st.update(state='fitted_and_reviewed_partial_not_accepted',review_completed=now,accepted=False)
save(R21/'run-state.json',st)
review['state']='rendered_and_reviewed_partial_not_accepted'
review['visual_review_evidence']='../phase-conclusion.json'
save(R21/'review/review-manifest.json',review)
save(R20/'phase-conclusion.json',{
    'created':now,'state':'fixed_articulation_registration_reviewed_partial_not_accepted',
    'new_body_fit_updates':1600,'new_gaussian_updates':0,'accepted':False,
    'conclusion':'Root/scale registration improves some source placement but leaves local limbs. Scene arm retained as the necessary control;021 independently starts from raw official predictions.',
    'statistics':read(R20/'review/review-manifest.json')['statistics'],
    'current_followup':str(R21/'phase-conclusion.json')})

status_path=BASE/'work-status.json'
status=read(status_path)
status.update(updated=now,current_phase='gvhmr_staged_articulation_reviewed_021',local_training_running=False,
    current_decision='身体先验已下载核验；020/021从头拟合完成并审阅。可见投影中位数30.15→22.79→8.13px，手腕改善；肘部、侧面及遮挡肢体仍未通过，保留候选继续诊断。0次新高斯更新，不部署00。')
status['latest_local_execution'].update(state='fitted_and_reviewed_partial_not_accepted',running=False,accepted=False,
    new_body_fit_updates=1200,related_new_body_updates_total=2800,review_directory=str(R21/'review'))
status['current_experiment'].update(latest_fitting_trial='021',latest_fitting_steps=1200,
    latest_fitting_arms=1,current_training_running=False,latest_fitting_gaussian_steps=0,
    current_candidate=str(R21/'fitted.npz'),candidate_accepted=False)
status['gvhmr_articulation_021']={'conclusion':str(R21/'phase-conclusion.json'),
    'body_geometry_accepted':False,'fresh_body_updates_total':2800,'new_gaussian_updates':0,
    'all59_source_contact_sheets_reviewed':True,'full_temporal_review_completed':False}
save(status_path,status)
plan=BASE/'long-term-plan.md'
text=plan.read_text('utf-8-sig')
before='下一步020从原始观测与官方GVHMR预测重新建立共享尺度和场景根变换，先固定局部骨架，仅检验配准，再比较有界关节修正。不能把019的骨盆居中展示当成世界对齐，也不能用官方最低关节归零代替实测地面。未训练新高斯，不部署00。候选原始/后处理数组与必要证据暂时保留；SMPLest-X仍为后备，未下载8.2GB权重。'
after='020两个独立配准组各800步完成；021重新从官方原始预测开始，分阶段开放根变换、共享关节/体形和有界逐帧关节，并加入HaMeR局部手姿态，1200步完成。合计2800次新身体更新，0次高斯更新。可见观测残差中位数30.15→22.79→8.13px；手腕贴合改善，时刻23膝弯曲13.96/10.06度。已检查全部59张联系表、6张完整叠图及同一世界坐标下7个方向。肘部残差仍约30/25px，侧面倾斜、遮挡肢体及体形未通过。不能将拟合残差当成三维准确率。\n\n下一步先检查肘/袖口观测的不确定性与身体根变换耦合，补查固定相机的时序形变及脚底高度。保留021诊断候选、020场景配准控制及019独立预测；不用旧个体参数初始化下一轮。视角显示不逐模型居中或归一化。尚未替换查看器、不部署00，SMPLest-X仍为未下载后备。'
assert before in text
plan.write_text(text.replace(before,after,1),'utf-8')
print(json.dumps({'review_complete':True,'new_body_updates':2800,'gaussian_updates':0,'accepted':False,'parameter_audit':diagnostic},ensure_ascii=False))
