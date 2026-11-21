"""Record reviewed bounded diagnostic without granting reconstruction admission."""
from pathlib import Path
from datetime import datetime,timezone
import json,shutil
import numpy as np
from scipy.spatial.transform import Rotation
from grayT123_head_rigid_common import *

H=Path(__file__).parent;C=H.parent/'shuolee-coordination';R=REC.parent/'robust-portrait-122'
L=Path('D:/00/工作日志/2026-10-28/人物自由表面训练123');stamp=datetime.now(timezone.utc).isoformat()
if (REC/'head-rigid-conclusion.json').exists():raise RuntimeError('Conclusion already recorded; inspect it, do not duplicate.')
result=read(OUT/'result.json');frozen=read(OUT/'frozen-proxy-fits.json');poses=np.array([o['transform'] for o in frozen['per_frame']])
relative=poses@np.linalg.inv(poses[0]);angles=Rotation.from_matrix(relative[:,:3,:3]).magnitude()*180/np.pi
assert result['complete_diagnostic'] and result['synthetic_passed']
summary=('灰衣人物的SMPL-X仍未对齐。已有鼻部语义点只作FIT代理、皮肤材料点独立验证：固定原K/外参和头部形状，'
    '从所有帧共享刚体调整改为每帧刚体调整后，35个皮肤留出视角中位误差由18.833降到6.934像素，最大仍12.217像素，33/35达到或超过1像素。'
    '这说明当前静态头部假设存在可被逐帧刚体自由度吸收的不一致，但没有证明全部来自真实头部运动。'
    '拟合出的相对旋转最大50.69度，不能称作已经测准的“小幅微动”；鼻部形状近似、遮挡推测点及局部相机误差也会被吸收。'
    '将这组局部变化按同一参考帧解释为整幅场景的相机修正，会把18次路面检查的中位误差由0.687放大到326.903像素，不能直接替换场景相机。'
    '实际审阅全部41帧鼻部原生裁片，侧脸末段部分检测点贴近或越过可见鼻部轮廓，原权重仍有效；这些语义点不能冒充物理材料对应。'
    '头部形状、真实运动和精确相机仍未唯一分离。当前所有精细准入仍拒绝，灰衣高斯未开始；历史模型、相机和训练图均未改动。')
review=dict(created=stamp,actual_visual_review=True,all41_native_pairs_actually_reviewed=True,
    postfit_four_held_native_projection_crops_actually_reviewed=True,raw_images_not_warped=True,
    marker_legend='青圈为皮肤观测，橙十字为静态投影，绿十字为鼻部FIT代理估计逐帧变换后的独立皮肤投影。图内无文字。',
    summary=summary,quality_accepted=False,relative_rotation_deg=dict(median=float(np.median(angles)),maximum=float(angles.max())),
    sources=[dict(path=str(p),sha256=sha(p)) for p in [OUT/'result.json',OUT/'frozen-proxy-fits.json',OUT/'prefit-actual-review.json',OUT/'same_frame_comparison.png']])
save(OUT/'actual-review.json',review)
con=dict(updated=stamp,summary=summary,static_skin=result['original_static_skin_held'],per_frame_skin=result['per_frame_skin_validation']['held_check'],
    shared_transform_skin_invariance_max_abs_px=result['shared_skin_invariance_max_abs_px'],
    road_counterfactual=result['road_counterfactual'],relative_rotation_deg=review['relative_rotation_deg'],
    runtime_seconds=result['seconds'],completed_registered_diagnostic=True,localization_or_pose_truth_certified=False,
    quality_accepted=False,training_started=False,cameras_or_training_images_changed=False,
    guard_completed=True,guard_tests=27,guard_entry_checks=4,native_existing_subject_denials=6,
    result=str(OUT/'result.json'),actual_review=str(OUT/'actual-review.json'),guard_result=str(REC/'alignment-admission-conclusion.json'),
    next='先对现有41帧代理观测做原生图像可见性与候选物理斑点身份核查，区分可见测量与遮挡推断；不改变权重或重新拟合。不得用本次逐帧变换更新相机或启动动态训练。')
save(REC/'head-rigid-conclusion.json',con)
save(REC/'grayT-head-rigid-completion.json',dict(completed=stamp,plan=str(PLAN),plan_sha256=sha(PLAN),
    result=str(OUT/'result.json'),result_sha256=sha(OUT/'result.json'),actual_review=str(OUT/'actual-review.json'),
    actual_registered_runs=1,real_pose_fits=42,new_camera_candidates=0,training_runs=0,complete=True,quality_accepted=False))
nextplan=dict(created=stamp,status='registered_not_executed',name='Native visibility and multi-material support audit, no fitting',
    purpose='Separate image-visible measurements from occluded semantic inference, and establish whether more than one skin material marker can be tracked independently before another geometry intervention.',
    source_frames=list(range(40,81)),source= str(ROOT/'data/training-inputs.json'),
    data_scope='Only the same original41frames and original9nose proxy slots. Do not use geometric residuals or fitted motion to choose/filter points.',
    actions=['Review original nose proxy positions against native RGB and original masks. Report everypoint and frame, label visible, boundary/inferred or uncertain; do not delete or reweight.',
        'Use native head crops to inventory candidate pigment marks across a fixed view sequence. Different marks must have clearly distinct physical IDs. No optical-flow detector or optimizer is authorized by this inventory.',
        'Retain all41frames and existing35skinheld observations; these observations are already seen and must not be relabeled fresh-independent test data.',
        'Only if multiple visible physical materials and a defensible freeze/holdout rule exist, preregister a separate bounded geometry/measurement experiment. Otherwise record the limited support without rerunning the failed semantic fit.'],
    budget=dict(audits=1,new_feature_extractions=0,pose_fits=0,camera_candidates=0,training_runs=0),
    stop='Do not infer that a detector confidence or apparent small fit residual proves material identity. No new training until the fine alignment admission actually passes.',
    quality_accepted=False)
save(REC/'grayT-proxy-visibility-material-plan.json',nextplan)
ledger=read(REC/'evidence-ledger.json');kind='head_rigid_predictive_diagnostic_failed_subpixel'
assert not any(e.get('kind')==kind for e in ledger['entries'])
ledger['entries'].append(dict(time=stamp,kind=kind,
    conclusion=str(REC/'head-rigid-conclusion.json'),result=str(OUT/'result.json'),summary=summary,quality_accepted=False));ledger['updated']=stamp;save(REC/'evidence-ledger.json',ledger)
q=read(REC/'queue-state.json');q.update(updated=stamp,state='active_alignment_diagnosis',current_stage='head-rigid-diagnostic-completed',training_started=False,
    latest_diagnosis=str(REC/'head-rigid-conclusion.json'),next_plan=str(REC/'grayT-proxy-visibility-material-plan.json'),reason=summary);save(REC/'queue-state.json',q)
n=read(REC/'next-action.json');n.update(updated=stamp,latest_diagnosis=str(REC/'head-rigid-conclusion.json'),next_plan=str(REC/'grayT-proxy-visibility-material-plan.json'),
    current_worker_pid_wsl=None,actual_UI_verified=False,training_allowed=False,next=con['next']);save(REC/'next-action.json',n)
q=read(R/'queue-state.json');q.update(updated=stamp,phase='LOCAL_SURFACE123_head_rigid_predictive_diagnostic_completed',
    latest_diagnosis=str(REC/'head-rigid-conclusion.json'),training_started=False,next_plan=str(REC/'grayT-proxy-visibility-material-plan.json'));save(R/'queue-state.json',q)
c=read(C/'local-research-status.json');c.update(updated_at=stamp,current_local_head_rigid_diagnostic=str(REC/'head-rigid-conclusion.json'),
    local_GPU_task_active=False,next=con['next']);save(C/'local-research-status.json',c)
for p in [REC/'head-rigid-conclusion.json',REC/'grayT-head-rigid-completion.json',REC/'grayT-proxy-visibility-material-plan.json']:
    shutil.copy2(p,L/p.name)
with (REC.parent.parent/'long-term-plan.md').open('a',encoding='utf-8') as f:f.write('\n\n'+stamp+' — '+summary+'\n下一步：'+con['next']+'\n')
print(json.dumps(dict(summary=summary,next_plan=str(REC/'grayT-proxy-visibility-material-plan.json')),ensure_ascii=False))
