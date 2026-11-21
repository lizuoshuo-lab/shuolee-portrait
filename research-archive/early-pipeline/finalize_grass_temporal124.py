"""Record the actual 13-frame review and deliberate quality stop."""
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import audit_gaussianavatars_params123 as a

out=a.REC/'cross-subject-sh0-124/grass-temporal'

def local(value):
    s=str(value)
    if s.startswith('/mnt/'): return Path(s[5].upper()+':/'+s[7:])
    if s.startswith('/opt/'): return Path('//wsl.localhost/Ubuntu-20.04'+s)
    return Path(s)


p=a.read(out/'plan.json');review=a.read(out/'review-013/result.json')
decision=a.read(out/'review-decision.json');progress=a.read(out/'temporal-progress.json')
exit_record=a.read(out/'process-exit.json');request=a.read(out/'review-request.json')
assert review['frames']==progress['frames']==13 and decision['actual_visual_review']
assert not decision['continue_same_candidate'] and exit_record['exit_code']==1
assert 'Stopped by actual pose review' in progress['error']
assert not a.lp(out/'conclusion.json').exists()
for path,digest in p['sources'].items():assert a.sha(local(path))==digest,path
byid={r['frame']:r for r in review['records']}
transition=next(t for t in review['transitions'] if t['current']=='frame_000033')
assert transition['previous']=='frame_000032'
start=a.read(out/'launch.json')['updated']
seconds=(datetime.fromisoformat(request['updated'])-datetime.fromisoformat(start)).total_seconds()
temporal=[x for x in progress['records'] if x.get('gradient_checked')]
assert len(temporal)==12 and all(x['temporal_gradient_finite'] and x['calls']>0 for x in temporal)
assert all(r['same_K_shape_camera_translation'] for r in review['records'])
with np.load(out/'official-fit/frames/frame_000032/winner-camera-space.npz') as new, np.load(
    a.O/'full94-shared-shape/frames/frame_000032/winner-camera-space.npz') as old:
    anchor_equal=all(np.array_equal(new[k],old[k]) for k in ['vertices_camera','body_pose_axis_angle','global_orient','camera_translation','betas'])
assert anchor_equal
stamp=datetime.now(timezone.utc).isoformat()
summary=('原版SMPLify-X连续帧接口已接入并完成草地13帧验证；相邻根朝向按相机旋转转换。'
    '约束实际生效，但第32→33帧右肘仍跳变52.77°（原独立拟合3.98°），第33帧右肘检测投影误差37.32→82.52px；'
    '侧视确认抬起的手臂仍伸向身后。已在13帧检查点停止，未训练或发布新高斯。')
detail=('这13帧的最大局部关节跳变119.23→52.77°，说明某些突变减轻，不能据此宣布整体改善：新方案将明显跳变提前到32→33。'
    '原生脸部与脚部仍存在投影偏差。参考32姿态与旧基线逐数组一致，其隐藏手臂深度原本已不正确；'
    '相邻初值和时间平滑可以传播这个错误。所有13帧内参、体型和相机平移完全相同；12个非参考帧均记录有限的实际时间项梯度。'
    f'启动至检查点耗时{seconds:.1f}秒，审阅等待不计入该值。未执行剩余81帧，44/45历史跳变在这轮未测试。'
    '检测残差和合成侧视都不是独立三维真值，原外参物理精度仍未知。')
next_action=('Grass temporal124 is stopped at13/94 after actual anatomy/projection review. Do not resume it, rerun completed controls, scan temporal weights, stitch old poses or launch GS. '
    'The unmodified reference32 anchor itself has wrong raised-arm depth;32->33 introduces a new elbow branch despite correct temporal/root-camera interfaces. '
    'Next inspect the saved reference32/33, original shoulder-gated orientation selection and original pose-prior/initialization contracts to identify an upstream-supported reproducible mechanism for the depth ambiguity. '
    'Use old results only for diagnosis, not new initializers. A distinct hypothesis, pose protections and finite budget must precede any new fit; no ad hoc full-sequence solver. '
    'Keep indoor head5, sofa and winterhat failures closed. Current demo120k SH0 stays published. Goal remains active and quality/15min end-to-end targets unachieved.')
images=[dict(path=byid['frame_000033']['images']['body']['path'],label='第33帧原图／旧独立拟合／连续约束，同相机原生像素'),
    dict(path=byid['frame_000033']['images']['sides']['path'],label='第33帧同相机四方向；左旧独立拟合，右连续约束；合成视角不是真值')]
v=dict(updated=stamp,actual_visual_review=True,frames=13,full94_reviewed=False,
    reviewed_native_parts=['body','face','feet'],all13_common_side_views_reviewed=True,
    enlarged_four_direction_frames=['frame_000026','frame_000033'],
    pages=[str(out/'review-013'/f'page-{n:02d}.png') for n in range(1,5)],
    face_feet_pages=[str(out/'review-013'/f'face-feet-{n:02d}.png') for n in range(1,5)],
    conclusion=summary,detail=detail,quality_accepted=False,gaussian_admitted=False,text_on_images=False)
a.save(out/'visual-review.json',v)
a.save(out/'conclusion.json',dict(updated=stamp,state='stopped_quality_failed',summary=summary,detail=detail,
    next_action=next_action,images=images,actual_visual_review=True,quality_accepted=False,default_applied=False,
    gaussian_admitted=False,frame_count=13,remaining_frames_not_run=81,upstream_sources_preserved=True,
    anchor_exact_old_parity=True,same_K_shape_camera_translation_all13=True,
    same_run_neighbour_pose_only=True,nonanchor_temporal_gradient_checks=12,
    launch_to_checkpoint_seconds=seconds,review_wait_excluded=True,
    native_record_images=52,record_directory='D:/00/记录/005',
    decisive_transition=transition,frame33_metrics={key:{k:v for k,v in val.items() if k!='all118_error_px'} for key,val in byid['frame_000033']['metrics'].items()},
    stop_was_deliberate=True,process_exit=exit_record,source_hashes=p['sources'],
    frozen_runner_bytes_unchanged=True,physical_calibration_accepted=False,goal_complete=False))
progress.update(state='stopped_quality_failed',pid=None,quality_accepted=False,gaussian_admitted=False,
    review=str(out/'visual-review.json'),deliberate_quality_stop=True)
a.save(out/'temporal-progress.json',progress)
print(summary)
