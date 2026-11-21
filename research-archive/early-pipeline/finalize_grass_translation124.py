"""Record the actual13-frame failed quality review; do not fit or select poses."""
from datetime import datetime
import audit_gaussianavatars_params123 as a

out=a.REC/'cross-subject-sh0-124/grass-temporal-roottranslation'
p=a.read(out/'plan.json');request=a.read(out/'review-request.json')
decision=a.read(out/'review-decision.json');review=a.read(out/'review-013/result.json')
exit_record=a.read(out/'process-exit.json');launch=a.read(out/'launch.json')
assert not decision['continue_same_candidate'] and exit_record['exit_code']==1
assert request['frames']==decision['reviewed_frames'] and len(request['frames'])==13
assert a.read(out/'failure.json')['computed_frames']==13
def local(path):return path[5].upper()+':/'+path[7:] if path.startswith('/mnt/') else path
for path,digest in p['sources'].items():assert a.sha(local(path))==digest,path
rows={r['frame']:r for r in review['records']}
for row in rows.values():
    for image in row['images'].values():assert a.sha(local(image['path']))==image['sha256']
good=[27,28,29,30,31,32,34,37,38];bad=[26,33,35,36]
assert sorted(good+bad)==list(range(26,39))
elapsed=(datetime.fromisoformat(request['updated'])-datetime.fromisoformat(launch['updated'])).total_seconds()
trans=next(t for t in review['transitions'] if t['previous']=='frame_000032')
def metric(frame,arm,key):return rows[f'frame_{frame:06d}']['metrics'][arm][key]
summary=('SMPLify-X连续约束已接入。新增仅放开整体平移3参数的草地13帧对照：9帧抬臂由身后回到胸前，'
    f'第32帧右肘投影误差{metric(32,"frozen_translation","arm_error_px")[1]:.2f}→{metric(32,"free_translation","arm_error_px")[1]:.2f}px。'
    '但26、33、35、36帧仍向后伸臂，32→33仍发生约0.813模型米的手腕相对深度／位置跳变；序列质量未通过，已停在13帧，不进入高斯。')
detail=(f'原生全身、脸、脚和共同侧视的全部13帧均已审阅。只增加camera.translation的三个自由参数，K、背景外参、共享体型、原版五阶段／先验／LBFGS与时间权重不变；'
    f'实际启动至检查点{elapsed:.1f}秒，不含审阅等待。新解的32→33最大局部关节变化{trans["free_translation"]["max_local_joint_deg"]:.2f}°，'
    '33→34又翻回，35—36再次向后伸臂。说明释放整体位置能改善部分单帧拟合，但仍不能保证连续的正确深度。'
    '面部与脚部有局部投影偏差，检测残差和合成侧视不是独立三维真值；人体长度为模型尺度，外参物理精度仍未核实。'
    '剩余81帧未运行，没有挑选9帧拼成“通过”序列，没有发布新点云。')
next_action=('The one grass-temporal-roottranslation candidate is closed at13/94; do not resume it, sweep weights or stitch poses. '
    'New evidence: freeing the existing3 placement parameters makes the zero/mean32 anchor front-arm geometry plausible and improves9/13, but33 and35-36 still switch depth despite the same-run rotation prior. '
    'Next prepare one explicitly bounded grass schedule-continuation control in a new directory: retain full original five stages for the anchor and use only existing final body/face-hand refinement stages for same-run seeded neighbours, with the now-free placement retained. '
    'This tests cold-start annealing versus warm continuation, not another temporal-weight scan. Review the prior sofa/demo warm-tail failures before registering; they remain failures and cannot justify acceptance. '
    'Do not execute until exact source diff, shared input hashes, all-frame order, native face/feet/arm protections,13-frame stop and no-old-pose initialization are written and preflighted. '
    'No new solver, selection, weights, camera calibration or GS before full pose review; goal remains active and15-minute end-to-end remains unachieved.')
images=[]
for frame,kind,label in [(32,'sides','第32帧同相机四方向：左固定整体位置，右联合位置；手臂由身后回到胸前'),
    (33,'sides','第33帧同相机四方向：右侧仍是向后伸臂的错误深度'),
    (32,'body','第32帧原图／固定整体位置mesh／联合位置mesh，原生同一裁片')]:
    images.append(dict(path=rows[f'frame_{frame:06d}']['images'][kind]['path'],label=label))
visual=dict(updated=datetime.now().astimezone().isoformat(),actual_visual_review=True,
    reviewed_frames=sorted(rows),body_side_pages=review['pages'],face_feet_pages=[str(out/f'review-013/face-feet-{i:02d}.png') for i in range(1,5)],
    enlarged_four_directions=[32,33],front_arm_direction_improved_frames=good,persistent_wrong_depth_frames=bad,
    all_native_face_feet_reviewed=True,mask_clothing_not_equated_to_naked_smpl=True,
    synthetic_sides_not_ground_truth=True,quality_accepted=False,gaussian_admitted=False)
a.save(out/'visual-review.json',visual)
conclusion=dict(updated=visual['updated'],state='stopped_quality_failed',summary=summary,detail=detail,next_action=next_action,
    images=images,frame_count=13,remaining_not_run=81,front_arm_direction_improved_frames=good,persistent_wrong_depth_frames=bad,
    selected_or_stitched_frames=False,actual_visual_review=True,quality_accepted=False,default_applied=False,
    gaussian_admitted=False,new_ply=False,goal_complete=False,launch_to_checkpoint_seconds=elapsed,
    review_wait_excluded=True,decisive_transition=trans,frame32_metrics=rows['frame_000032']['metrics'],
    source_hashes_verified=True,frozen_sources_unchanged=True,preflight_failures_preserved=True,process_exit=exit_record)
a.save(out/'conclusion.json',conclusion)
progress=a.read(out/'temporal-progress.json');progress.update(updated=visual['updated'],state='stopped_quality_failed',pid=None,frames=13,total=94,quality_accepted=False)
a.save(out/'temporal-progress.json',progress)
print(summary)
