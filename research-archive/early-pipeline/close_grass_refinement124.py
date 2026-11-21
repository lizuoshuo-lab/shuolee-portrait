"""Close the completed candidate on observed failures; never admit it by averages."""
from datetime import datetime, timezone
import numpy as np
import audit_gaussianavatars_params123 as a

root=a.REC/'cross-subject-sh0-124';out=root/'grass-temporal-refinement'
assert not a.lp(out/'conclusion.json').exists()
assert a.read(out/'temporal-progress.json')['state']=='completed'
assert a.read(out/'process-exit.json')['exit_code']==0
p=a.read(out/'plan.json');r=a.read(out/'review-094/result.json')
assert r['frames']==94
def local(path):return path[5].upper()+':/'+path[7:] if str(path).startswith('/mnt/') else path
for path,digest in p['sources'].items():assert a.sha(local(path))==digest,path
rows={x['frame']:x for x in r['records']}
for x in r['records']:
    for im in x['images'].values():assert a.sha(local(im['path']))==im['sha256']
for ck in [13,27]:
    for old in a.read(out/f'review-{ck:03d}/result.json')['records']:
        new=rows[old['frame']]
        assert old['metrics']==new['metrics'] and old['images']==new['images']
for old in a.read(out/'orientation-preview/result.json')['records']:
    assert old['metrics']==rows[old['frame']]['metrics'] and old['images']==rows[old['frame']]['images']
stats={}
for label in ['independent','refinement']:
    angles=np.array([x[label]['max_local_joint_deg'] for x in r['transitions']])
    stats[label]=dict(max_local_joint_degrees=dict(median=float(np.median(angles)),p90=float(np.percentile(angles,90)),maximum=float(angles.max())),
        frame_weighted_mean_medians_px={g:float(np.median([x['metrics'][label][g+'_weighted_mean_px'] for x in r['records']])) for g in ['body','face','feet']})
trans=[x for x in r['transitions'] if x['current'] in ['frame_000074','frame_000080']]
stamp=datetime.now(timezone.utc).isoformat()
wrist=a.read(out/'occluded-arm-audit/wrist-contract.json')
prior=a.read(out/'review-027/visual-review.json')
visual=dict(updated=stamp,actual_visual_review=True,generated_frames=94,
    previously_reviewed_frames=prior['reviewed_frames'],
    newly_reviewed_overview_pages=[r['pages'][i-1] for i in [19,20,21,24]],
    newly_enlarged_native_body_frames=[1,8,15,46,48,50,73,74,75,76],
    newly_enlarged_four_direction_frames=[73,74,76],
    newly_reviewed_native_face_feet_frames=[73,74,76,77,78,79,80,93,94],
    actual_all94_native_visual_review=False,
    review_scope='Reject at decisive observed failure. All94 fitted and rendered; do not claim every remaining native crop was visually inspected or accepted.',
    unchanged_previous27_and_preview_verified=True,
    findings=[
        'The earlier chest-front arm improvements remain. Aggregate continuity improves but does not protect all transitions.',
        'At73to74 the left elbow changes64.566degrees; the previously hanging far arm becomes raised in common-view meshes. The native image does not support treating this as a successful pose.',
        'At79to80 a51.263degree right-elbow branch change remains. Later meshes can be smooth while retaining both wrists near the visible chest hand.',
        'The74 left body-wrist and left hand-root observations disagree156.736pixels at one exactly identical3D SMPL-X joint;76 left/right hand centroids differ3.034pixels. Native observation overlays show both hand groups occupying the visible hand.',
        'Toe/ankle and facial projection mismatches remain. Detector residuals and synthesized common views are not physical3D ground truth.',
        'A previous unique-wrist-root control and parent-confidence control already failed protected joints. This evidence does not authorize repeating them or claiming that removing duplicate roots alone fixes the sequence.'
    ],failed_transitions=trans,quality_accepted=False,gaussian_admitted=False,
    result_sha256=a.sha(out/'review-094/result.json'),wrist_contract_sha256=a.sha(out/'occluded-arm-audit/wrist-contract.json'))
a.save(out/'review-094/visual-review.json',visual)
summary=('SMPLify-X连续帧约束已接入并完成草地94帧：相邻关节最大变化的中位数8.81°→4.59°，P90为49.04°→10.49°；'
    '但73→74仍跳64.57°、79→80跳51.26°，姿态质量未通过，不进入高斯。已定位侧面遮挡时左右手观测重合，74帧同一个左腕的两个目标相差156.74px。')
next_action=('The single grass-temporal-refinement candidate completed94 and is CLOSED/rejected on actual73/74 and79/80 pose evidence. Do not rerun it or start the prepared grass-refinement-sh0 helper. '
    'Preserve full94 results, original13/27 passes as intermediate-only, current demo SH0 viewer model, and all failed controls. '
    'Next inspect a DISTINCT cross-hand identity/occlusion issue from existing whole-body observations, with a bounded all-four-person read-only comparison and explicit genuine-hand-contact protection. '
    'Reuse duplicate-wrist-anchor-audit and grass/pose-cause-audit/hand-coherence.json; do not repeat root-only suppression or parent-confidence multiplication, both already rejected. '
    'No additional feature extraction, threshold sweep, camera change, pose stitching, full-sequence fitting or Gaussian training until a minimal observation-interface proposal is registered and validated. '
    'Continuity extension is implemented, but the four-person reconstruction goal and15min end-to-end target remain unmet.')
a.save(out/'conclusion.json',dict(updated=stamp,state='completed94_quality_rejected',summary=summary,next_action=next_action,
    detail=('全部94帧已拟合并生成同相机叠加；此前27帧和预览像素／指标原样核验。后段出现决定性失败，已据此拒绝；不声称所有94帧原生裁片都已完成目视验收。'
        '当前发现是遮挡时整只手的身份冲突，不能混同于已失败的“只去掉重复手腕”方案，也尚未证明唯一根因。'),
    actual_visual_review=True,actual_all94_native_visual_review=False,frames=94,quality_accepted=False,gaussian_admitted=False,
    statistics=stats,failed_transitions=trans,
    recorded_fitting_elapsed_seconds_including_review_wait=a.read(out/'official-fit/fit/official-fit-result.json')['elapsed_seconds'],
    cold_start_end_to_end_benchmark=False,goal_complete=False,source_hashes_verified=True,
    images=[dict(path=wrist['images'][1]['path'],label='第74帧：原图／原生2D观测／当前SMPL-X。红圈是身体左腕，橙点是左手；蓝圈与青点是右手。图中不加文字。'),
        dict(path=rows['frame_000074']['images']['sides']['path'],label='第74帧共同四方向：左旧独立拟合，右连续精修；远侧手臂仍错误抬起。合成视角不是真值。')]))
a.save(out/'gaussian-admission.json',dict(updated=stamp,pose_admitted_for_gaussian_control=False,frames=94,
    reason='Decisive temporal/depth failure remains; no full native visual acceptance.',
    visual_review_sha256=a.sha(out/'review-094/visual-review.json'),quality_accepted=False,training_started=False))
handoff=a.read(out/'gaussian-handoff-plan.json')
handoff.update(state='not_admitted_pose_review_failed',training_started=False,quality_accepted=False,
    pose_rejection=str(out/'gaussian-admission.json'))
a.save(out/'gaussian-handoff-plan.json',handoff)
print(summary)
