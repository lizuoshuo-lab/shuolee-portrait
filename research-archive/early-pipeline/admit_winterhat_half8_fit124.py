"""Actual all-source/side review: admit a bounded1000-step diagnostic only."""
from run_official_gaussians123 import read, save, sha, now
from run_winterhat_half8_fit124 import OUT

assert not (OUT/'conclusion.json').exists()
review=read(OUT/'review/result.json');result=read(OUT/'result.json')
assert len(review['records'])==8 and read(OUT/'process-exit.json')['exit_code']==0
for row in review['records']:
    for image in row['images'].values():assert sha(image['path'])==image['sha256']
note=('全部8帧原图／mesh与四方向、原生16/28/71/109脸及16/96手部已实际检查。'
      '56/57/71正面整体较稳；16/28侧身有前倾，16/96/109抬手仍离开胸前肩带而向前伸。'
      '这是身体手腕深度错误，不能归为可忽略的手指细节。四方向仍是连贯且关节方向可成立的人体，'
      '未见新增膝肘反折、断肢或躯干塌陷。仅准入1000步有界高斯探索，不代表姿态质量通过。')
save(OUT/'conclusion.json',dict(updated=now(),actual_visual_review=True,
    inspected_frames=[r['frame'] for r in review['records']],regions=['body','face','hands','feet','four_novel_directions'],
    finding_zh=note,exploratory_gaussian_admitted=True,pose_quality_accepted=False,quality_accepted=False,
    initial_gaussian_review_step=1000,automatic_continue_to18000=False,
    hand_policy='Uniform original hand2D data weights zero, body25 wrists and hand pose priors retained; this does not solve wrist-depth ambiguity.',
    face='Native detector RMS3.21-7.63px; visible face direction broadly matches, not subpixel/physical truth. Hat is outside the bare initialization.',
    limbs='Raised wrist/forearm depth wrong in16/96/109, sides lean. Boots have residuals. Anatomy under the long skirt and hidden far fingers unknown.',
    estimated_supported_arc_deg=review['estimated_arc_degrees'],physical_angular_calibration=False,
    next='Fresh bound-Gaussian1000diagnostic only. Inspect all8same-view and novel-view arms/strap/hat/skirt; stop if forearm stretch, duplicate arm, floating hand or severe regression. No automatic18k.',
    old_fit_or_optimizer_reused=False,source_targets_unchanged=True,
    source_review_sha256=sha(OUT/'review/result.json'),fit_sha256=sha(OUT/'selected-fit.npz'),
    fitting_seconds=result['seconds']))
print('WINTERHAT_HALF8_BOUNDED1000_DIAGNOSTIC_ADMITTED_QUALITY_OPEN')
