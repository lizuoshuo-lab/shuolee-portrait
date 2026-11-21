"""Actual1000-step gate; continue this optimizer without changing settings."""
from train_winterhat_half8_sh0124 import OUT
from run_official_gaussians123 import read, save, sha, now

assert read(OUT/'progress.json')['state']=='review_pending'
assert not (OUT/'review-1000-decision.json').exists()
snapshot=OUT/'gaussian-run/step-001000'
review=read(snapshot/'review/result.json');assert len(review['records'])==8 and len(review['pages'])==10
for r in review['records']:
    for p in r['images']:assert sha(p['path'])==p['sha256']
proof=OUT/'ui-review-1000'
proofs={str(proof/name):sha(proof/name) for name in ['rear-free-view.png','side-free-view.png','opposite-free-view.png']}
finding=('1000步全部8个原视角的原图／当前mesh／高斯及脸、双手、靴子10页已实际检查。'
         '高斯已学习帽子、蓝衣和长裙外观，图像中的抓肩带位置比裸mesh前伸手更接近原图；'
         '手指、衣褶与靴子仍模糊，红包有孔洞和碎片。原生world.ply在临时查看器按-Y朝上，'
         '后侧与两处斜前自由视角实际检查，前侧整体连贯，未见新增双臂、膝肘反折或明显长条；'
         '未拍后侧明显混色破碎。图像贴合和近似预览都不能证明网格手腕深度或跨视角材料对应已修复。'
         '继续同一优化器到已登记最多18000步；质量保持未通过。')
save(snapshot/'visual-review.json',dict(updated=now(),actual_visual_review=True,
    inspected_frames=[r['frame'] for r in review['records']],inspected_regions=['body','face','left_hand','right_hand','feet'],
    inspected_pages=review['pages'],finding_zh=finding,point_count=read(snapshot/'result.json')['points'],
    old_eight_comparison_common_seven=review['common_seven_foreground_mae'],
    source_fit_pose_quality_accepted=False,wrong_wrist_depth_not_claimed_fixed=True,
    unknown_view_preview='Actual local OpenFile native-world PLY, -Y up, three free views. Approximate preview with no paired-camera or portable-export acceptance.',
    viewer_proofs=proofs,quality_accepted=False,decision='continue',no_parameters_changed=True,
    reason='Registered source and free-view diagnostic shows no new gross anatomical collapse; finish the same bounded exploratory candidate, then assess all-source and36native novel views.'))
save(OUT/'review-1000-decision.json',dict(updated=now(),decision='continue',world_ply_sha256=sha(snapshot/'world.ply'),
    actual_visual_review=True,visual_review=str(snapshot/'visual-review.json'),next_hard_stop=18000,
    reason='Actual native and approximate free-view gate completed; same optimizer and settings only.',quality_accepted=False))
print('WINTERHAT_HALF8_SAME_OPTIMIZER_CONTINUE_TO18000')
