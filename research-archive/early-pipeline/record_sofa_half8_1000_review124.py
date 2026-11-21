"""Observed1000step gate; no training-parameter changes."""
from pathlib import Path
from run_official_gaussians123 import read,save,sha,now
from train_sofa_half8_sh0124 import OUT

snapshot=OUT/'gaussian-run/step-001000'
assert not (OUT/'review-1000-decision.json').exists()
assert read(OUT/'progress.json')['state']=='review_pending'
review=read(snapshot/'review/result.json');assert len(review['records'])==8
for r in review['records']:
    for im in r['images']:assert sha(im['path'])==im['sha256']
ui=OUT/'ui-review-1000'
proof={str(ui/name):sha(ui/name) for name in ['rear.png','oblique.png','front-free-view.png']}
note='1,000步全部8帧原图／当前mesh／高斯及脸、双手、脚已实际检查。坐姿、头部和衣服整体连贯；手指、黑鞋和侧面衣袖仍偏软或破碎，未发现新增严重肢体反折。临时查看器自由视角可见粗糙背部和袖边缺口；未拍背面没有真值，临时裸PLY的朝上方向也不作为交付验收。继续同一个优化器到最多18,000步。'
save(snapshot/'visual-review.json',dict(updated=now(),actual_visual_review=True,inspected_frames=[r['frame'] for r in review['records']],
    inspected_regions=['body','face','left_hand','right_hand','feet'],finding_zh=note,
    unknown_view_preview='Actual local viewer OpenFile of source world.ply, three free views. Approximate renderer, no paired-camera/axis acceptance at this gate.',
    viewer_proof=proof,point_count=read(snapshot/'result.json')['points'],quality_accepted=False,
    old_seven_comparison=review['common_seven_foreground_mae'],
    decision='continue',reason='No new gross visible anatomical collapse; finish the registered bounded exploratory pipeline.',
    user_authorized_max_steps=18000,no_parameters_changed=True,reference_frame='frame_000026'))
save(OUT/'review-1000-decision.json',dict(updated=now(),decision='continue',world_ply_sha256=sha(snapshot/'world.ply'),
    actual_visual_review=True,visual_review=str(snapshot/'visual-review.json'),next_hard_stop=18000,quality_accepted=False))
print('SOFA_HALF8_SAME_OPTIMIZER_CONTINUE_TO18000')
