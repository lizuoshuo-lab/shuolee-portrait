"""Persist actual reviews; release only the already-running 300k checkpoint gate."""
import audit_gaussianavatars_params123 as a
from run_official_gaussians123 import now

root=a.REC/'demo-vhap-initialization'
long=root/'head-gaussian-long300k'
request=a.read(long/'review-1000-request.json')
snap=long/'gaussian-run/step-001000'
assert a.sha(snap/'world.ply')==request['world_ply_sha256']
review=dict(updated=now(),actual_all59_review=True,reviewed_sheets=5,
    summary_zh='30万步独立长训练：1000步检查未见整体爆炸，帽檐和颈部仍模糊；按已登记时序继续。',
    reviewed_frames=59,native_scale=1,quality_accepted=False,novel_view_review=False,
    findings=['59个训练视角均已审阅，能辨认脸与帽子，未见整个头部崩塌。',
              '帽檐、耳边与颈部仍有模糊和轮廓欠覆盖，嘴部与侧面仍有偏差。',
              '当前仅10144初始点，尚未到5000步增密起点，不能与18k短时序的1000步等同。',
              '遮挡帧不提供可信面部外观真值；此次仅为继续计算检查，不是质量验收。'])
a.save(snap/'visual-review.json',review)
a.save(long/'review-1000-decision.json',dict(updated=now(),decision='continue',
    world_ply_sha256=request['world_ply_sha256'],actual_all59_review=True,
    quality_accepted=False,reason=review['summary_zh']))

short=root/'head-gaussian-first-cycle'
delivery=short/'delivery/step-018000'
summary='面部帽子18000步已完成（199842点）：训练视角颜色更清楚，但帽沿、颈部散点和未知视角仍未通过；已启动独立30万步对照。'
quality=dict(updated=now(),actual_all59_review=True,actual_same_camera_review=True,
    reviewed_sheets=5,reviewed_frames=59,native_scale=1,continuous_novel_views_reviewed=36,
    source_sha256=a.read(delivery/'validation.json')['sha256'],quality_accepted=False,
    same_camera_quality_accepted=False,novel_quality_accepted=False,summary=summary,summary_zh=summary,
    findings=['训练视角颜色误差中位数0.06436降为0.04099，轮廓IoU约86.9%至86.8%，未改善轮廓。',
              '帽檐边缘有毛刺、缺口，颈部及领口仍有浮点，部分侧脸位置不一致。',
              '36个连续未知视角已逐页查看，出现SH颜色变化、面部拖影和背部表面破碎。',
              '未拍清的背面与设备遮挡脸部缺少可靠真值，不视为恢复成功。'])
a.save(short/'gaussian-run/step-018000/visual-review.json',quality)
a.save(delivery/'quality-review.json',quality)
print('RECORDED_ACTUAL_REVIEWS_AND_RELEASED_300K')
