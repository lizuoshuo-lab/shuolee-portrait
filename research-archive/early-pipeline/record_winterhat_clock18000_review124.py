"""Record the actual final native/orbit review before publishing the candidate."""
from datetime import datetime, timezone
from pathlib import Path
import audit_gaussianavatars_params123 as a

out=a.REC/'cross-subject-sh0-124/winterhat-clock18k-cache124'
snap=out/'gaussian-run/step-018000';delivery=out/'delivery/step-018000'
assert a.read(out/'process-exit.json')['exit_code']==0
assert a.read(out/'progress.json')['state']=='bounded_control_completed'
assert not a.lp(snap/'visual-review.json').exists()
v=a.read(delivery/'validation.json');assert v['format_passed'] and v['native_rotated_render_parity']['passed']
finding='全部8个同相机全身／面部／脚部及36个连续旋转方向已实际审阅。同18,000步旧时序相比，帽边散点减少，蓝色衣物、裙褶和靴子细节更清楚；胸前手仍保持原动作。红色提包大孔洞未解决，眼嘴与帽边仍有细碎误差。转至未拍侧后方时，衣服、裙子和帽子仍呈碎片与孔洞；SH0只限制方向颜色，不能补出可靠的未观测表面。最终74,330点，未达到约20万；mesh最大偏移0.725毫米，没有新增法向翻折，受保护面部位移为零。保留为可查看的流程对照，未通过完整人体质量验收。'
review=dict(updated=datetime.now(timezone.utc).isoformat(),step=18000,actual_visual_review=True,
    overview_body_face_feet_reviewed=[13,16,41,56,57,71,96,109],
    native_crops_reviewed=['frame_000071 body','frame_000071 face','frame_000071 feet'],
    matched_baseline_native_reviewed=['frame_000071 body','frame_000071 face'],
    baseline_run='winterhat-selected-sh0-pipeline124',
    novel_views_reviewed=True,novel_synthetic_views=36,
    finding=finding,quality_accepted=False,novel_quality_accepted=False,
    geometry=a.read(out/'gaussian-run/geometry-step-018000.json'),
    point_target_accepted=False,cold_video_15min_accepted=False,
    decision='bounded_run_finished_publish_separate_diagnostic_do_not_extend')
a.save(snap/'visual-review.json',review)
a.save(delivery/'visual-review.json',review)
for name in ('orbit_00-11.png','orbit_12-23.png','orbit_24-35.png'):
    dest=Path('D:/00/记录/001')/('分析_冬帽时序换算124_018000_'+name)
    assert not dest.exists()
    dest.write_bytes(a.lp(delivery/'novel'/name).read_bytes())
print('CLOCK18000_ALL8_AND36_NOVEL_ACTUAL_REVIEW_RECORDED')
