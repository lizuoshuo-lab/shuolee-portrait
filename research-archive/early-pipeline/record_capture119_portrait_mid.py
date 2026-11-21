"""Actual1000-step clothing/face review; continue only this finite fresh run."""
from capture119_clothed import ROOT,REC,rt
root=ROOT/'portrait';snap=root/'gaussian-run/step-001000';folder=REC/'portrait-train-001000-review'
d=dict(created=rt.now(),snapshot_sha256=rt.sha(snap/'world.ply'),actual_same_camera_review=True,
    actual_center_distribution_review=True,continue_same_run=True,max_total_steps=18000,
    formal_geometry_accepted=False,physical_calibration_accepted=False,quality_accepted=False,
    reviewed_frames=['frame_000001','frame_000023','frame_000045','frame_000067','frame_000089','frame_000112'],
    images=[dict(path=str(folder/f),sha256=rt.sha(folder/f)) for f in ['body-contact.png','face-contact.png','frame_000067_body_comparison.png','frame_000067_face_comparison.png']],
    summary='同1000步对照：新版裙摆、包和帽子外形明显收敛，中心点覆盖完整穿衣轮廓；旧版裙摆呈大范围色雾和尖刺。新版手臂/衣肩仍散边，鼻嘴眼睛仍有明显重影，靴子细节不足。允许完成这一轮既定18k来验证衣物分布，不代表相机或面部质量通过。')
rt.save(root/'reviews/gaussian-step-001000-decision.json',d);rt.save(folder/'actual-review.json',d)
print('CONTINUE_THIS_FRESH_RUN_TO_18000_CLOTHING_IMPROVED_FACE_NOT_ACCEPTED')
