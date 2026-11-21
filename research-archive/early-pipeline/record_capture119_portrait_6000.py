from capture119_clothed import ROOT,REC,rt
root=ROOT/'portrait';snap=root/'gaussian-run/step-006000';folder=REC/'portrait-train-006000-review'
rt.save(folder/'actual-review.json',dict(created=rt.now(),snapshot_sha256=rt.sha(snap/'world.ply'),
    actual_same_camera_review=True,actual_center_distribution_review=True,step=6000,baseline_step=1000,
    quality_accepted=False,physical_calibration_accepted=False,
    images=[dict(path=str(folder/f),sha256=rt.sha(folder/f)) for f in ['body-contact.png','face-contact.png','frame_000067_body_comparison.png']],
    summary='实际审阅六个原图/旧1000步/新6000步/中心点视角：新版裙褶和包形成完整外形，中心没有收回裸模；6000步正面脸较1000步清楚，但右侧视角仍有重复眼鼻口，衣肩和包边有刺状散点。此图两版本步数不同，因果比较以此前双方1000步图为准。继续本轮既定18k，不称质量通过。'))
print('ACTUAL6000_REVIEW_RECORDED')
