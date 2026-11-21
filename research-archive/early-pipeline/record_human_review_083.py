"""Record actual inspection of fixed source views, unseen paths and numerical gates."""
from pathlib import Path
import json,hashlib
from datetime import datetime
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
R=B/'current/person-static-human-surface-init-fresh-083'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
pair=read(R/'review/paired-transfer-comparison.json');depth=read(R/'review/depth/result.json')
assert not pair['all_numeric_gates_passed'] and depth['fixed_queries'] and depth['queries']==140
files=[f'head-{i:03d}.jpg' for i in [0,33,69,99,135,174]]+['body-099.jpg','novel-above25-bg0.25.jpg','novel-below25-bg0.25.jpg']+[f'continuous-{i:03d}.jpg' for i in [0,30,60,90,120,149]]
assert all((R/'review'/f).is_file() for f in files)
d=dict(updated=datetime.now().astimezone().isoformat(timespec='seconds'),round='083',actually_inspected=True,
    inspected_files=[dict(path=str(R/'review'/f),sha256=hashlib.sha256((R/'review'/f).read_bytes()).hexdigest()) for f in files],
    quality_accepted=False,root_cause_established=False,human_initialization_change_adopted=False,best_model_promoted=False,
    continuous_video_full_review=False,continuous_samples_reviewed=6,continuous_frames=150,
    summary='083完成175帧整图相机、人体辅助初始化、20万点从头1.8万步静态训练。单独调整1772个中央脸部起点没有消除鼻口重影，预定质量验收未通过。',
    viewer_status='083实验候选：整图相机＋人体表面初始化，18,000步完成；鼻口重影仍存在，质量未通过。',
    report='body-smplx.html#human-surface-init',
    observed_conclusion='六个固定实拍相机中，完整人物与衣物外观大体保留，但两轮整体接近。第100帧鼻口前方重复轮廓、眼镜外散仍清楚可见，第70帧镜架两侧与肩部仍有拖影。第136和175帧的鼻尖、镜片附近也仍有不稳定边缘。仅改变中央面部初始位置没有获得可接受的人脸改善。',
    novel_conclusion='上方、下方25度及连续路径的六个固定位置仍有脸部重叠、袖口和鞋底破碎；侧后方衣物仍有拉伸和漂浮。077与083表现接近。只逐图检查了六个路径采样位置，未逐帧看完150帧视频，未拍摄方向没有实拍真值。',
    mechanism_conclusion='140条固定射线中，脸外48个位置有46个在本轮仍有可测贡献；共同可测位置的中心深度90%宽度中位数从人物跨度3.032%降至2.668%，但不透明度大于0.05的数量从48降至43。下部纹理24处的中心深度宽度反而从1.183%增至1.389%。变化并不支持“已恢复单层脸面”，也不能由此排除相机、特征对应或局部微动的影响。',
    next_step='冻结083，围绕第100帧的实际重影贡献高斯，比较它们与新初始化中央脸部表面的距离、法向与跨视图可见支持；区分先验在自由优化中失效与先验本身投影不准。先完成这一归因，再判断是否值得参考GaussianAvatars的表面绑定机制。维持整图相机和同一个静态穿衣人物，暂不启动额外步数、权重扫描或新的训练。',
    all_numeric_gates_passed=False,primary_lower_box=pair['groups']['face_lower']['metrics']['rgb_center'],
    fixed_queries=140,not_full_official_gaussianavatars_reproduction=True,pc00_opened_this_round=False,
    local_standalone_open_verified=True,
    single_training_pair=True,repeat_validation=False,unseen_view_GT=False,
    ppt_summary='整图相机＋人体表面初始化已跑通，重影仍未解决。',
    ppt_effect='人物轮廓距离2.331 → 2.289像素，衣物与手脚大体保留。',
    ppt_limit='下半脸框对应误差6.172 → 6.224像素，预定验收失败。\n下一步检查人体先验在优化中为何未能保持单一表面。')
(R/'review/visual-decision.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
pair.update(visual_inspection_complete=True,quality_accepted=False)
(R/'review/paired-transfer-comparison.json').write_text(json.dumps(pair,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(reviewed_images=len(files),quality_accepted=False,points=300220)))
