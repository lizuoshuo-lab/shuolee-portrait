"""Record actual image inspection and fixed-gate measurements for079."""
from pathlib import Path
import json,datetime,hashlib
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
R=B/'current/person-static-surface-consistency-fresh-079'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
pair=read(R/'review/paired-transfer-comparison.json');depth=read(R/'review/depth/result.json')
assert not pair['all_numeric_gates_passed'] and depth['fixed_queries'] and depth['queries']==140
files=['head-'+f'{i:03d}'+'.jpg' for i in [0,33,69,99,135,174]]+['body-099.jpg','novel-above25-bg0.25.jpg','novel-below25-bg0.25.jpg']+['continuous-'+f'{i:03d}'+'.jpg' for i in [0,30,60,90,120,149]]
assert all((R/'review'/f).is_file() for f in files)
d=dict(updated=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),round='079',actually_inspected=True,
    inspected_files=[dict(path=str(R/'review'/f),sha256=hashlib.sha256((R/'review'/f).read_bytes()).hexdigest()) for f in files],
    quality_accepted=False,root_cause_established=False,new_term_adopted=False,best_model_promoted=False,
    continuous_video_full_review=False,continuous_samples_reviewed=6,continuous_frames=150,
    summary='079完成整图相机、人体与衣物初始化、20万点从头1.8万步训练及实图对照。表面约束未减少鼻口重复，预定验收失败，维持静态主线并返回相机与观测一致性诊断。',
    viewer_status='079实验候选：整图相机与人体初始化已完成，鼻口重影仍存在，未通过质量验收。',
    report='body-smplx.html#surface-consistency',
    observed_conclusion='六个固定实拍视角中，衣物外观大体保留。第100帧鼻口前方的重复轮廓、眼镜外散及肩部重影仍然可见，第70帧两侧镜架也没有恢复成稳定的单层结构。本轮没有得到足以接受的人脸改善。',
    novel_conclusion='上方、下方25度及连续路径的6个固定采样位置仍显示脸部重叠、袖口和鞋底破碎，较侧后方的衣物仍有拉伸与漂浮。两轮表现接近。这里只逐图检查了这6个采样位置，没有逐帧审阅全部150帧视频，也没有这些未拍摄方向的真实图像。',
    mechanism_conclusion='共同有贡献的47处脸外采样中，高斯中心深度90%宽度的中位数由人物跨度的2.999%降至2.232%，但平面交点深度宽度由2.866%至2.863%，几乎未变。48个固定位置中，不透明度大于0.05的数量由48降至38，说明部分变化伴随可见贡献消失。中心更集中不能等同于恢复了正确表面。',
    next_step='冻结高斯，在同一整图特征池上核查跨帧一致性，按几何一致性筛选轨迹，并对场景与人体观测分组验收。用留出观测比较共同相机偏差和人体局部微动两种解释，先定位原因，再决定相机修正或表面方法对照。最终仍输出同一个静态人物模型；当前不追加步数或扫描权重。',
    ppt_effect='整个人物轮廓平均距离由2.33降至2.30个训练像素。',
    ppt_summary='整图相机与人体初始化已完成，表面约束未解决鼻口重影。',
    ppt_limit='下半脸对应误差由6.17增至6.36像素，未通过预定验收。\n下一步区分相机偏差与人体局部不一致，再决定训练改动。',
    all_numeric_gates_passed=False,primary_lower_face_median=pair['groups']['face_lower']['metrics']['rgb_center'],
    fixed_queries=140,pc00_opened_this_round=False,local_standalone_open_verified=False)
(R/'review/visual-decision.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(reviewed=True,quality_accepted=False,inspected_images=len(files)),ensure_ascii=False))
