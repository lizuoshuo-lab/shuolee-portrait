from pathlib import Path
import json, hashlib, datetime
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
R=B/'current/person-static-native-align-fresh-113'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
d=read(R/'decision.json'); t=read(R/'gaussian-run/result.json')
assert t['steps']==18000 and t['from_scratch'] and t['points']==304609
d.update(updated=datetime.datetime.now().astimezone().isoformat(),state='completed_quality_not_accepted',
 actual_visual_review_completed=True, observed_training_step=18000,new_ply_available=True,
 status='113已完成18,000步；PLY兼容通过，面部和全身重影尚未修复。',
 summary='113从头初始化20万点，固定175个074实拍相机，在原生3DGS训练中加入共享三维特征对齐。最终304,609个高斯。原图、098与113同相机比较仍显示重复鼻尖、嘴部、镜框及身体轮廓残留，未通过质量验收。',
 quantitative_summary='在共同有效的面部查询上，高斯沿视线代表位置相对估计目标的误差中位数下降46.9%；它不是三维真值精度。175帧面部颜色MAE反而增加1.1%，全身前景MAE增加1.9%。头部轮廓外覆盖变化不足1%，各身体区域轮廓内缺失量均增加，不能称为整体边界修复。',
 visual_summary='第100帧的同裁剪脸部仍有鼻口重复与镜框外散；肩袖、衣摆与腿脚仍有残留。固定的新俯仰视角和连续路径抽样也未显示稳定清晰改善，因此不升级为最佳模型。',
 limits='对齐目标由图像对应和估计相机求得，并非独立三维真值。175帧颜色统计属于训练视角；另行留出的几何查询仍不是新相机图像。连续路径共150帧，已检查6个预定关键帧及5个高分辨率新视角面部图，尚未完成全视频连续播放验收。数值代表位置与真正可见表面仍可能不一致。',
 next_bounded_work=dict(scope='用户已明确指定两个Git仓库：aipixel/GaussianAvatar全身方法与jmaraval/GaussianAvatars_SMPLX。分别从头训练，固定同一175图像、18k步预算和评测相机，比较面部及全身重影。先核对公开代码与必要本地适配，不再追加相机优化策略对照。',start_new_experiment=False,reason='repository identity confirmed; audit source completeness and native pipeline adapters'),
 slide_title='训练期对齐：约束有效，重影尚未消除',
 slide_summary='20万点从头训练18,000步；固定相机比较面部与全身',
 slide_observation='代表位置误差下降46.9%；鼻口、镜框及身体轮廓仍有重影。',
 slide_conclusion='本轮未通过；下一步从头比较GaussianAvatar与GaussianAvatars_SMPLX。')
d['reviewed_images']=['gaussian-run/review-current.jpg','evaluation/head-000070.jpg','evaluation/body-000070.jpg','evaluation/head-000100.jpg','evaluation/face-000100.jpg','evaluation/body-000136.jpg']+['evaluation/continuous-%03d.jpg'%n for n in [0,30,60,90,120,149]]+['evaluation/novel-detail/'+v+'-head.png' for v in ['front','left45','right45','above25','below25']]
d['full_video_playback_reviewed']=False
save(R/'decision.json',d)
screens=[R/'standalone-gaussian.png',R/'standalone-points.png']
assert all(p.is_file() for p in screens)
save(R/'viewer-ui-review.json',dict(file_opened_through_UI=True,format_verified=True,observed_count=304609,modes=['Gaussian','points'],
 browser='iab',tab_id='17',method='actual Open File chooser, fit whole body, select Gaussian and point modes',
 screenshots=[dict(path=str(p),sha256=sha(p)) for p in screens],quality_accepted=False,pc00_verified=False,
 note='Local standalone import validates format, upright world coordinates, and both modes; known reconstructed ghosting remains.'))
print('113 decision and actual standalone UI receipt saved')
