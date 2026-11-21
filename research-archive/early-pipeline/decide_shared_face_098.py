from pathlib import Path
import json,hashlib,datetime

R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/person-static-shared-face-fresh-098')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
e=read(R/'evaluation/compact-summary.json')
assert e['same_feature_3d_consistency']['all_available']['features']==50
assert (R/'standalone-gaussian.png').is_file() and (R/'standalone-points.png').is_file()
d=dict(created=datetime.datetime.now().astimezone().isoformat(),round='098',
 actual_visual_review_completed=True,quality_accepted=False,goal_complete=False,
 summary='同一个面部特征在175帧的可见观测中共用唯一三维坐标，50个有效特征点已接入高斯训练。本轮从20万点独立初始化，完成18,000步。共享目标使反投影落点更集中，但鼻口、眼镜和身体边缘仍有明显重影，未通过质量验收。',
 quantitative_summary='在两组共同可见的相机中，逐特征计算三维落点均方根散布后取平均，098比原方案降低25.1%；鼻部降低31.9%，嘴部降低11.2%。这些是受训练约束的特征。独立下半脸纹理对应误差中位数仅从6.10变为6.00像素，90分位误差从20.18增至21.32像素，不能称为稳定改善。躯干与腿鞋没有明显改善。',
 limits='当前实现约束的是各高斯平面到共享点的距离。多个不同平面仍可经过同一空间点，却与相机射线相交在不同位置，因此还不足以保证同名特征落到同一个三维位置。相机内外参本轮保持固定，共享点留出投影误差仍为5.37像素。只读诊断发现可跨特征预测的共同图像偏移，但它尚不能区分相机误差、头部微动或检测偏差。下一步先检查完整三维落点约束，并将全图对应与面部共享点用于联合相机校验。',
 status='098已完成18,000步并导出兼容PLY。同名特征落点散布降低25.1%，重影仍存在，质量未通过；当前显示最新实验，不代表最佳结果。',
 slide_conclusion='共享点使落点散布降低25%，重复五官与全身边界仍未修复。',
 actual_images_reviewed=['gaussian-run/review-current.jpg','evaluation/face-000070.jpg','evaluation/face-000100.jpg','evaluation/face-000136.jpg','evaluation/body-000100.jpg','evaluation/continuous-030.jpg','evaluation/continuous-090.jpg','evaluation/continuous-120.jpg','standalone-gaussian.png','standalone-points.png'],
 video_review=dict(playback_tested=True,framewise_GT=False,observed_playback_seconds=[1.685457,3.245237,4.800948,6.0],
   screenshots=['playback-0.png','playback-1.png','playback-2.png'],
   observation='Both models retain repeated face contours, translucent sleeve regions and degraded shoes in below-view observations. No visible stable improvement over the baseline. Playback and sampled frames checked, no subjective quality acceptance.'),
 next_bounded_work=dict(scope='先验证同名特征的完整三维落点残差；再用整图稳定特征和同一个面部三维点库做有留出检验的联合相机校正诊断。通过后才安排新的20万点、18,000步从头训练。',
   first='Freeze both completed models and compare direct world-hit-to-X errors against the plane loss; preserve residual vectors and visibility. Check a per-contributor point-to-point formulation, not only blended depth.',
   camera_diagnostic='Use all175 full-image camera observations. Retain stable full-image tracks and one global X per facial ID in one bundle objective, with robust uncertainty, gauge and scale fixed. Initially keep shared intrinsics fixed. Never replace full-image calibration with face-only pose fitting or per-view image translation.',
   acceptance='Evaluate withheld full-image tracks, held facial observations, residual-vector coherence and all-body silhouette consistency. Reject a face-only gain that worsens background or body. Separate micro-motion and detector errors from camera explanations.',
   restrictions=['No automatic weight sweep or longer training','No use of098 learned Gaussian/subject geometry as next initialization','Static clothed person only','Do not deploy failed local experiment to00','No camera correction is claimed yet']),
 presentation_notes=[
  '用户定义：同一鼻尖、眼角或嘴角对应唯一全局三维点X_k，所有可见相机的观测u_ik共同约束X_k。不同特征有不同X_k。二维同名编号必须经过可见性检查。移动的遮挡轮廓点不充当固定表面点。',
  '数据：175帧固定整图相机，5815条可见观测，50个有效三维特征。排除轮廓0至16，内唇66无可见支持而缺省。DLT初始化后，所有可见观测参与稳健重投影优化。五折按连续5帧分组留出，投影误差中位数5.36765像素，故共享目标带不确定性，不能当成真实三维标注。',
  '投影关系为u_ik≈π(K_i(R_iX_k+t_i))。本轮固定K_i、R_i、t_i，只有每个k一个X_k。高斯训练在所有可见观测像素处查询同一X_k，计算各贡献高斯平面到该点的平方距离期望，再用观测与留出误差设置尺度的稳健函数。不是每个相机另设一组三维目标。',
  '先验核验：标准渲染与新增矩渲染的RGB和梯度一致性通过。一个前后双层反例确认新项不会被平均深度互相抵消。新增项从第1步启用，固定权重0.002，空射线阈值仅1e-6。175帧完整相机及像素中心变换均核验。',
  '训练：本轮重新拟合初始化范围，重新生成穿衣点云及头部支持点，共200000点，种子6060，1536像素高，18000步。初始化位置、相机记录、原有优化器配置和采样序列均与093control相同；原有颜色、轮廓、多视图/NCC和纹理对应项保持一致。全新优化器，未加载旧高斯继续训练。最终310961点。',
  '图像：本页左为原视频100，中为093control18k，右为098共享点18k。原照片、相机、裁剪一致。右图鼻口与眼镜重影仍明显。70正面、136侧面、100全身及连续上下视角均检查，衣袖和腿鞋没有可见稳定改善。',
  '结果：受约束特征的三维落点散布平均降低25.057%，鼻部31.938%，嘴部11.163%。点到平面均方根残差从0.04656降至0.00216世界单位。世界单位不是已标定米。独立下半脸纹理对应的中位误差6.102到5.999像素，但P90从20.176恶化到21.319；上半脸、躯干无改善，不能用训练约束残差下降宣布修复。',
  '约束缺口：不同平面都可包含X_k，但其与同一观测射线的交点仍不同。shared-plane-nullspace.json给出零点平面误差而射线落点相距1.5任意单位的代数反例。这个例子证明约束不充分，不单独证明实际重影的全部成因。',
  '相机/观测诊断：用半数特征拟合每帧二维共同偏移，另一半误差中位数从5.36765降到2.87176像素。无相机、图像或模型更新。共同偏移可来自外参误差、头部微动或检测偏差，仍需整图留出对应作区分。',
  '下一步：核验完整三维点到点向量约束；在保留全图稳定观测和固定规范自由度的条件下联合校验相机与共享三维点。需要全图和面部留出误差同时验证，不能只为脸部对齐而牺牲场景。通过后新实验仍从零初始化，不在098模型上续训。',
  '交付：+Y向上标准3DGS PLY，完整协方差和所有SH同步旋转。原生/导出数值核验以及独立打开文件的高斯和点云模式均通过，310961点。数值差异中的近深度排序例外有单独记录。格式通过不等于质量通过，本轮未在00部署。'
 ])
(R/'decision.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(round='098',quality_accepted=False,summary=d['summary']),ensure_ascii=False))
