from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,numpy as np
LAB=Path(__file__).resolve().parents[2];SITE=Path('D:/Projects/EngineLab/engine-home');OUT=LAB/'outputs/dynamic-human-enerf-direct-pose-065';BASE=LAB/'outputs/dynamic-human-enerf-articulated-control-064'
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
assert read(OUT/'direct_reprojection/native-evaluation/independent-image-audit.json')['images']==102
pair=read(OUT/'paired-comparison/summary.json');temporal=read(OUT/'temporal-comparison/summary.json');pose=read(OUT/'projection-comparison.json');detail_summary={name:dict(mean_psnr_delta_db=float(np.mean([r['delta_db'] for r in pair['details'] if r['region']==name])),improved=sum(r['delta_db']>0 for r in pair['details'] if r['region']==name),count=6) for name in ['dark_head','dark_forearm','light_forearm','dark_shoe']}
small={k:v for k,v in pose.items() if k!='records'};small['full_local_report_sha256']=hashlib.sha256((OUT/'projection-comparison.json').read_bytes()).hexdigest();save(OUT/'projection-summary.json',small)
priorreview=read(BASE/'visual-review.json');save(OUT/'visual-review.json',dict(status='12fixed_fullperson24fixed_details_and_worst_regression_actually_viewed',full_person=priorreview['full_person'],details=priorreview['details'],additional_worst='camera16-frame000745-full-person.png',
 findings=['四个固定机位中，黑衣人物脸部、眼镜与头发轮廓较前一轮集中；与原照片仍有平滑和局部姿态差异。','两台固定细节机位的手指和摆动前臂仍模糊，部分肢体厚边增加；鞋带未清楚恢复。','机位01/04的衣服图案没有与头部同步改善，机位08/17的玩偶拉花加重；原先较好的区域未得到整体保持。','最差机位16帧745中，右列玩偶呈明显横向条纹，人体边缘也有偏差，故保留负例。'],
 temporal_visual_scope='Three separated validation frames per four cameras inspected, plus worstcase; all4videos fully decoded but not claiming full playback visual review.',acceptance='not_passed',goal_complete=False))
save(OUT/'final-summary.json',dict(status='direct_pose3000plus_Gaussian15000updates_and102native_image_audit_complete_global_candidate_rejected',
 pose_projection=pose['summary'],validation_foreground=pair['validation_foreground'],fixed_detail_regions=detail_summary,temporal_difference_residual=temporal['mean_temporal_residual'],
 interpretation='Head becomes visibly sharper, but fullforeground average drops0.634dB and78of90images regress. This combined wholebody/head pose change is not adopted as the overall baseline. Projection residual is not an adequate proxy for appearance.',
 next='Isolate neck/head refinement while freezing064body motion, translation, shape, timing and allotherjoints; then evaluate samebudget reconstruction before addressing local residual deformation of limbs/clothes/toys.',
 qualification='Repeatedly inspected development-time validation; predicted keypoints/mattes notGT; single seed, no newcamera generalization or largescene completion claim.',goal_complete=False))
p=LAB/'knowledge/32_dynamic_human_scene_reconstruction.md';text=p.read_text(encoding='utf8')
addition=r'''

### 姿态投影更准确之后，还要检查真实纹理

![同一原生头部区域，原照片、旧姿态重建与直接多视角姿态修正后的重建](../outputs/dynamic-human-enerf-direct-pose-065/paired-comparison/camera04-frame000755-dark_head.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实际训练 · 机位 04、第 755 帧，预留时刻。

将人体模型直接投影到各个相机，可以在图像平面优化姿态。设 $\boldsymbol\theta(t)$ 为随时间变化的关节旋转，$\boldsymbol\beta$ 为固定体形，$\mathbf J_j(\boldsymbol\theta(t),\boldsymbol\beta)$ 为第 $j$ 个三维关节点，$\mathbf y_{v,t,j}$ 为图像中的预测位置，则一类拟合目标为

$$
\mathcal L_{\mathrm{pose}}=
\frac{1}{\sum_{v,t,j}w_{v,t,j}}
\sum_{v,t,j}w_{v,t,j}\,
\rho_{\delta}\!\left[
\pi_v\!\left(\mathbf J_j(\boldsymbol\theta(t),\boldsymbol\beta)\right)-\mathbf y_{v,t,j}
\right]
+\lambda_{\mathrm{reg}}\mathcal L_{\mathrm{reg}}.
$$

$\pi_v$ 包含相机内外参；$w_{v,t,j}$ 表示置信度及可用观测权重；残差与鲁棒损失的尺度参数 $\delta$ 以像素计。关节旋转采用弧度，三维位置采用场景长度单位。正则项约束姿态变化与时间曲线，避免少量错误关键点任意扭曲身体。本地目标采用逐坐标的平滑鲁棒损失，尺度为 3 像素；只让训练时刻参与优化，另用中间时刻检查。

上图中的头部重影明显减少，说明原先固定头部局部旋转的连续模型确实缺少解释这段运动的自由度。不过本轮同时重新拟合了身体姿态，不能把所有变化只归因于头部解锁。

![同一次修正中的全人物画面，头部更清楚而手持物仍有明显拉花](../outputs/dynamic-human-enerf-direct-pose-065/paired-comparison/camera04-frame000755-full-person.png)
> 图源：同上 · 原照片、旧姿态重建、直接姿态修正后的重建 · 原生像素，无增强。

全图揭示了另一面：衣服花纹、手指与玩偶没有一同恢复，部分区域反而退化。六个固定头部区域的平均 PSNR 提高 4.26 dB，90 张预留时刻的全前景平均 PSNR 却下降 0.63 dB。因此，关键点拟合更准、局部图更清楚和整体重建更好，是三个需要分别验证的结论。当前保留旧身体运动基线，下一步单独检验头颈运动，再处理不服从人体骨架的表面变化。见[直接姿态修正的完整效果与负例](../experiments/37_dynamic_human_reconstruction.md#直接多视角姿态修正与头部运动)。

'''
if '### 姿态投影更准确之后，还要检查真实纹理' not in text:
 anchor='### 连续运动中的同预算对照';assert anchor in text;p.write_text(text.replace(anchor,addition+anchor),encoding='utf8')
p=LAB/'experiments/37_dynamic_human_reconstruction.md';text=p.read_text(encoding='utf8')
addition='''

### 直接多视角姿态修正与头部运动

在上一轮估计拍摄时间的分支上，固定形状、尺度、相机和时间偏移，直接拟合训练照片中的人体关键点，增加颈部与头部的连续旋转自由度。七个偶数时刻的 3,185 个观测参与 3,000 次优化；白衣人物背向相机时不使用鼻子和眼睛，手指没有作为 SMPL 约束。模型仍没有独立手指和玩偶运动。本次同时改动全身姿态拟合与头颈自由度，属于组合改动，尚不是只改变头部的消融。

为避免重复计算无关网格顶点，实现只计算目标关节及面部／脚部所需顶点；实际前向值与姿态梯度先同完整 SMPL 比对，最大差异分别为 1.79×10⁻⁷ 与 9.54×10⁻⁷。完成后又在 CPU 重新运行全部 195 个完整人体状态，核对仿射变换及全部投影残差。训练使用的观测编号全部为偶数，没有将奇数时刻关键点加入优化。

预留时刻中，身体与脚部的误差中位数从 5.938 降至 4.132 像素；头部从 7.550 降至 3.197 像素。但头部均方根误差仍有 31.341 像素，说明错误关键点和遮挡大残差没有消失；这些都是相对于网络预测的检查，不是真实关节误差。

随后从与上一轮完全相同的高斯初值开始，使用相同 105 张照片、抽样次序、背景、优化规则和 15,000 次更新，得到 238,502 个高斯。全部 102 张原生渲染的来源、PNG 像素和指标均独立复核；同样检查预先固定的 12 张全人物图、24 个细节区域和最差负例。

![头部细节中的明确改善](../outputs/dynamic-human-enerf-direct-pose-065/paired-comparison/camera04-frame000755-dark_head.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站运行 · 原照片、上一轮姿态、本轮直接拟合。

六个固定头部区域中有五个 PSNR 改善，平均提高 4.2594 dB，目视可见眼镜、脸部和头发的重影减少。与此同时，六个黑衣前臂区域平均下降 0.7443 dB，白衣前臂下降 1.2216 dB，鞋部下降 0.6765 dB。细节矩形中包含背景和遮挡，不能将这些矩形指标当作精确的人体部位分割指标。

![保留头部改善与其他区域问题的全图](../outputs/dynamic-human-enerf-direct-pose-065/paired-comparison/camera04-frame000755-full-person.png)
> 图源：同上 · 第 755 帧，预留时刻。

全前景平均 PSNR 从 24.1344 降至 23.5000 dB，90 张图中仅 12 张改善；15 台相机的平均变化均为负。最差变化为机位 16、第 745 帧，下降 2.7917 dB，玩偶出现更强的横向条纹。**本轮整体候选不替换上一轮基线，清晰人物重建仍未通过。**

![全前景指标下降最大的真实负例](../outputs/dynamic-human-enerf-direct-pose-065/paired-comparison/camera16-frame000745-full-person.png)
> 图源：同上 · 原照片、上一轮姿态、本轮直接拟合。

75 组连续差分残差的平均值从 0.04152 增至 0.04302，四段六帧视频均完成解码检查。该量包含运动、遮挡与掩码误差，不能等同于真实运动或闪烁；目视范围仍为上述固定离散画面。视频：[01](../outputs/dynamic-human-enerf-direct-pose-065/temporal-comparison/camera01-reserved-time-comparison.mp4)、[04](../outputs/dynamic-human-enerf-direct-pose-065/temporal-comparison/camera04-reserved-time-comparison.mp4)、[08](../outputs/dynamic-human-enerf-direct-pose-065/temporal-comparison/camera08-reserved-time-comparison.mp4)、[17](../outputs/dynamic-human-enerf-direct-pose-065/temporal-comparison/camera17-reserved-time-comparison.mp4)。4 帧/秒只表示展示顺序，不认证拍摄速度。

下一步采用更明确的分离对照：保留原有身体运动和世界平移，只优化颈部、头部的连续旋转；若头部改善可保留，再研究手臂、衣物及手持物的残余运动。此处使用的是反复检查过的开发验证时刻，尚未完成新相机与新片段的最终泛化验证。

报告：[本轮结论](../outputs/dynamic-human-enerf-direct-pose-065/final-summary.json)、[姿态优化条件](../outputs/dynamic-human-enerf-direct-pose-065/protocol.json)、[前向与梯度比对](../outputs/dynamic-human-enerf-direct-pose-065/subset-forward-gradient-audit.json)、[全部人体状态复算](../outputs/dynamic-human-enerf-direct-pose-065/independent-audit.json)、[姿态误差摘要](../outputs/dynamic-human-enerf-direct-pose-065/projection-summary.json)、[原生图像复算](../outputs/dynamic-human-enerf-direct-pose-065/direct_reprojection/native-evaluation/independent-image-audit.json)、[成对指标](../outputs/dynamic-human-enerf-direct-pose-065/paired-comparison/summary.json)、[连续图像检查](../outputs/dynamic-human-enerf-direct-pose-065/temporal-comparison/summary.json)、[实际看图范围](../outputs/dynamic-human-enerf-direct-pose-065/visual-review.json)。
'''
if '### 直接多视角姿态修正与头部运动' not in text:p.write_text(text+addition,encoding='utf8')
p=SITE/'scripts/sync-gs-lab.mjs';text=p.read_text(encoding='utf8')
if 'dynamic-human-enerf-direct-pose-065/${name}' not in text:
 lines=text.splitlines(keepends=True);idx=next(i for i,l in enumerate(lines) if 'dynamic-human-enerf-articulated-control-064/${name}' in l)+1
 lines.insert(idx,"  ...['final-summary.json','protocol.json','subset-forward-gradient-audit.json','independent-audit.json','projection-summary.json','direct_reprojection/native-evaluation/independent-image-audit.json','paired-comparison/summary.json','temporal-comparison/summary.json','visual-review.json','paired-comparison/camera04-frame000755-dark_head.png','paired-comparison/camera04-frame000755-full-person.png','paired-comparison/camera16-frame000745-full-person.png',...['01','04','08','17'].map(c => `temporal-comparison/camera${c}-reserved-time-comparison.mp4`)].map(name => [`outputs/dynamic-human-enerf-direct-pose-065/${name}`, '直接人体姿态与真实重建']),\n");p.write_text(''.join(lines),encoding='utf8')
p=LAB/'manifests/dynamic-human-work.json';m=read(p);m['status']='direct_pose_control_complete_head_improves_global_candidate_rejected';m['current_runs']=[];m['outdoor_direct_pose']='outputs/dynamic-human-enerf-direct-pose-065/final-summary.json';m['current_best_continuous_control']='outputs/dynamic-human-enerf-articulated-control-064/estimated_offsets';m['last_recorded_utc']=datetime.now(timezone.utc).isoformat();m['goal_turn_classification']='Meaningful progress: completed timing controls and directpose reconstruction; actualhead clarity improvement isolated as candidate mechanism, wholebodychange rejected on90imagevalidation. No external blocker.'
m['completed'].append('065actual15000update reconstruction,102nativepixel audits,12full24detailplusworstreview,75temporalchecks4decodedvideos; headROIs+4.26dB butfullforeground-0.63dB,globalcandidate rejected')
m['next_actions']=['Optimize only neck/head continuous rotations with064body pose/translation/timing/shape fixed; head-only trainingtargets, no validationfit','Train identical15000update Gaussian reconstruction and compare all90developmenttimes plus fixed regions; retain unchanged-body control','Then evaluate local residual deformation for limbs/clothes and separate handheldobject motion; currentSMPL binding is insufficient','Once development clarity improves, verify separate cameras/times before claiming generalization or large-scene completion'];save(p,m)
shutil.copy2(__file__,OUT/Path(__file__).name);print(json.dumps(dict(validation=pair['validation_foreground'],details=detail_summary),ensure_ascii=False))
