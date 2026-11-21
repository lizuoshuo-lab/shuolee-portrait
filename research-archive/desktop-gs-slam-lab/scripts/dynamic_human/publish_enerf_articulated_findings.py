from pathlib import Path
import json,shutil,hashlib
LAB=Path(__file__).resolve().parents[2];SITE=Path('D:/Projects/EngineLab/engine-home');OUT=LAB/'outputs/dynamic-human-enerf-articulated-control-064'
assert json.loads((OUT/'final-summary.json').read_text(encoding='utf8'))['goal_complete'] is False
diagnostic=LAB/'outputs/dynamic-human-enerf-continuous-body-063/projection-diagnostic.json';d=json.loads(diagnostic.read_text());d.pop('records');d['full_local_report_sha256']=hashlib.sha256(diagnostic.read_bytes()).hexdigest();diagnostic.with_name('projection-summary.json').write_text(json.dumps(d,indent=2)+'\n')
p=LAB/'knowledge/32_dynamic_human_scene_reconstruction.md';text=p.read_text(encoding='utf8')
addition=r'''

### 连续人体运动如何带动高斯形状

![原照片、相同帧编号运动模型与考虑拍摄时间偏移后的连续人体重建](../outputs/dynamic-human-enerf-articulated-control-064/paired-comparison/camera04-frame000755-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实际渲染 · 机位 04、第 755 帧，预留时刻 · 三列均保留原生像素，无增强。

高斯随人体运动时，中心位置与空间形状必须使用一致的变换。设一个规范姿态高斯的中心为 $\boldsymbol\mu_0$，协方差为 $\boldsymbol\Sigma_0$。在局部仿射近似 $\mathbf x_t=\mathbf A_t\mathbf x_0+\mathbf b_t$ 下，有

$$
\boldsymbol\mu_t=\mathbf A_t\boldsymbol\mu_0+\mathbf b_t,
\qquad
\boldsymbol\Sigma_t=\mathbf A_t\boldsymbol\Sigma_0\mathbf A_t^{\mathsf T}.
$$

$\mathbf A_t\in\mathbb R^{3\times3}$ 表示局部线性变换，无量纲；$\mathbf b_t\in\mathbb R^3$ 为平移，单位为场景长度；协方差单位为长度的平方。前式移动中心，后式同时改变椭球的方向与尺度。只移动中心而保持原协方差，会使细长高斯在手臂转动后仍沿旧方向铺开。以上是高斯在仿射变换下的标准关系；用人体表面附近的局部变换近似衣物运动，是额外的建模假设。

本地实现以一个人体表面顶点作为每个高斯的运动参照。设该顶点在规范姿态和当前姿态的位置为 $\mathbf v_0,\mathbf v_t$，蒙皮产生的局部线性变换为 $\mathbf M_0,\mathbf M_t$，则在 $\mathbf M_0$ 可逆时取

$$
\mathbf A_t=\mathbf M_t\mathbf M_0^{-1},
\qquad
\mathbf b_t=\mathbf v_t-\mathbf A_t\mathbf v_0.
$$

因此，参照顶点必定从 $\mathbf v_0$ 运动到 $\mathbf v_t$；高斯允许相对顶点保留可学习的偏移，以覆盖衣物和头发。该关系没有保证宽松衣物、手指或手持物遵循同一运动，距离人体较远的玩偶尤其不能仅靠“绑定最近顶点”获得正确形变。

上图比较两种连续运动假设：相同文件帧编号代表相同时刻，以及每台相机带有估计的时间偏移。两组训练照片、初始高斯和更新预算相同。右列部分衣物纹理与抬脚轮廓更集中，但脸、手和玩偶仍模糊。90 张预留时刻图像的平均前景 PSNR 从 22.60 提高到 24.13 dB，79 张改善，仍有 11 张下降。这支持继续修正运动对应，不能当作清晰人体已经完成。

![同一预留时刻的脸部原生细节，时间偏移后仍有重影](../outputs/dynamic-human-enerf-articulated-control-064/paired-comparison/camera04-frame000755-dark_head.png)
> 图源：同上 · 原照片、相同帧编号模型、考虑拍摄时间的模型 · 固定头部区域。

脸部例子说明，时间模型与姿态模型需要分别检验。自由三维关节点轨迹拟合得好，转换为受骨骼结构限制的人体模型后，不一定保持同样的投影精度。此次模型转换使身体与脚部的预留观测误差中位数从 2.64 增至 5.94 像素；头部局部旋转还曾被固定。这些证据指向直接修正图像中的人体姿态，而不是继续增加高斯数量。完整条件、负例及连续效果见[连续人体重建对照](../experiments/37_dynamic_human_reconstruction.md#连续人体运动与拍摄时间的实际重建对照)。

'''
if '### 连续人体运动如何带动高斯形状' not in text:
 anchor='### 连续运动中的同预算对照';assert anchor in text;p.write_text(text.replace(anchor,addition+anchor),encoding='utf8')
p=LAB/'experiments/37_dynamic_human_reconstruction.md';text=p.read_text(encoding='utf8')
addition='''

### 连续人体运动与拍摄时间的实际重建对照

先把两种关节点轨迹拟合为连续 SMPL 姿态，得到“相同帧编号”与“估计拍摄时间”两组运动。人体形状与尺度共同固定为前一轮结果，局部颈部、头部和手部旋转当时没有可靠运动目标，保持不变。两组各完成 2,500 次姿态更新；每组 195 个相机与时刻组合均由 CPU 重新运行完整 SMPL，验证顶点、仿射矩阵和参照点运动。模型参数和变形数据仅保存在本机。

接着各自训练连续各向异性三维高斯 15,000 次，采用相同 52,109 个初始点、相同 105 张原生 1920×1080 照片、相同抽样次序、随机背景和优化规则。训练时刻是 744～756 中的七个偶数编号；六个奇数编号只用于开发验证。模型通过局部仿射变换同时携带高斯中心和完整协方差，颜色采用规范空间方向的二阶球谐函数。人体姿态在图像训练期间冻结。训练输入采用作者发布的前景透明度，不是人工语义真值；它与此前只修补第 750 帧的掩码实验条件不同。

两组最终分别有 237,066 与 249,560 个高斯。对每组全部 90 张预留时刻图像和 12 张训练诊断图，独立检查了原始照片、去畸变输入、408 张 PNG、102 张双列图及浮点误差指标。两组初始参数、抽样序列和最终优化器更新次数也逐项核对。没有锐化、插帧或将原照片融合进渲染结果。

估计拍摄时间的分支，预留时刻平均前景 PSNR 从 22.6015 提升到 24.1344 dB，90 张中 79 张改善、11 张下降；机位 08 平均下降 0.1926 dB。最差变化发生在机位 01、第 747 帧，下降 2.4975 dB，因此没有把平均改善写成逐帧改善。另有 12 张训练诊断图，平均提升 0.4124 dB。

![预留时刻的真实全人物对照](../outputs/dynamic-human-enerf-articulated-control-064/paired-comparison/camera04-frame000755-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站运行 · 左原照片，中相同帧编号，右估计拍摄时间。

![PSNR 下降最大的机位与时刻](../outputs/dynamic-human-enerf-articulated-control-064/paired-comparison/camera01-frame000747-full-person.png)
> 图源：同上 · 机位 01、第 747 帧 · 同一原生区域。

实际看图覆盖预先固定的 12 张全人物图、24 个头部／前臂／鞋部区域，并补看上述最差负例。抬脚边缘与部分衣物图案更集中，但脸部和眼镜重影、手指融合、鞋带模糊、玩偶拉花仍存在；遮挡处仍有厚边。**清晰人物重建仍未通过。**

连续检查另计算了全部 15 台相机的 75 组相邻预留时刻差分残差，平均值从 0.05359 降为 0.04152。该量同时包含运动、遮挡和掩码误差，不是光流补偿后的闪烁指标。四段视频均验证六帧完整解码；4 帧/秒仅用于按顺序观看，不代表经认证的实际拍摄速度。目视结论来自上述离散画面，没有声称逐帧观看全部视频。

视频：[机位 01](../outputs/dynamic-human-enerf-articulated-control-064/temporal-comparison/camera01-reserved-time-comparison.mp4)、[机位 04](../outputs/dynamic-human-enerf-articulated-control-064/temporal-comparison/camera04-reserved-time-comparison.mp4)、[机位 08](../outputs/dynamic-human-enerf-articulated-control-064/temporal-comparison/camera08-reserved-time-comparison.mp4)、[机位 17](../outputs/dynamic-human-enerf-articulated-control-064/temporal-comparison/camera17-reserved-time-comparison.mp4)。

这一结果仅是已知训练相机上的开发时间验证，不能替代新机位测试；机位 00、06、12 的相对拍摄时刻仍未在此实验中确定。估计的时间差只是候选解释，尚未认证真实相机时钟。人体模型本身还把预留身体关节点的投影误差中位数从自由轨迹的 2.636 增大到 5.938 像素，说明仍需改进姿态拟合。

报告：[本轮结论](../outputs/dynamic-human-enerf-articulated-control-064/final-summary.json)、[训练条件](../outputs/dynamic-human-enerf-articulated-control-064/protocol.json)、[真实渲染梯度与协方差检查](../outputs/dynamic-human-enerf-articulated-control-064/runtime-smoke.json)、[成对指标](../outputs/dynamic-human-enerf-articulated-control-064/paired-comparison/summary.json)、[第一组独立像素复算](../outputs/dynamic-human-enerf-articulated-control-064/fixed_offsets/native-evaluation/independent-image-audit.json)、[第二组独立像素复算](../outputs/dynamic-human-enerf-articulated-control-064/estimated_offsets/native-evaluation/independent-image-audit.json)、[连续图像检查](../outputs/dynamic-human-enerf-articulated-control-064/temporal-comparison/summary.json)、[实际目视范围](../outputs/dynamic-human-enerf-articulated-control-064/visual-review.json)、[连续人体检查](../outputs/dynamic-human-enerf-continuous-body-063/final-summary.json)、[全部仿射复算](../outputs/dynamic-human-enerf-continuous-body-063/independent-audit.json)、[人体投影残差](../outputs/dynamic-human-enerf-continuous-body-063/projection-diagnostic.json)。
'''
if '### 连续人体运动与拍摄时间的实际重建对照' not in text:p.write_text(text+addition,encoding='utf8')
p=SITE/'scripts/sync-gs-lab.mjs';text=p.read_text(encoding='utf8')
if 'dynamic-human-enerf-articulated-control-064/${name}' not in text:
 lines=text.splitlines(keepends=True);idx=next(i for i,l in enumerate(lines) if 'dynamic-human-enerf-pose-sequence-062/${name}' in l)+1
 lines.insert(idx,"  ...['final-summary.json','independent-audit.json','projection-diagnostic.json'].map(name => [`outputs/dynamic-human-enerf-continuous-body-063/${name}`, '连续人体模型与投影']),\n  ...['final-summary.json','protocol.json','runtime-smoke.json','paired-comparison/summary.json','fixed_offsets/native-evaluation/independent-image-audit.json','estimated_offsets/native-evaluation/independent-image-audit.json','temporal-comparison/summary.json','visual-review.json','paired-comparison/camera04-frame000755-full-person.png','paired-comparison/camera04-frame000755-dark_head.png','paired-comparison/camera01-frame000747-full-person.png',...['01','04','08','17'].map(c => `temporal-comparison/camera${c}-reserved-time-comparison.mp4`)].map(name => [`outputs/dynamic-human-enerf-articulated-control-064/${name}`, '连续人体真实重建对照']),\n")
 p.write_text(''.join(lines),encoding='utf8')
for target in [LAB/'experiments/37_dynamic_human_reconstruction.md',SITE/'scripts/sync-gs-lab.mjs']:
 value=target.read_text(encoding='utf8');target.write_text(value.replace('projection-diagnostic.json','projection-summary.json'),encoding='utf8')
shutil.copy2(__file__,OUT/Path(__file__).name);print('063/064 bounded knowledge and evidence prepared')
