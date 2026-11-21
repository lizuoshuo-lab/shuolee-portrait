from pathlib import Path
import json,shutil
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-enerf-hand-trajectories-095';SITE=Path('D:/Projects/EngineLab/engine-home')
read=lambda p:json.loads(p.read_text(encoding='utf8'))
assert read(O/'final-summary.json')['adopt_as_development_baseline'] is False
p=L/'knowledge/32_dynamic_human_scene_reconstruction.md';s=p.read_text(encoding='utf8');shutil.copy2(p,O/'knowledge-before.md')
old='下一步围绕手部表面与局部连续运动的耦合继续验证，保留跨视角与跨时刻检查。[对照图、实验条件与验收结果](../experiments/37_dynamic_human_reconstruction.md#双视角拟合与局部密度控制)集中保留在实验记录中。'
assert old in s
s=s.replace(old,'[双视角与密度对照](../experiments/37_dynamic_human_reconstruction.md#双视角拟合与局部密度控制)保留完整实验条件。')
needle='### 相机与人物必须对应同一个拍摄时刻'
assert s.count(needle)==1
addition=r'''增加运动自由度同样需要画面验证。本地诊断在原连续模型上为手部高斯增加平移残差：

$$
\boldsymbol\mu_i(t)=\boldsymbol\mu_i^{\mathrm{base}}(t)+
\sum_{k=1}^{7}\left[b_k(\tau(t))-b_k(0)\right]\mathbf a_{ik},
\qquad \tau(t)=\frac{t-726}{6}.
$$

$t$ 采用帧为单位，$\tau$ 无量纲；$b_k$ 是三次B样条基函数，$\mathbf a_{ik}$ 与中心位置使用相同的场景长度单位。减去参考时刻的基函数值，使726帧的新增位移为零。两组实验分别用邻近控制节点插值得到系数，或让每个高斯拥有独立系数；同一高斯始终用同一个时间函数解释所有机位。这里新增的仅是平移，不包含协方差方向或局部颜色坐标系的旋转，也没有手指关节结构。

![连续运动自由度增加后仍未恢复清楚的手指](../outputs/dynamic-human-enerf-hand-trajectories-095/comparison/native-motion-hand-comparison.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 04、13机位的留出时刻731帧 · 原照片、原模型、节点平移、逐高斯平移 · 原生像素。

两组都完成了相同预算训练与跨视角、跨时刻检查，平均误差仅小幅改善；上图中张开的手掌仍破碎，逐高斯模型还有局部黑斑。**时间函数连续、可训练参数更多，不等于恢复了连续的手指表面**。下一步引入手部关节结构前，需要先验证照片中的姿态预测是否可信，再检查它与多视角几何、遮挡及运动是否一致，不能把一个形状完整的预测手模型直接当作观测真值。[条件、连续帧与完整结论](../experiments/37_dynamic_human_reconstruction.md#局部连续运动自由度的同预算对照)见实验记录。

'''
s=s.replace(needle,addition+needle);p.write_text(s,encoding='utf8')
p=L/'experiments/37_dynamic_human_reconstruction.md';s=p.read_text(encoding='utf8');shutil.copy2(p,O/'evidence-before.md');assert '### 局部连续运动自由度的同预算对照' not in s
s+='''

### 局部连续运动自由度的同预算对照

从保留的079连续模型出发，两组各训练3000步，使用完全一致的105张训练图、相机／帧／背景采样和表面学习率。只更新6330个手部邻域高斯的表面属性；其余246654个高斯、原384个运动节点、绑定、相机与时间偏移均冻结，无增密。新增平移场分别采用32个共享节点或6330条逐高斯轨迹，参数数为672和132930；时间基函数、位移限制和正则化相同。这是局部研究改动，不能称为官方方法复现。

各轨迹用7个紧支撑三次B样条，参考时刻726、归一化时间尺度6帧；系数以双曲正切限制在场景尺度的0.03倍。原始系数学习率0.003、末端衰减到三分之一；归一化位移平方均值权重0.1，相邻整数帧二阶差分平方均值权重10。两组只增加平移，没有新增协方差／球谐坐标旋转或手指关节监督。与更早的细节点实验相比还有学习率与正则化差异，不能混为只改变节点数的消融。

![相同连续时间监督下的手部原生对照](../outputs/dynamic-human-enerf-hand-trajectories-095/comparison/native-motion-hand-comparison.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 04、13机位731帧 · 原照片、原模型、节点平移、逐高斯平移。

90张已知机位留出时刻的平均前景PSNR由24.637变为24.663／24.664dB；18张原留出机位、留出时刻图由22.699变为22.717／22.722dB。15个同一时刻手部窗口均值仅提高0.031／0.033dB；6个目标右手窗口提高0.201／0.147dB。这些窗口包含衣物与手持物，不能解释为指尖误差。逐高斯分支在10机位729帧的全前景下降0.169dB，在12机位729帧下降0.145dB。

实际查看15个同一时刻手部窗口、6个目标手部窗口、12个双手与物品窗口、6组最差／原留出全身图，以及04／13机位各13个连续帧。全身轮廓仍可辨认，但两组手指都破碎或粘连；逐高斯模型在多个掌面视角出现黑斑。张手过程的后段也没有恢复清晰解剖结构。两组均不替换079开发对照。[完整结论](../outputs/dynamic-human-enerf-hand-trajectories-095/final-summary.json)、[实际看图范围](../outputs/dynamic-human-enerf-hand-trajectories-095/visual-review.json)。

连续对照：[04机位](../outputs/dynamic-human-enerf-hand-trajectories-095/temporal/camera04-hand-continuity.mp4)、[13机位](../outputs/dynamic-human-enerf-hand-trajectories-095/temporal/camera13-hand-continuity.mp4)。视频为5fps慢放查看副本，原始PNG逐帧保留；不是拍摄帧率声明。全部104个区域图、26幅对照与24组相邻差分已核验，实际看图覆盖全部26个时刻窗口。[时间评测](../outputs/dynamic-human-enerf-hand-trajectories-095/temporal/summary.json)。

134个原生1920×1080上下文、4个模型的1072张颜色／透明度PNG和指标已独立复核。新增位移在234个相机时刻上下文中以NumPy重建，最大差异小于5×10⁻⁹场景长度单位，参考时刻位移严格为零；这些是实现一致性检查，不是位移真值。[运动检查](../outputs/dynamic-human-enerf-hand-trajectories-095/motion-audit.json)。时间摘要中原先误写的PNG计数78已改为104，实际循环与数据本来就检查104张，图像和指标未改变。

该结果只说明当前平移残差设计与预算不足以恢复手部，不能推断所有局部运动表示都无效。所有机位仍属于反复使用的同场景开发验证，独立新场景、完整长序列和静态背景联合重建尚未验收。手部结构先验接入已开始：用户提供的SMPL+H与MANO模板保留在本地，先核验姿态预测和多视角投影，再决定是否用于训练；模型模板完整不等于重建结果清晰。
''';p.write_text(s,encoding='utf8')
root='dynamic-human-enerf-hand-trajectories-095';files=['final-summary.json','visual-review.json','motion-audit.json','comparison/native-motion-hand-comparison.png','temporal/summary.json','temporal/camera04-hand-continuity.mp4','temporal/camera13-hand-continuity.mp4']
p=SITE/'scripts/sync-gs-lab.mjs';s=p.read_text(encoding='utf8');assert root+'/${name}' not in s;lines=s.splitlines(keepends=True);i=next(i for i,l in enumerate(lines) if 'dynamic-human-enerf-hand-adaptive-094/${name}' in l)+1;lines.insert(i,'  ...'+json.dumps(files)+'.map(name => [`outputs/'+root+'/${name}`, \'局部连续运动对照\']),\n');p.write_text(''.join(lines),encoding='utf8')
(O/'published-resources.json').write_text(json.dumps(dict(paths={root:files},licensed_models_exported=False),indent=2)+'\n');shutil.copy2(__file__,O/Path(__file__).name);print('Two documents updated; seven research-only resources allowlisted.')
