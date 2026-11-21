"""Bounded knowledge update; keep training artifacts and licensed templates local."""
from pathlib import Path
import json,shutil
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-enerf-mano-surface-099';Q=L/'outputs/dynamic-human-enerf-hand-prior-control-098';SITE=Path('D:/Projects/EngineLab/engine-home')
read=lambda p:json.loads(p.read_text(encoding='utf8'))
assert read(O/'final-summary.json')['visual_review_complete'] and read(Q/'final-summary.json')['clarity_accepted'] is False
p=L/'knowledge/32_dynamic_human_scene_reconstruction.md';s=p.read_text(encoding='utf8');assert '#### 手部表面高斯与关节运动' not in s;shutil.copy2(p,O/'knowledge-before.md')
start=s.index('增加运动自由度同样需要画面验证。');end=s.index('手部参数模型用共同的形状和关节连接',start)
s=s[:start]+'''增加运动自由度也不能自动恢复手指。在相同多视角观测下，为每个高斯增加连续平移残差，仍然可能得到连续移动的碎片。手部关节结构提供了更强的形状约束，但它首先要与照片中的姿态和遮挡相符。[连续运动对照](../experiments/37_dynamic_human_reconstruction.md#局部连续运动自由度的同预算对照)保留具体条件。

'''+s[end:]
start=s.index('**重投影一致性与照片重建质量需要分别验证。**');end=s.index('### 相机与人物必须对应同一个拍摄时刻',start)
replacement=r'''**重投影一致性与照片重建质量需要分别验证。** 共享模型使多视角关节预测更一致，但仍有指尖偏移。把已有高斯的位置和运动软约束到这个候选表面后，高斯确实更接近模型，实际渲染却仍有破碎的手指。几何先验的损失下降，不能代替照片重建质量的检验。[共享模型检查](../experiments/37_dynamic_human_reconstruction.md#连续手部模型的跨视角检验)与[先验和表面高斯对照](../experiments/37_dynamic_human_reconstruction.md#手部先验与关节表面高斯的图像检验)保留实验条件。

#### 手部表面高斯与关节运动

让高斯直接附着在手部三角面上，可以把关节运动同时传递给中心位置和协方差方向。以下是本地验证采用的参数化。对三角面$i$，以三个顶点的平均位置为中心，以两条正交切向量与单位法向量组成旋转矩阵$\mathbf F_i(t)$：

$$
\begin{aligned}
\mathbf c_i(t)&=\frac{\mathbf v_{i1}(t)+\mathbf v_{i2}(t)+\mathbf v_{i3}(t)}{3},\\
\mathbf F_i(t)&=\begin{bmatrix}\mathbf e_{i1}(t)&\mathbf e_{i2}(t)&\mathbf n_i(t)\end{bmatrix},\\
\boldsymbol\mu_i(t)&=\mathbf c_i(t)+\mathbf F_i(t)\boldsymbol\delta_i,\\
\boldsymbol\Sigma_i(t)&=\mathbf F_i(t)
\begin{bmatrix}
\sigma_{i1}^{2}(t)&0&0\\
0&\sigma_{i2}^{2}(t)&0\\
0&0&\sigma_{in}^{2}(t)
\end{bmatrix}
\mathbf F_i(t)^{\mathsf T}.
\end{aligned}
$$

$\boldsymbol\delta_i$是可训练的局部位置偏移；$\sigma_{i1},\sigma_{i2}$控制沿表面的覆盖范围，$\sigma_{in}$控制法向厚度，均采用场景长度单位。手指弯曲时，三角面的位置与方向随关节变化，高斯也随之平移、旋转。局部偏移和尺度保留有限调整能力，以容纳模板与真实表面的差异。

本地对照在每个手部三角面放置一个高斯，共1538个；学习共享的颜色、不透明度、局部偏移与尺度。一组固定先验姿态，另一组允许原始RGB损失小幅修正连续关节姿态。两组都与已有身体、玩偶高斯共同进行深度排序和透明度合成，使用相同的105张训练照片及6000次更新。

![关节表面高斯在留出时刻的实际渲染](../outputs/dynamic-human-enerf-mano-surface-099/native-articulated-hand-comparison.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 上为04机位，下为13机位，均为留出时刻731帧 · 原照片、原模型、固定手部姿态、图像优化手部姿态 · 原生像素。

__VISUAL_FINDING__

判断新表面是否真正参与图像，应检查它在最终像素中的合成权重：

$$
w_{\mathrm{hand}}(\mathbf p,t)=
\sum_{i\in\mathcal H}T_i(\mathbf p,t)\,\alpha_i(\mathbf p,t),
\qquad
T_i=\prod_{j<i}\left(1-\alpha_j\right).
$$

$\mathcal H$表示新增手部高斯，$\alpha_i$为投影后的有效不透明度，$T_i$为到达该高斯前的透射率。三角面存在、参数得到梯度，并不保证它在相机中有足够的可见贡献；已有表面的深度与不透明度也参与合成。贡献大同样不代表表面正确，仍需同时对照原照片、未训练机位和连续时刻。具体参数、对照结果与贡献检查见[实验附录](../experiments/37_dynamic_human_reconstruction.md#手部先验与关节表面高斯的图像检验)。

'''
finding=read(O/'final-summary.json')['knowledge_visual_finding'];replacement=replacement.replace('__VISUAL_FINDING__',finding);s=s[:start]+replacement+s[end:];p.write_text(s,encoding='utf8')
p=L/'experiments/37_dynamic_human_reconstruction.md';s=p.read_text(encoding='utf8');shutil.copy2(p,O/'evidence-before.md');assert '### 手部先验与关节表面高斯的图像检验' not in s
results=read(O/'final-summary.json');s+='''

### 手部先验与关节表面高斯的图像检验

首先检验软先验：在6330个可更新高斯中，按参考时刻距离、手腕范围和既有多相机支持筛选1937个，与连续MANO三角面建立对应；其余上下文高斯保留原有光度优化。使用与无先验逐点运动模型相同的105张训练照片、3000次更新及随机背景。增加鲁棒表面／运动损失后，对候选模型的平均加权跟踪距离从0.02694降至0.00837个场景单位，但这不是相对于真实手部几何的误差。15个同时刻手部窗口相对无先验模型仅改善约0.040 dB，6个固定目标窗口约0.027 dB；原生图中张手仍破碎，局部黑斑与粘连未消除。该候选未替换现有模型。[软先验结论](../outputs/dynamic-human-enerf-hand-prior-control-098/final-summary.json)、[近景与连续帧检查](../outputs/dynamic-human-enerf-hand-prior-control-098/visual-review.json)。

随后比较两种关节表面高斯：每组增加1538个三角面高斯，固定原模型的几何、颜色与运动，只允许6330个原上下文高斯调整不透明度。新高斯初始不透明度0.02，颜色在时间和相机之间共享。两个切向标准差初始化为三条边长均方根的0.55倍，法向为0.0825倍，各轴可乘以1/3至3的有界尺度因子；局部偏移每个坐标限制在场景尺度的0.005倍以内。姿态优化组额外学习9个时间控制点的轴角和手腕位置残差，各坐标分别限制在0.25弧度和场景尺度的0.01倍以内；固定姿态组保留零残差。

两组从同一原模型出发，用相同6000次更新、相同照片和背景顺序；与3000次软先验实验的预算不同，只有两组关节表面高斯构成直接同预算对照。00、06、12机位及全部奇数时刻均不参与拟合。总计每组134张原生1920×1080图像，含90张已知机位的留出时刻、18张留出机位的留出时刻、3张留出机位的732帧；其余为训练范围抽查。完整对比保留原图、全身、手部与玩偶上下文，窗口不是精确的手部语义掩码。

![实际手部渲染对照](../outputs/dynamic-human-enerf-mano-surface-099/native-articulated-hand-comparison.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 04、13机位731帧 · 原照片、原模型、固定姿态、图像优化姿态 · 未缩放的原生窗口。

'''+results['evidence_result_paragraph']+'''

两组冻结参数与6000步优化器状态核对通过，新增高斯与原高斯共同参与实际渲染；附加标记通道记录新增高斯的透射率加权不透明度，三个固定图像检查确认标记通道不改变RGB结果。另检查04、13机位720至732帧的连续图像及相邻帧残差。该区间反复用于开发，结果不代表独立场景验证，也不代表已完成街景背景联合重建。[结果与参数边界](../outputs/dynamic-human-enerf-mano-surface-099/final-summary.json)、[画面检查](../outputs/dynamic-human-enerf-mano-surface-099/visual-review.json)、[可见贡献](../outputs/dynamic-human-enerf-mano-surface-099/contribution/summary.json)。
''';p.write_text(s,encoding='utf8')
resources={'dynamic-human-enerf-hand-prior-control-098':['final-summary.json','visual-review.json'],'dynamic-human-enerf-mano-surface-099':['final-summary.json','visual-review.json','native-articulated-hand-comparison.png','contribution/summary.json']}
p=SITE/'scripts/sync-gs-lab.mjs';s=p.read_text(encoding='utf8');assert 'dynamic-human-enerf-mano-surface-099/${name}' not in s;lines=s.splitlines(keepends=True);i=next(i for i,l in enumerate(lines) if 'dynamic-human-hand-consensus-097/${name}' in l)+1
for root,files in resources.items():lines.insert(i,'  ...'+json.dumps(files)+'.map(name => [`outputs/'+root+'/${name}`, \'手部先验与关节表面高斯\']),\n');i+=1
p.write_text(''.join(lines),encoding='utf8');(O/'published-resources.json').write_text(json.dumps(dict(paths=resources,licensed_models_exported=False),indent=2)+'\n');shutil.copy2(__file__,O/Path(__file__).name);print('Existing chapter updated; six public PNG/JSON assets, no model files.')
