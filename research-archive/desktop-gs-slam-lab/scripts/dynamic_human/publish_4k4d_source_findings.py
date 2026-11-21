from pathlib import Path
LAB=Path(__file__).resolve().parents[2]
knowledge=LAB/'knowledge/32_dynamic_human_scene_reconstruction.md';text=knowledge.read_text(encoding='utf8')
addition=r'''### 多视角颜色融合需要区分前景支持与表面可见性

![原照片、原颜色融合与排除前景外源颜色后的真实渲染](../outputs/dynamic-human-4k4d-source-selection-043/camera06-frame000750-source_foreground_gate-comparison.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站运行 · 同一几何、投影和透明度，机位 06 · 原生像素，无图像增强。

左列保留原始观测；中列使用原颜色融合；右列排除投影落在源图前景掩码外的颜色。右肩部分黑边减轻，但手、衣物和相互遮挡仍未恢复清晰。这个例子区分了“取色有效”与“表面正确”两个条件。

在图像驱动的颜色模型中，设三维点为 $\mathbf X$，源相机编号为 $s$，投影为 $\pi_s$，源图颜色为 $\mathbf C_s$，观察方向为 $\mathbf d$。一类模型用归一化权重融合源颜色，再叠加与观察方向有关的残差：

$$
\widehat{\mathbf c}(\mathbf X,\mathbf d)
=\sum_{s=1}^{S}w_s(\mathbf X)\,\mathbf C_s\!\left(\pi_s(\mathbf X)\right)
+\mathbf r(\mathbf X,\mathbf d),
\qquad w_s\geq0,\quad\sum_{s=1}^{S}w_s=1.
$$

$S$ 是参与融合的源机位数量，$w_s$ 无量纲，颜色采用归一化 RGB；$\mathbf r$ 可以由球谐函数等方向模型表示。这是用于解释图像驱动取色的形式，不能当作传统 3DGS 直接优化球谐系数的同一流程。

若源图已经乘以前景掩码，掩码外的黑色代表无效观测，不能直接当作物体表面很暗。令 $M_s$ 为源图前景支持，定义

$$
h_s(\mathbf X)=\mathbf 1\!\left[M_s\!\left(\pi_s(\mathbf X)\right)\geq\tau\right],
\qquad
\widetilde w_s(\mathbf X)=
\frac{w_s(\mathbf X)h_s(\mathbf X)}{\sum_j w_j(\mathbf X)h_j(\mathbf X)}.
$$

只有分母大于零时才使用右式。本地固定模型对照取 $\tau=0.5$，没有有效源机位时保留原融合，未擅自删除三维点。筛选依据是掩码，而不是颜色亮度，因此真实黑衣服仍可参与取色。

前景支持还不是遮挡可见性。背后的点投影到前方人的衣服上，仍满足 $h_s=1$，但源相机实际看见的是另一个表面。正确取色还依赖几何、深度排序、相机对应与拍摄时刻。简单减少源机位或只选择最大权重，可能让局部花纹更强，也可能把错误表面的颜色直接带进来；强烈的纹理不等于正确的重建。当前对照只证明部分边缘颜色受到无效源观测影响，尚未证明手指和衣物细节已恢复。完整条件见[颜色融合对照](../experiments/37_dynamic_human_reconstruction.md#前景支持筛选与源视角选择)。

'''
anchor='### 同一帧号不一定对应同一运动状态'
if '### 多视角颜色融合需要区分前景支持与表面可见性' not in text:
    assert text.count(anchor)==1;knowledge.write_text(text.replace(anchor,addition+anchor),encoding='utf8')
page=LAB/'experiments/37_dynamic_human_reconstruction.md';text=page.read_text(encoding='utf8')
addition2='''

### 前景支持筛选与源视角选择

固定同一个 2,000 次更新模型、几何、点半径、不透明度与源机位，使用同一批排序片元，仅改变源颜色的融合规则。排除源掩码小于 0.5 的观测并重新归一化后，六个视角的前景 PSNR 均提高，幅度约 0.09～0.36 dB；三个留出机位分别为 +0.36、+0.10、+0.32 dB。六组人物对照均已查看：部分肩部、袖口黑边减轻，但手、衣物图案、玩具和交叉肢体仍模糊。另做了对源颜色解除掩码预乘的控制，没有稳定优于仅筛选支持的方案。见[固定模型结果](../outputs/dynamic-human-4k4d-source-support-042/final-summary.json)与[18 张原生图像核验](../outputs/dynamic-human-4k4d-source-support-042/independent-audit.json)。

继续固定几何，对比仅保留最近两个源机位、将融合权重的四次方重新归一化、每点仅选择最大权重机位。最近两机位只改善了六个视角中的两个，另外两种方法在六个视角都降低前景指标。三个留出机位的三类对照共九张图已查看：最近机位有时让衣服图案更强，但肢体和衣物出现斑块；最大权重选择在白衣和皮肤上产生明显杂色。因此未采用强制源选择。全部 30 张原生图像与 24 张对照图均已核验，见[源选择结果](../outputs/dynamic-human-4k4d-source-selection-043/final-summary.json)及[独立核验](../outputs/dynamic-human-4k4d-source-selection-043/independent-audit.json)。

这些属于不改变模型参数的取色控制，尚不能证明修改训练流程后也有效。下一组从相同模型与优化器状态出发，原融合和前景支持融合各继续 1,000 次更新，累计均为 3,000 次；保持原投影和原透明度计算，以隔离融合规则的影响。见[成对训练条件](../outputs/dynamic-human-4k4d-source-gate-training-044/protocol.json)。
'''
if '### 前景支持筛选与源视角选择' not in text:page.write_text(text+addition2,encoding='utf8')
sync=Path('D:/Projects/EngineLab/engine-home/scripts/sync-gs-lab.mjs');text=sync.read_text(encoding='utf8')
if 'dynamic-human-4k4d-source-support-042/${name}' not in text:
    lines=text.splitlines(keepends=True);idx=next(i for i,l in enumerate(lines) if 'dynamic-human-4k4d-alpha-composition-041/${name}' in l)+1
    lines.insert(idx,"  ...['final-summary.json', 'independent-audit.json'].map(name => [`outputs/dynamic-human-4k4d-source-support-042/${name}`, '固定几何的源前景支持取色对照']),\n  ...['final-summary.json', 'independent-audit.json', 'camera06-frame000750-source_foreground_gate-comparison.png'].map(name => [`outputs/dynamic-human-4k4d-source-selection-043/${name}`, '真实颜色融合控制；部分边缘恢复']),\n  ['outputs/dynamic-human-4k4d-source-gate-training-044/protocol.json', '源前景支持成对训练条件'],\n")
    sync.write_text(''.join(lines),encoding='utf8')
print('Source-color theory and evidence prepared')
