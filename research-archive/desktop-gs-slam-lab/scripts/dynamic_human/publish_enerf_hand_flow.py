"""Integrate motion-versus-clarity theory into the existing local knowledge chapter."""
from pathlib import Path
import json,shutil
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-enerf-open-hand-flow-083';SITE=Path('D:/Projects/EngineLab/engine-home')
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
s=read(O/'final-summary.json');assert not s['adopt_as_development_baseline'] and s['actual_render_panels_reviewed']==36
p=L/'knowledge/32_dynamic_human_scene_reconstruction.md';txt=p.read_text(encoding='utf8');assert '#### 图像运动约束与重建清晰度' not in txt;shutil.copy2(p,O/'knowledge-before.md')
a=txt.index('局部图像跟踪也不能直接替代手指观测。');b=txt.index('### 相机与人物必须对应同一个拍摄时刻',a)
txt=txt[:a]+r'''#### 图像运动约束与重建清晰度

沿一条近似直线的边缘移动时，多个位置可能具有相似图块。这是光流中的孔径问题：局部亮度通常只能约束垂直于等亮度线的运动，不能唯一确定沿线方向的运动。往返跟踪一致、图块相关性很高，也未必代表完整的二维对应正确。[原生图块中的反例](../experiments/37_dynamic_human_reconstruction.md#局部图像对应的歧义)说明了这一点。

在图像梯度非零处，定义单位方向与有限帧间位移的法向分量

$$
\mathbf n(\mathbf p)=\frac{\nabla I(\mathbf p)}{\|\nabla I(\mathbf p)\|_2},\qquad
d_n(\mathbf p)=\mathbf n(\mathbf p)^{\mathsf T}\mathbf v(\mathbf p).
$$

$I$ 为灰度图像，$\mathbf p$ 为像素位置，$\mathbf v\in\mathbb R^2$ 为两张照片之间的位移，单位为像素。这里的“法向”是**图像等亮度线的法向**，不是三维表面法线。只约束 $d_n$ 可以保留沿边缘方向的不确定性；它仍须经过可见性、人物归属和遮挡筛选，不能把玩偶表面的运动当成手的运动。

对动态高斯，可先将同一高斯中心在两个时刻的投影相减，再按源图中的可见贡献加权：

$$
\widehat{\mathbf v}(\mathbf p)=
\frac{\sum_i\overline w_i(\mathbf p)
\left[\pi_c\!\left(\boldsymbol\mu_i(t')\right)-\pi_c\!\left(\boldsymbol\mu_i(t)\right)\right]}
{\sum_i\overline w_i(\mathbf p)}.
$$

$\boldsymbol\mu_i(t)$ 是第 $i$ 个高斯的世界坐标中心，$\pi_c$ 为相机投影；$\overline w_i$ 是由源图透明度与前方遮挡确定的贡献。此式是高斯**中心位移的图像近似**，没有精确描述一个高斯覆盖范围内所有表面点的运动。若为稳定优化而固定源图权重，还须在优化后按当前几何重新计算可见性，检查近似是否失效。

在原有图像与正则化损失 $\mathcal L_0$ 上加入一项运动约束，可写为

$$
\mathcal L=\mathcal L_0+
\frac{\lambda_n}{|\Omega|}\sum_{\mathbf p\in\Omega}
\rho\!\left(\frac{\mathbf n(\mathbf p)^{\mathsf T}\widehat{\mathbf v}(\mathbf p)-d_n(\mathbf p)}{\delta}\right).
$$

$\Omega$ 为通过筛选的可见手部像素，$\delta$ 是像素误差尺度，$\lambda_n$ 为无量纲权重，$\rho$ 为鲁棒损失。图像估计的 $d_n$ 仍是候选观测，不能视为真实三维运动。尤其是只覆盖手背和手腕的约束，并未提供每根手指的完整形状与屈伸信息。

![原照片、仅图像续训与增加手部运动约束后的原生重建](../outputs/dynamic-human-enerf-open-hand-flow-083/comparison/camera04-frame000731-both_hands_and_objects.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 机位04、留出帧731 · 左为原照片，中、右为两种真实高斯渲染。

上图右侧人物的原始手掌可辨认出弯曲手指；两种重建仍有皮肤与玩偶碎片重叠。本地对照只更新手附近的运动节点，保持高斯几何与外观不变。增加上述约束后，训练点的运动残差下降约35%，留出图像和手物区域的误差却反而增大。按当前几何重算可见性后，运动残差仍下降，因而不能把这项变化只归因于固定权重的近似。

这个结果说明，局部运动观测更一致，不等于已有表面和颜色已经正确，也不保证指尖变清楚。这里的约束覆盖不完整，且没有同时调整局部几何与外观；这两项仍需分别检验。该候选没有替换原基线。条件、全身图与不利案例见[手部法向运动与真实重建对照](../experiments/37_dynamic_human_reconstruction.md#手部法向运动与真实重建对照)。

'''+txt[b:];p.write_text(txt,encoding='utf8')
p=L/'experiments/37_dynamic_human_reconstruction.md';txt=p.read_text(encoding='utf8');assert '### 手部法向运动与真实重建对照' not in txt;shutil.copy2(p,O/'evidence-before.md')
txt+='''

### 手部法向运动与真实重建对照

本轮仍使用 actor5_6 帧720至732。先在15个训练机位的728、730、732帧运行官方 RAFT-Large 与 DIS 光流，并检查45个手部区域分割候选。排除玩偶混入的14、16机位后，在13个机位筛出1527个手背、手腕及少量皮肤褶皱对应。源、目标均须位于保守手部内部，往返误差低于0.5像素，两种算法沿图像梯度方向的位移差不超过1像素。只使用这个法向分量；这些观测不是完整光流真值，也不是手指关节标签。04机位正面张手区只有2个点，覆盖不足。

![原生手部照片与保留下来的对应](../outputs/dynamic-human-enerf-open-hand-flow-083/normal-flow/camera13-review.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站对应筛选 · 机位13 · 每行依次为源图、源图保留点、目标保留点、目标原图。

两支模型从同一个15000步检查点开始，各续训1500步。仅优化27个局部运动节点，其余357个节点，以及252984个高斯的几何、颜色、不透明度与协方差均精确冻结；身体运动、相机和时间偏移保持不变。两支使用相同105张训练照片、随机抽样、背景与正则化。候选额外加入权重0.05、误差尺度2像素的平滑L1法向运动项。训练分别约96秒，框架显存分配峰值约1.86GB；没有图像锐化、上采样或生成补画。

1527个训练观测的平均绝对法向残差由1.245降到0.807像素，下降35.2%。这是对估计运动的拟合程度。另按训练后几何重新计算可见性，残差为1.250→0.824像素，改善仍存在；因此不能仅用固定可见权重解释这个下降。

![相同留出时刻的完整人物重建](../outputs/dynamic-human-enerf-open-hand-flow-083/comparison/camera04-frame000731-full-person.png)
> 图源：同上 · 机位04、留出帧731 · 原照片、仅图像续训、加入法向运动约束。

评测保留120个原生1920×1080机位时刻：90张原训练机位的奇数帧、18张另外三个机位的奇数帧，以及12张训练诊断图。原掩码、前景口径与24个细节窗口均不变。两组留出图的平均前景PSNR分别为24.646→24.539dB、22.668→22.616dB；第一组27张改善、63张退化，第二组9张改善、9张退化。12个手与物品窗口平均变化−0.243dB，脚部窗口基本不变。时间差分残差也没有改善。这是同一场景的开发验证，不能宣称通过独立新场景验收。

![留出机位的手与玩偶仍然破碎](../outputs/dynamic-human-enerf-open-hand-flow-083/comparison/camera12-frame000731-both_hands_and_objects.png)
> 图源：同上 · 机位12、留出帧731 · 原生像素范围。

已查看24个细节对照与12个全身对照，包含两组留出图各自三个最差变化案例。04机位的张手仍有重影，06、07、12机位的手物交界仍有白黑碎片；大尺度衣服和身体可以辨认，手指没有稳定恢复。最差的原机位案例10/731下降1.254dB；另一组12/731下降0.548dB。**本轮候选不替换原基线，清晰人体与完整场景均未完成。**后续需检验局部几何、外观和运动的共同适配，不能继续把降低这项训练残差当作清晰度提升。

记录：[结论](../outputs/dynamic-human-enerf-open-hand-flow-083/final-summary.json)、[全部配对结果](../outputs/dynamic-human-enerf-open-hand-flow-083/comparison/summary.json)、[细节窗口](../outputs/dynamic-human-enerf-open-hand-flow-083/comparison/detail-metrics.json)、[独立数值复核](../outputs/dynamic-human-enerf-open-hand-flow-083/numerical-audit.json)、[实际看图范围](../outputs/dynamic-human-enerf-open-hand-flow-083/render-visual-review.json)、[当前可见性诊断](../outputs/dynamic-human-enerf-open-hand-flow-083/visibility-diagnostic/summary.json)。240张预测的数组、480张PNG与训练抽样一致性均已核验，原始数据与旧模型完整保留。
''';p.write_text(txt,encoding='utf8')
files=['final-summary.json','comparison/summary.json','comparison/detail-metrics.json','numerical-audit.json','render-visual-review.json','visibility-diagnostic/summary.json','normal-flow/camera13-review.png','comparison/camera04-frame000731-full-person.png','comparison/camera04-frame000731-both_hands_and_objects.png','comparison/camera12-frame000731-both_hands_and_objects.png']
p=SITE/'scripts/sync-gs-lab.mjs';txt=p.read_text(encoding='utf8');assert O.name+'/${name}' not in txt;lines=txt.splitlines(keepends=True);i=next(i for i,l in enumerate(lines) if 'dynamic-human-enerf-hand-motion-082/${name}' in l)+1;lines.insert(i,'  ...'+json.dumps(files)+f".map(name => [`outputs/{O.name}/${{name}}`, '手部法向运动与重建清晰度']),\n");p.write_text(''.join(lines),encoding='utf8')
save(O/'published-resources.json',dict(paths={O.name:files},licensed_models_exported=False));shutil.copy2(__file__,O/Path(__file__).name);print('Existing theory section updated; bounded evidence appended;10assets allowlisted')
