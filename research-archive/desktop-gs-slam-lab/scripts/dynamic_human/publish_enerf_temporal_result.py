"""Add a bounded temporal-motion explanation and real comparison images."""
from pathlib import Path
import json,shutil
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-temporal-basis-078';S=Path('D:/Projects/EngineLab/engine-home')
read=lambda p:json.loads(p.read_text(encoding='utf8'))
q=read(R/'final-summary.json')
assert q['actual_images_reviewed']==51 and q['retain_as_temporal_candidate'] and not q['goal_complete']
addition=r'''
### 时间基函数与局部运动

![原照片、三次多项式时间模型与增加局部时间基函数后的完整人物](../outputs/dynamic-human-enerf-temporal-basis-078/paired-comparison/camera04-frame000751-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap与本站实测 · 机位04、帧751 · 原生图像，无增强。

变形图决定空间上哪些高斯共同运动，时间基函数决定这种运动可以怎样变化。一个全局三次多项式用同一组系数描述整个片段；调整某一段运动时，也可能影响其他时刻。具有局部支撑的三次B样条可以增加局部时间变化的自由度。下式是本站的研究扩展，保留原有空间节点、身体运动与颜色损失。

令 $\tau=(t-750)/6$，$t$ 为连续帧索引。对每个节点的旋转向量和平移，在原多项式上增加

$$
\begin{aligned}
\boldsymbol\omega_k(\tau)&=\sum_{q=1}^{3}\mathbf a_{kq}\tau^q+\sum_{j=1}^{7}\mathbf u_{kj}\psi_j(\tau),\\
\mathbf d_k(\tau)&=\sum_{q=1}^{3}\mathbf b_{kq}\tau^q+\sum_{j=1}^{7}\mathbf v_{kj}\psi_j(\tau),\\
\psi_j(\tau)&=B_3\!\left(\frac{\tau-c_j}{h}\right)-B_3\!\left(\frac{-c_j}{h}\right).
\end{aligned}
$$

其中 $h=0.5$，$c_j\in\{-1.5,-1,-0.5,0,0.5,1,1.5\}$，均无量纲。减去参考值使 $\psi_j(0)=0$，所以参考时刻的残余变换仍为恒等变换。这里的局部支撑指原始B样条；减去参考常数后，$\psi_j$ 本身不再具有相同的紧支撑。

$$
B_3(x)=
\begin{cases}
\dfrac{2}{3}-x^2+\dfrac{|x|^3}{2},& |x|<1,\\
\dfrac{(2-|x|)^3}{6},&1\leq |x|<2,\\
0,& |x|\geq2.
\end{cases}
$$

该基函数及其前两阶导数连续。新增旋转系数 $\mathbf u_{kj}$ 的各分量限制在 $\pm0.03$ 弧度，平移系数 $\mathbf v_{kj}$ 的各分量限制在 $\pm0.01s$，其中 $s$ 为前文定义的场景尺度。新增系数从零开始，初始渲染与原模型一致；训练后仍需通过相同的旋转矩阵与协方差传播计算图像。

这次相同15000次更新的对照中，原机位留出时刻平均前景PSNR提高0.251dB，另外三个机位提高0.073dB。部分衣物和手持物更集中，但图中的抬脚、手指与交叉遮挡仍偏软。时间自由度的增加没有提供新的可见表面、手指关节或正确的遮挡对应，因此不能仅凭平均误差下降就认定细节已恢复。[局部时间表示的原图对照与条件](../experiments/37_dynamic_human_reconstruction.md#局部时间基函数与人物细节)保留了改进与失败区域。

'''
p=L/'knowledge/32_dynamic_human_scene_reconstruction.md';s=p.read_text(encoding='utf8')
marker='### 运动控制节点的密度与观测支持'
assert marker in s
if '### 时间基函数与局部运动' not in s:
    shutil.copy2(p,R/'knowledge-before.md')
    p.write_text(s.replace(marker,addition+marker,1),encoding='utf8')
evidence=f'''

### 局部时间基函数与人物细节

这组对照在原三次多项式运动上增加七个以参考时刻为零的三次B样条项，旋转和平移均可学习。保持105张原始训练照片、冻结身体运动、384个空间控制节点、52109个初始高斯、抽样顺序、颜色损失、正则项、增密日程与15000次更新一致。图运动参数从6912个增加到23040个；这是本地研究扩展，不是4K4D或OmniRe的官方方法复现。

训练约485秒，最终223105个高斯，框架记录的显存分配峰值约1.67GB。完成102张原机位、18张另机位的1920×1080渲染；实际查看37张预定对照、2张最差变化图与12张扩大后的手鞋区域，共51张。新增参数为零时与旧渲染逐像素一致，训练后四组运动参数均经历15000次更新；这些核对说明实现按预期工作，不等于清晰度达标。

原机位90张留出时刻的平均前景PSNR为26.253→26.504dB，72张改善；另外三个机位18张图为23.745→23.818dB，12张改善。其中机位06平均仍下降0.022dB。时间差分残差由0.03254→0.03163、0.04003→0.03945，分别对应两组机位；这个指标包含运动、遮挡与掩码误差，不能单独作为闪烁真值。七段视频已做编码与解码检查，本次视觉判断来自PNG对照。

![原照片、全局三次时间模型与增加局部时间基函数后的鞋部](../outputs/{R.name}/expanded-detail/camera04-frame000751-dark_shoe.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap与本站实测 · 机位04、帧751 · 相同原生像素区域。

全身外观和衣服图案已有可辨结构，新增时间项带来小幅改善，但抬起的鞋仍有拖影，手指仍粘连，细衣纹仍偏软。下面的另机位还保留胸前条纹与遮挡重影，不能用更好的平均分数掩盖这些问题。

![另机位中仍未消除的胸前条纹与遮挡错误](../outputs/{R.name}/heldout/paired-comparison/camera06-frame000745-full-person.png)
> 图源：同上 · 机位06、帧745 · 原照片、原时间模型、新时间模型。

保留该时间模型作为下一轮候选，旧开发基线继续保存。当前结果来自单一随机种子和反复使用的开发片段；下一步先在独立运动片段中保持参数不变做对照，再判断是否替换基线。**人物清晰度尚未达到要求，静态大场景与人体联合重建也未完成。**

报告：[训练条件](../outputs/{R.name}/protocol.json)、[真实渲染与梯度检查](../outputs/{R.name}/runtime-probe.json)、[共同条件](../outputs/{R.name}/training-protocol-audit.json)、[变形数值检查](../outputs/{R.name}/trained-graph-independent-audit.json)、[原机位指标](../outputs/{R.name}/paired-comparison/summary.json)、[另机位指标](../outputs/{R.name}/heldout/paired-comparison/summary.json)、[实际看图范围](../outputs/{R.name}/visual-review.json)、[本轮结论](../outputs/{R.name}/final-summary.json)。
'''
p=L/'experiments/37_dynamic_human_reconstruction.md';s=p.read_text(encoding='utf8')
if '### 局部时间基函数与人物细节' not in s:
    shutil.copy2(p,R/'evidence-before.md');p.write_text(s+evidence,encoding='utf8')
assets=['protocol.json','runtime-probe.json','training-protocol-audit.json','trained-graph-independent-audit.json','paired-comparison/summary.json','heldout/paired-comparison/summary.json','visual-review.json','final-summary.json','paired-comparison/camera04-frame000751-full-person.png','expanded-detail/camera04-frame000751-dark_shoe.png','heldout/paired-comparison/camera06-frame000745-full-person.png']
p=S/'scripts/sync-gs-lab.mjs';s=p.read_text(encoding='utf8')
if R.name+'/${name}' not in s:
    lines=s.splitlines(keepends=True)
    index=next(i for i,l in enumerate(lines) if 'dynamic-human-enerf-detail-weight-076/${name}' in l)+1
    lines.insert(index,'  ...'+json.dumps(assets)+f".map(name => [`outputs/{R.name}/${{name}}`, '局部时间基函数对照']),\n")
    p.write_text(''.join(lines),encoding='utf8')
(R/'published-resources.json').write_text(json.dumps(dict(paths=assets,licensed_models_exported=False),indent=2)+'\n',encoding='utf8')
shutil.copy2(__file__,R/Path(__file__).name)
print('078 bounded knowledge, actual comparisons and11explicit resources prepared.')
