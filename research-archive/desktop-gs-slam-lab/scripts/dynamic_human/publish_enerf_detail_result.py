"""Publish a bounded explanation and native examples, never model binaries."""
from pathlib import Path
import json,shutil
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-detail-weight-076';S=Path('D:/Projects/EngineLab/engine-home')
read=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
q=read(R/'final-summary.json');assert q['actual_images_reviewed']==50 and not q['adopt_as_development_baseline']
addition=r'''
颜色监督还取决于局部区域的权重。令 $\Omega_d$ 为手部、鞋部附近的有效像素窗口并集，保留原有目标 $\mathcal L_0$，则额外的局部颜色项可写为

$$
\mathcal L=\mathcal L_0+
\frac{\lambda_d}{3|\Omega_d|}
\sum_{\mathbf p\in\Omega_d}
\left\|\widehat{\mathbf I}_{\mathbf b}(\mathbf p)-\mathbf I^*_{\mathbf b}(\mathbf p)\right\|_1.
$$

$\lambda_d$ 是无量纲权重；分母的 $3$ 对应RGB三个通道。空窗口的额外项为零。窗口可以包含衣物、背景或另一个人物，只是采样范围，并非手部的语义真值。这个目标增加小区域对优化的影响，却没有增加新的视角或正确的表面对应；它也可能改变增密过程。

![原照片、原目标训练与增加手鞋局部颜色权重后的真实重建](../outputs/dynamic-human-enerf-detail-weight-076/expanded-detail/camera04-frame000751-dark_shoe.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap与本站实测 · 机位04、帧751 · 原生像素，无增强。

本地采用 $\lambda_d=0.2$ 的对照，在70张训练图中增加局部项，其余35张保留原损失。上图的抬脚仍有拖影，鞋带没有恢复到原照片的清晰程度；扩大后的手部对照也仍有指尖粘连。原机位未训练时刻的平均误差略降，另外三个机位总体退化，因此不能把增加局部权重等同于细节恢复。[局部颜色权重与跨视角清晰度](../experiments/37_dynamic_human_reconstruction.md#局部颜色权重与跨视角清晰度)保留了完整条件与结果。

'''
file=L/'knowledge/32_dynamic_human_scene_reconstruction.md';text=file.read_text(encoding='utf8');marker='### 增密梯度与细节恢复'
assert marker in text
if '颜色监督还取决于局部区域的权重' not in text:file.write_text(text.replace(marker,addition+marker,1),encoding='utf8')
evidence=f'''

### 局部颜色权重与跨视角清晰度

这组对照检验手部、鞋部附近的颜色误差是否被全身平均稀释。保持105张原始训练照片、前景透明度、正确RGB预处理后的冻结人体运动、初始化、384个局部变形节点、抽样顺序和15000次更新一致，只在70张图的局部窗口内增加权重为0.2的RGB绝对误差。另35张图因窗口中的可见部位存在明显错配而保留原损失。窗口不是关节或人体部位真值。

完成15000次更新，得到280237个高斯，训练约586秒，框架统计的显存分配峰值约2.05GB；最终高斯数会随新增损失的梯度而变化，这不是固定点数的消融。102张原机位与18张另机位的1920×1080渲染、原始像素裁图及误差均已复核。预定37张对照、额外一张最差变化图及12张扩大后的手鞋裁图，共50张图片实际查看。

![原照片、原目标与增加局部权重后的完整人物](../outputs/{R.name}/paired-comparison/camera04-frame000751-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap与本站实测 · 机位04、帧751 · 灰色为检查人物使用的合成背景。

原机位90张留出时刻的人物区域平均PSNR为26.253→26.304dB，52张改善；另外三个机位的18张图为23.745→23.628dB，仅2张改善。时间差分残差在原机位略降、另机位略升；这个量包含运动、遮挡和掩码误差，不是独立的闪烁真值。七段视频已完成编码与解码检查，本轮视觉判断来自上述50张PNG。

![扩大到完整手掌与指尖后的原生对照](../outputs/{R.name}/expanded-detail/camera04-frame000751-dark_forearm.png)
> 图源：同上 · 机位04、帧751 · 每列为相同原始像素范围。

主体、衣物图案和手持物已有可辨外观，但指尖仍粘连，抬起的鞋仍有拖影，胸前条纹和交叉遮挡的重影也没有解决。**这轮不替换保留的开发基线。** 对照只能说明当前窗口与权重组合不足以恢复细节，不能据此断言所有局部加权方法无效，或已经证明模糊只来自运动模型。下一步重点检查失败部位的局部形变与多视角对应。当前仍是同一场景、单一随机种子和反复使用的开发视角，不是最终独立测试；静态大场景与人体的联合重建仍未完成。

报告：[训练条件](../outputs/{R.name}/protocol.json)、[实际梯度检查](../outputs/{R.name}/residual_graph/detail-gradient-audit.json)、[共同条件核验](../outputs/{R.name}/training-protocol-audit.json)、[变形数值核验](../outputs/{R.name}/trained-graph-independent-audit.json)、[原机位指标](../outputs/{R.name}/paired-comparison/summary.json)、[另机位指标](../outputs/{R.name}/heldout/paired-comparison/summary.json)、[看图范围](../outputs/{R.name}/visual-review.json)、[最终判断](../outputs/{R.name}/final-summary.json)。
'''
file=L/'experiments/37_dynamic_human_reconstruction.md';text=file.read_text(encoding='utf8')
if '### 局部颜色权重与跨视角清晰度' not in text:file.write_text(text+evidence,encoding='utf8')
assets=['protocol.json','residual_graph/detail-gradient-audit.json','training-protocol-audit.json','trained-graph-independent-audit.json','paired-comparison/summary.json','heldout/paired-comparison/summary.json','visual-review.json','final-summary.json','paired-comparison/camera04-frame000751-full-person.png','expanded-detail/camera04-frame000751-dark_forearm.png','expanded-detail/camera04-frame000751-dark_shoe.png']
file=S/'scripts/sync-gs-lab.mjs';text=file.read_text(encoding='utf8')
if R.name+'/${name}' not in text:
    lines=text.splitlines(keepends=True);index=next(i for i,line in enumerate(lines) if 'dynamic-human-enerf-native-motion-tracks-075/${name}' in line)+1
    lines.insert(index,'  ...'+json.dumps(assets)+f".map(name => [`outputs/{R.name}/${{name}}`, '局部颜色权重对照']),\n");file.write_text(''.join(lines),encoding='utf8')
save(R/'published-resources.json',dict(paths=assets,licensed_models_exported=False))
shutil.copy2(__file__,R/Path(__file__).name)
print('076 theory, real comparisons, and bounded evidence prepared;11allowed assets.')
