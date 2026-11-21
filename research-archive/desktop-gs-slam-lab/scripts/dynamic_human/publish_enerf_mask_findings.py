from pathlib import Path
import json
LAB=Path(__file__).resolve().parents[2]
knowledge=LAB/'knowledge/32_dynamic_human_scene_reconstruction.md'
text=knowledge.read_text(encoding='utf8')
anchor='### 同一帧号不一定对应同一运动状态'
addition=r'''前景估计还会犯相反的错误：把真实的人体像素当作背景删除。若真实表面点 $\mathbf X$ 在某个可见机位的投影满足 $M_v(\pi_v(\mathbf X))=0$，严格轮廓交集就有 $\mathbf X\notin\mathcal V$。其他机位看见这块表面，也不能在“所有机位必须通过”的规则下保住它。这里描述的是初始化的集合约束；后续优化是否能重新恢复这部分几何，仍取决于表示、梯度和剩余观测。

![原照片中可见的前臂、作者透明度造成的缺失与输入分割候选](../outputs/dynamic-human-enerf-mask-quality-038/camera01-full-person-sam2-comparison.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap，SAM 2.1 分割候选 · 三列均由原照片与掩码合成，属于输入诊断，不是三维渲染。

先看左侧原照片，黑衣人物的前臂和手实际可见；中间透明度合成却抹去了一部分皮肤；右侧分割候选保留了它。若源图先乘此前景透明度，表面颜色观测也会随之受损。因此，检查清晰度应同时对照原照片、输入掩码和真实渲染。分割网络的高置信度不代表掩码正确，人物相互遮挡、手持物和衣物内部都要逐一检查。当前候选仅用于有界训练对照，尚未证明人体重建变清楚；见[输入掩码与训练效果的区别](../experiments/37_dynamic_human_reconstruction.md#前景掩码漏掉可见人体后的输入对照)。

'''
if '严格轮廓交集就有' not in text:
    assert text.count(anchor)==1;knowledge.write_text(text.replace(anchor,addition+anchor),encoding='utf8')
page=LAB/'experiments/37_dynamic_human_reconstruction.md';text=page.read_text(encoding='utf8')
old='保留这一结果后，另启动从相同模型与 Adam 状态继续 9,000 次更新的固定预算验证，使用相同图像和相机。'
new='从相同模型与 Adam 状态继续 9,000 次更新的验证现已完成，累计 10,000 次；图像、相机和来源机位保持不变。三个留出机位的人物前景 PSNR 仅提高 0.16～0.21 dB，完整有效图像 PSNR 反而降低约 0.20～0.41 dB。全部六组人物对照和三张完整留出渲染已查看，手、玩具、交叉腿部仍模糊，轮廓黑色点状伪影仍存在。**增加当前训练预算没有达到清晰度要求。**'
if old in text:text=text.replace(old,new)
addition2='''

![相同帧的原透明度参照、一千次与一万次更新的真实渲染](../outputs/dynamic-human-4k4d-official-035/continuation-10000/paired-comparison/camera06-frame000750-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站运行 · 机位 06，第 750 帧 · 原生像素，无增强。

这三列中的左列仍是有误差的作者透明度合成参照。不能因为预测更接近一个错误的参照，就认为人体更完整。见[较长训练的结果](../outputs/dynamic-human-4k4d-official-035/continuation-10000/final-summary.json)与[实际看图范围](../outputs/dynamic-human-4k4d-official-035/continuation-10000/visual-review.json)。

### 前景掩码漏掉可见人体后的输入对照

在第 750 帧训练机位 01 的原照片中，黑衣人物的前臂实际可见，但作者透明度在其中一个人工检查的内部像素为零；另外三个内部像素也只保留约 0.11～0.43 的透明度。严格要求全部 15 个训练轮廓通过的视觉外壳初始化，可能因此剔除对应三维候选点；源图乘透明度后也会损伤该处颜色。当前尚未测量被剔除的准确体积，不能把所有模糊归因于这一项。

已用官方 SAM 2.1 Hiera Tiny 在 CPU 上处理 15 个训练视角，分别对两个人物和手持物提供人工框／点提示。整组单框会漏掉另一人，自动多点方案会在衣服和鞋上挖洞，均未采用。单独物体提示仍在遮挡处留下孔洞，因此最终候选只在分割掩码形态学闭运算的支持范围内，补回原透明度大于 0.95 的像素；核的尺寸为原生分辨率下 49×49 像素。曾尝试向外扩展支持范围，但会带回墙面阴影，该版本已保留为失败记录。

15 个最终候选均检查了原图、原透明度和新掩码的完整人体对照，并独立核对保存图中的像素。它们保留了可见前臂、衣物和鞋的主要内部区域，但靠近身体的墙面残留与硬边界仍存在。**这些图片是原照片与分割掩码的输入合成，不能当作模型已经重建清晰的效果。** 它们只进入有界的训练输入对照，不是真值标注；没有使用第 750 帧留出相机 RGB 生成候选。参见[输入检查结论](../outputs/dynamic-human-enerf-mask-quality-038/final-summary.json)、[视觉检查及剩余问题](../outputs/dynamic-human-enerf-mask-quality-038/visual-review.json)、[图像像素核验](../outputs/dynamic-human-enerf-mask-quality-038/independent-panel-audit.json)。

新对照保持原相机、原生像素、相同模型和 1,000 次更新预算，重新从训练机位掩码构建视觉外壳。变化涉及初始化、裁剪、源图乘掩码和训练监督的整个输入流程，不能称为仅修改损失函数的消融。另存的去畸变图像经过官方解码往返核对，原始照片和标定文件不改写；静态投影偏移候选 037 未应用。见[输入对照条件](../outputs/dynamic-human-4k4d-mask-control-039/protocol.json)。
'''
if '### 前景掩码漏掉可见人体后的输入对照' not in text:text+=addition2
page.write_text(text,encoding='utf8')
sync=Path('D:/Projects/EngineLab/engine-home/scripts/sync-gs-lab.mjs');text=sync.read_text(encoding='utf8')
if 'dynamic-human-enerf-mask-quality-038/${name}' not in text:
    lines=text.splitlines(keepends=True);idx=next(i for i,l in enumerate(lines) if 'dynamic-human-enerf-static-calibration-037/${name}' in l)+1
    lines.insert(idx,"  ...['continuation-10000/final-summary.json', 'continuation-10000/visual-review.json', 'continuation-10000/paired-comparison/camera06-frame000750-full-person.png'].map(name => [`outputs/dynamic-human-4k4d-official-035/${name}`, '一万次更新真实渲染；清晰度未通过']),\n  ...['camera01-full-person-sam2-comparison.png', 'final-summary.json', 'visual-review.json', 'independent-panel-audit.json'].map(name => [`outputs/dynamic-human-enerf-mask-quality-038/${name}`, '原图与输入掩码诊断；分割图不是重建结果']),\n  ['outputs/dynamic-human-4k4d-mask-control-039/protocol.json', '原生人体输入掩码对照条件'],\n")
    sync.write_text(''.join(lines),encoding='utf8')
print('Updatedtheory/evidenceandsourceallowlist')
