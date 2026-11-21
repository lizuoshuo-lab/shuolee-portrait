from pathlib import Path
import json,shutil
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-hand-prior-096';SITE=Path('D:/Projects/EngineLab/engine-home')
assert json.loads((O/'final-summary.json').read_text(encoding='utf8'))['Gaussian_parameters_changed'] is False
p=L/'knowledge/32_dynamic_human_scene_reconstruction.md';s=p.read_text(encoding='utf8');shutil.copy2(p,O/'knowledge-before.md');needle='### 相机与人物必须对应同一个拍摄时刻';assert s.count(needle)==1
addition='''手部参数模型可以把任意点的运动，约束成有关节连接的候选手形。HaMeR从图像预测MANO手部形状与姿态；这个先验有助于提出手指结构，却不能直接确认照片中被遮挡的手指，也没有恢复皮肤纹理。

![可见手掌与遮挡手部的模型先验](../outputs/dynamic-human-hand-prior-096/prior-visible-versus-occluded.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 原照片；HaMeR与MANO本地推理 · 原照片、预测关节、预测网格 · 上为04机位732帧，下为03机位726帧 · 蓝色网格是先验预测。

上排的张开手掌提供了较清楚的结构候选；下排抓握位置的预测更不稳定，仅改变输入裁剪范围，就能使该图关节预测的中位变化达到约17像素。**蓝色网格形状完整，不代表照片中的人物已经重建清楚**。叠加预览尚未使用玩偶的遮挡深度，也不能凭网格盖住了玩偶，就断定隐藏手指的真实位置。应先用共享手部姿态、多视角投影及连续时间约束检验候选，再以软约束参与表面优化。当前尚未将整批预测作为训练真值，也没有把这组蓝色网格计为高斯重建改进。[先验接入与检验范围](../experiments/37_dynamic_human_reconstruction.md#手部关节先验的接入与可用性)。

''';s=s.replace(needle,addition+needle);p.write_text(s,encoding='utf8')
p=L/'experiments/37_dynamic_human_reconstruction.md';s=p.read_text(encoding='utf8');shutil.copy2(p,O/'evidence-before.md');assert '### 手部关节先验的接入与可用性' not in s
s+='''

### 手部关节先验的接入与可用性

用户提供的SMPL+H与MANO压缩包已接入本地。SMPL+H中性模型的6890个顶点、52个蒙皮关节和手指旋转梯度已检查；HaMeR使用独立MANO右手模板，输出778个顶点和21个关节。旧模板的Chumpy数组仅转换为数值数组，逐项数值完全一致，原文件与许可保留。模型文件、权重和原始三维参数均未加入网站资源。

官方HaMeR权重已完整下载、校验并加载，无缺失或多余键，加载后的全部MANO缓冲与用户模板相同。105张原训练图分别使用0.85、1.0、1.15三个手部上下文裁剪，完成315次推理；不使用00、06、12机位，也不使用奇数留出时刻。RGB顺序、裁剪仿射、归一化、关节重投影与旋转矩阵均已核验。推理阶段约33.7秒、峰值显存2.61GB；该时间不含下载、模型加载与后续CPU网格预览。

![结构先验在可见手掌与遮挡位置的不同表现](../outputs/dynamic-human-hand-prior-096/prior-visible-versus-occluded.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap；HaMeR、MANO与本地推理 · 左为原图，中为预测关节，右为预测网格；不是高斯训练结果。

全部105张图的关节裁剪扰动中位数再取中位数为1.37像素，第95百分位为3.40像素，但03机位726帧达到17.23像素。该例抓握与遮挡明显，预测不应直接当作精确标签。04／17机位沿用时间模型的曝光差约0.00894帧，7个偶数时刻的关节Sampson残差中位数为3.27至9.61像素，说明同语义关节的多视角一致性仍需要优化；这些并非人工标注或指尖真值，估计曝光差也不是硬件同步证明。

实际查看23组原生手部网格叠加（15机位732帧，加03／04／13／17机位720及726帧）和2组裁剪变化对照。张开手掌可提供连贯的手指候选；抓握位置仍有姿态偏差与不确定性。网格覆盖照片时没有模拟玩偶／身体遮挡，不能从覆盖区域直接推断隐藏手指正确与否。全批先验未作为硬标签，尚未更新高斯参数。[本轮结论与下一步](../outputs/dynamic-human-hand-prior-096/final-summary.json)、[实际看图记录](../outputs/dynamic-human-hand-prior-096/visual-review.json)。

下一步先在固定相机与现有估计曝光时间下，拟合跨机位共享、随时间连续的手部关节姿态；裁剪不稳定或被遮挡的预测只作为不确定的软观测。重新检查原照片投影后，再考虑关节驱动的局部高斯表面与纹理训练。接入可运行、形状完整的手模板，不等于完成清晰人物或大场景联合重建。

来源：[HaMeR论文](https://arxiv.org/pdf/2312.05251.pdf)、[作者项目页](https://geopavlakos.github.io/hamer/)、[官方实现](https://github.com/geopavlakos/hamer)、[MANO](https://mano.is.tue.mpg.de/)。本地CPU预览兼容修改仅补充OSMesa返回RGB时的深度覆盖通道，没有改变网络输出、顶点或相机；详细运行档案留在本地。
''';p.write_text(s,encoding='utf8')
root='dynamic-human-hand-prior-096';files=['final-summary.json','visual-review.json','prior-visible-versus-occluded.png'];p=SITE/'scripts/sync-gs-lab.mjs';s=p.read_text(encoding='utf8');assert root+'/${name}' not in s;lines=s.splitlines(keepends=True);i=next(i for i,l in enumerate(lines) if 'dynamic-human-enerf-hand-trajectories-095/${name}' in l)+1;lines.insert(i,'  ...'+json.dumps(files)+'.map(name => [`outputs/'+root+'/${name}`, \'手部关节先验检查\']),\n');p.write_text(''.join(lines),encoding='utf8');(O/'published-resources.json').write_text(json.dumps(dict(paths={root:files},licensed_models_exported=False),indent=2)+'\n');shutil.copy2(__file__,O/Path(__file__).name);print('Hand-prior evidence published locally in source;3assetsallowlisted.')
