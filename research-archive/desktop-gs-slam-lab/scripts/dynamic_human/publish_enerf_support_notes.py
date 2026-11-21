"""Append bounded support evidence and extend the existing local export allowlist."""
from pathlib import Path
LAB=Path(__file__).resolve().parents[2]
page=LAB/'experiments/37_dynamic_human_reconstruction.md'
section=r'''

### 运动阶段的尺度与不透明度对照

沿用局部时间对照的共同索引 34～58 和相同 375 张训练观测，固定 150,543 个高斯及其索引，分别继续训练：尺度／不透明度都固定、仅尺度可变、仅不透明度可变、两者均可变。四组共享已经完成 6,000 次优化的首帧状态，各自再进行 24×600＝14,400 次运动阶段更新；本轮新增有效更新共 57,600 次，不能把共享首帧算成四份新增训练。每组对应的完整历史为 20,400 次更新。

尺度学习率为 0.0005，对数尺度以初始值为中心限制在正负 $\ln 2$，相邻时刻对数尺度差的均方项权重为 0.01。不透明度对数几率学习率为 0.002，以初始值为中心限制在正负 2，实际不透明度相邻差的均方项权重为 0.01。中心、旋转、球谐颜色及其学习率、局部刚性／旋转／等距／颜色连续性项、采样顺序和原生像素均保持原设置。不进行分裂或剪枝。这个干预包括开放参数、边界与正则，不能解释成完全无约束的参数自由度比较。

首帧参数完全一致，但 CUDA 训练重做第一个运动时刻时出现数值差异，因而重新运行完整固定参数基线。该新基线与上一轮固定参数结果在 75 张图上的前景 PSNR 差值均值为 +0.00105 dB，平均绝对差为 0.02902 dB，最大绝对差为 0.13185 dB。这是一次同种子重做的数值敏感性检查，不是多随机种子置信区间；额外用于这项检查的 600 次更新未作为正式模型。

![四种尺度与不透明度条件的实际细节对照](../outputs/dynamic-human-enerf-support-030/comparison/camera00-frame046-person1.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站渲染 · 原始前景、两项固定、仅尺度变化、仅不透明度变化、两者变化；原生裁图，无增强。

四组使用完全相同的 75 张外观留出参照，平均前景 PSNR 依次为 **25.237、25.335、25.299、25.378 dB**。两者均变化相对新基线平均提高 0.14054 dB，63 张改善、9 张退化，3 张共同初始图不变。前景亮度 Sobel 梯度误差从 0.010255 降到 0.009997；它衡量与原图边缘的一致程度，不是越锐越好的锐度分数。轮廓 IoU 从 0.96138 到 0.96168，作者掩膜包含手持物和边缘错误，不能当作精确人体几何真值。三个机位都曾在独立早期图像上参与时间对应检查，因此这里仅留出外观训练；没有时间留出，且属于开发序列。

![同时开放两参数的最差配对退化](../outputs/dynamic-human-enerf-support-030/comparison/regression-camera12-frame044-person1.png)
> 图源：ENeRF-Outdoor 与本站渲染 · 机位 12，索引 44 · 原始前景、两项固定、两者变化。

最差配对在机位 12、索引 44，前景分数下降 0.19905 dB；两个人物及前后相邻帧均已看图。另完成三个训练机位 01、07、14 在 34、46、58 的 36 张诊断渲染，每组 9 张。训练前景 PSNR 均值由固定组的 29.637 到两者变化组的 30.159 dB，但后段训练图中的手和物品仍有混合、拉丝。训练均值与留出均值来自不同机位及数量，不可直接将差值解释成纯泛化差距。

![训练视角中尚未解决的运动细节](../outputs/dynamic-human-enerf-support-030/training-comparison/camera07-frame058-person1-compact.png)
> 图源：ENeRF-Outdoor 与本站渲染 · 训练机位 07，索引 58 · 原始前景、两项固定、两者变化。

人工检查范围为固定索引 34、46、50、58 的 24 张人物对照、最差退化点前后三帧的 6 张对照、末帧 6 张训练人物对照及 3 张完整渲染，共 39 张检查图。主要衣物、脸部和肢体轮廓可辨，但手指、物品和袖口的细节仍不符合要求；初始共同状态已经存在部分问题。机位 06 末帧完整图中仍有右侧人物旁的孤立碎片，没有删除或补绘。结合先前独立困难单帧拟合的失败，当前证据不支持把问题简单归为运动累积，或认定开放尺度与透明度即可解决。

已独立核验 100 个模型状态、336 张 RGB 及相应 alpha 输出、84 张参照、672 个人物区域与 288 组相邻图像变化残差。三个连续视频各 25 帧，全部解码通过；12 fps 是展示速度，未经核验的采集帧率不作声明。相邻残差没有光流补偿，不能直接称为闪烁指标；人工看图范围如上，不宣称逐帧看完了视频。

完整图：[机位 00](../outputs/dynamic-human-enerf-support-030/evaluation/camera00-frame058-both.png)、[机位 06](../outputs/dynamic-human-enerf-support-030/evaluation/camera06-frame058-both.png)、[机位 12](../outputs/dynamic-human-enerf-support-030/evaluation/camera12-frame058-both.png)。连续对照：[机位 00](../outputs/dynamic-human-enerf-support-030/comparison/camera00-continuous-comparison.mp4)、[机位 06](../outputs/dynamic-human-enerf-support-030/comparison/camera06-continuous-comparison.mp4)、[机位 12](../outputs/dynamic-human-enerf-support-030/comparison/camera12-continuous-comparison.mp4)。

核对材料：[训练协议](../outputs/dynamic-human-enerf-support-030/protocol.json)、[模型状态](../outputs/dynamic-human-enerf-support-030/independent-state-audit.json)、[留出图像核验](../outputs/dynamic-human-enerf-support-030/independent-image-audit.json)、[训练诊断核验](../outputs/dynamic-human-enerf-support-030/independent-training-image-audit.json)、[配对统计](../outputs/dynamic-human-enerf-support-030/paired-statistics.json)、[实际看图范围](../outputs/dynamic-human-enerf-support-030/visual-review.json)、[阶段结论](../outputs/dynamic-human-enerf-support-030/final-summary.json)。此轮对照结束，移动人物清晰度和完整场景目标仍未完成。

### 行走遮挡片段与背景几何准备

只使用训练机位 08 筛选另两组人物序列，在 actor2_3 与 actor5_6 中查看了六组、共 18 张缩小展示的源图。actor5_6 的索引 750 附近具有较明显行走及相互遮挡，已选为下一组候选；尚未在该片段训练模型或检查留出相机。作者前景合成在移动人物附近保留了墙面纹理残影，因此不能把这些像素全部作为正确人体表面监督。最初预览的整数乘法溢出已经在检查前修正，原错误产物作为本地记录保留，光流筛选使用原始 RGB，不受此预览错误影响。[筛选与观察范围](../outputs/dynamic-human-enerf-performer-selection-031/visual-review.json)。

在 actor5_6 自身的优化相机坐标中，使用 15 个训练机位提供的静态背景参照，经互相匹配、至少三视图轨迹与已知相机三角化得到 **6,480 个背景候选点**。没有使用来源视角未知的作者点云，也没有使用机位 00、06、12 的背景图。独立重投影复算通过，每条轨迹最大误差限制为 1.5 像素；这验证的是候选点与已知相机的几何一致性，不是已经训练好背景，更不是人物与背景联合重建完成。墙面重复纹理仍可能产生错误匹配，地面覆盖也需补足。不能直接把另一组演员的点拼进这个坐标系。[独立几何核验](../outputs/dynamic-human-enerf-background-032/independent-geometry-audit.json)、[实际源图检查](../outputs/dynamic-human-enerf-background-032/visual-review.json)。
'''
old=page.read_text(encoding='utf8')
assert '### 运动阶段的尺度与不透明度对照' not in old
page.write_text(old.rstrip()+section+'\n',encoding='utf8')
sync=Path('D:/Projects/EngineLab/engine-home/scripts/sync-gs-lab.mjs')
old=sync.read_text(encoding='utf8'); assert 'outputs/dynamic-human-enerf-support-030/${name}' not in old
lines=old.splitlines(keepends=True)
index=next(i for i,line in enumerate(lines) if 'outputs/dynamic-human-enerf-local-time-029/${name}' in line)+1
entries="""  ...['protocol.json', 'independent-state-audit.json', 'independent-image-audit.json', 'independent-training-image-audit.json', 'paired-statistics.json', 'visual-review.json', 'final-summary.json', 'comparison/camera00-frame046-person1.png', 'comparison/camera00-frame046-person1-compact.png', 'comparison/regression-camera12-frame044-person1.png', 'training-comparison/camera07-frame058-person1-compact.png', ...['00','06','12'].flatMap(c => [`comparison/camera${c}-continuous-comparison.mp4`, `evaluation/camera${c}-frame058-both.png`])].map(name => [`outputs/dynamic-human-enerf-support-030/${name}`, '同起点的运动尺度与不透明度对照；真实原生渲染仍有手部破碎，未通过清晰度验收']),
  ...['visual-review.json'].map(name => [`outputs/dynamic-human-enerf-performer-selection-031/${name}`, '仅训练机位的行走遮挡候选筛选；尚未训练或验证该人物片段']),
  ...['independent-geometry-audit.json', 'visual-review.json'].map(name => [`outputs/dynamic-human-enerf-background-032/${name}`, '训练机位背景参照的已知相机三角化核验；背景种子不是训练或联合重建结果']),
"""
lines.insert(index,entries);sync.write_text(''.join(lines),encoding='utf8')
print('Evidence section appended; new local resources explicitly allowlisted. Models and raw archives excluded.')
