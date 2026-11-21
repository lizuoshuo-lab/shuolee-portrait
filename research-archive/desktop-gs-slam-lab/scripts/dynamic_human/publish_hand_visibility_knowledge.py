from pathlib import Path
import json,shutil
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-enerf-hand-visibility-100';SITE=Path('D:/Projects/EngineLab/engine-home');read=lambda p:json.loads(p.read_text(encoding='utf8'));result=read(O/'final-summary.json');assert result['visual_review_complete']
p=L/'knowledge/32_dynamic_human_scene_reconstruction.md';s=p.read_text(encoding='utf8');shutil.copy2(p,O/'knowledge-before.md');assert '高斯颜色的光度梯度' not in s
start=s.index('本地对照在每个手部三角面放置一个高斯');end=s.index('判断新表面是否真正参与图像',start)
s=s[:start]+'''本地对照在每个手部三角面放置一个高斯，共1538个，学习颜色、不透明度、局部偏移、尺度与连续关节姿态。为减轻旧表面对新表面的遮挡，两组都把新高斯的不透明度初始化为0.2；一组把1937个几何关联高斯的不透明度降至不高于0.02，另一组对原有6330个手部邻域高斯这样初始化。旧高斯均可重新学习不透明度，几何与颜色保留。两组使用相同的105张照片和6000次更新。

![不透明度初始化对照的实际手部渲染](../outputs/dynamic-human-enerf-hand-visibility-100/native-articulated-hand-comparison.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 上为04机位，下为13机位，均为留出时刻731帧 · 原照片、原模型、1937点初始化、6330点初始化 · 原生像素。

'''+result['knowledge_visual_finding']+'\n\n'+s[end:]
anchor='具体参数、对照结果与贡献检查见[实验附录](../experiments/37_dynamic_human_reconstruction.md#手部先验与关节表面高斯的图像检验)。'
assert anchor in s
replacement=r'''对于共享颜色$\mathbf c_i$，高斯颜色的光度梯度也包含同样的可见权重：

$$
\frac{\partial\mathcal L}{\partial\mathbf c_i}
=\sum_{\mathbf p,t}
T_i(\mathbf p,t)\alpha_i(\mathbf p,t)
\frac{\partial\mathcal L}{\partial\mathbf C(\mathbf p,t)}.
$$

因此，前方旧表面接近不透明时，新表面即使位于合适的投影位置，也可能只能获得很小的颜色梯度。降低旧高斯的不透明度可以改变优化初期的可见性；它不是手部标签，也不能保证真实几何。如果旧表面很快重新变得不透明，新表面仍可能难以承担重建。扩大调整范围还必须检查前臂与玩偶是否受损。[先验与表面表示](../experiments/37_dynamic_human_reconstruction.md#手部先验与关节表面高斯的图像检验)和[不透明度初始化对照](../experiments/37_dynamic_human_reconstruction.md#旧表面遮挡与不透明度初始化)保留参数和实际效果。
'''
s=s.replace(anchor,replacement);p.write_text(s,encoding='utf8')
p=L/'experiments/37_dynamic_human_reconstruction.md';s=p.read_text(encoding='utf8');shutil.copy2(p,O/'evidence-before.md');assert '### 旧表面遮挡与不透明度初始化' not in s
s+='''

### 旧表面遮挡与不透明度初始化

对先前姿态优化模型进行只读分组渲染：1937个几何关联高斯、其余4393个手部邻域高斯、246654个冻结高斯，以及1538个新增表面高斯。04机位731帧、13机位731帧、03机位732帧、04机位732帧的分解显示，仅暂时屏蔽1937个关联点，新增表面的可见权重仍只有其单独渲染权重的6.0%至14.7%；屏蔽全部6330个邻域点也只能达到24.9%至30.8%，且画面出现缺口。这些是同一参数的遮挡归因，并非新重建结果。

进一步检查参考时刻三角面上的欧氏最近点是否落在高斯的三倍标准差协方差椭球内，并结合距离和多视角支持筛选，未增加任何新的候选点。这是保守的最近点启发式，不是椭球与整个三角面的精确相交检验。该负结果被保留，没有把更大的邻域冒充为可靠手部标签。随后做两组可恢复的不透明度初始化：新表面均从0.2开始，分别对1937点或全部6330点把原不透明度上限初始化为0.02；之后6330个旧点的不透明度都可重新学习。两组从同一原模型出发，学习同样的手部表面与连续姿态，使用相同6000次更新、105张原照片及背景顺序；唯一差别是额外4393个旧点的初始不透明度。该邻域可能含前臂和玩偶贡献，需通过实际图像检验而非按名称保证保留。

![不透明度初始化后的留出时刻对照](../outputs/dynamic-human-enerf-hand-visibility-100/native-articulated-hand-comparison.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 04、13机位731帧 · 原照片、原模型、1937点初始化、6330点初始化。

'''+result['evidence_result_paragraph']+'''

沿用134组原生1920×1080输入的评测，包含训练抽查、已知机位奇数时刻、00／06／12留出机位奇数时刻及732帧。另检查04、13机位各13张连续图像、手部和玩偶上下文以及全身图。相机、估计曝光时间与非邻域参数保持固定；此区间仍属于反复使用的开发集，尚不构成独立场景验证。[完整结果](../outputs/dynamic-human-enerf-hand-visibility-100/final-summary.json)、[实际画面检查](../outputs/dynamic-human-enerf-hand-visibility-100/visual-review.json)、[不透明度变化](../outputs/dynamic-human-enerf-hand-visibility-100/opacity-effect.json)、[可见贡献](../outputs/dynamic-human-enerf-hand-visibility-100/contribution/summary.json)。
''';p.write_text(s,encoding='utf8')
root=O.name;files=['final-summary.json','visual-review.json','native-articulated-hand-comparison.png','opacity-effect.json','contribution/summary.json'];p=SITE/'scripts/sync-gs-lab.mjs';s=p.read_text(encoding='utf8');assert root+'/${name}' not in s;lines=s.splitlines(keepends=True);i=next(i for i,l in enumerate(lines) if 'dynamic-human-enerf-mano-surface-099/${name}' in l)+1;lines.insert(i,'  ...'+json.dumps(files)+'.map(name => [`outputs/'+root+'/${name}`, \'手部可见性与不透明度初始化\']),\n');p.write_text(''.join(lines),encoding='utf8');(O/'published-resources.json').write_text(json.dumps(dict(paths={root:files},licensed_models_exported=False),indent=2)+'\n');shutil.copy2(__file__,O/Path(__file__).name);print('Updated bounded hand-surface explanation and actual comparison; five public assets.')
