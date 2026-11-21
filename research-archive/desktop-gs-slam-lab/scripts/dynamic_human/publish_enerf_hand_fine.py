"""Keep theory concise; link matched fine-motion evidence separately."""
from pathlib import Path
import json,shutil
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-enerf-hand-fine-085';SITE=Path('D:/Projects/EngineLab/engine-home')
read=lambda p:json.loads(p.read_text(encoding='utf8'))
assert read(O/'final-summary.json')['actual_render_panels_reviewed']==40
p=L/'knowledge/32_dynamic_human_scene_reconstruction.md';txt=p.read_text(encoding='utf8');assert '局部节点方案尚未训练' in txt;shutil.copy2(p,O/'knowledge-before.md');a=txt.index('![张手时的原照片、原模型、局部联合优化及增加运动约束后的原生对照]');b=txt.index('### 相机与人物必须对应同一个拍摄时刻',a)
txt=txt[:a]+r'''![张手时原照片与增加局部运动控制后的原生重建](../outputs/dynamic-human-enerf-hand-fine-085/comparison/camera04-frame000731-target_right_hand.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 机位04、留出帧731 · 原照片、原模型、细节点与图像联合优化、再加入运动约束。

原照片中的手指已经与玩偶分离，三种重建仍出现皮肤与玩偶边缘混合的碎片。此次同时优化局部高斯的几何、颜色、不透明度和运动，并增加32个局部控制节点。加入运动项后，训练观测的法向残差从约1.24降到0.67像素，留出图像中的指间分离仍未恢复。因此，运动约束拟合更好，不能直接解释为表面重建更准确。

还要区分**表面表示的密度**与**运动控制的空间分辨率**。前文[变形图](#骨架之外的局部变形如何进入渲染)中，高斯运动由邻近节点及权重共同决定。增加节点，可以使邻近区域获得更多不同运动；它并不提供新的皮肤表面，也不保证手和玩偶已正确分离。本地新增节点沿用原模型的候选材料位置，一些投影仍落在衣服或玩偶边缘，不能把这些节点当成手部关节。

新增变形已经实际生效，但不足以恢复手指。相同训练条件下，更细节点相对原有节点仅带来很小的图像变化；两种候选均未替换原模型。接下来的检查需同时看高斯对手部像素的**实际可见贡献**：中心不在手部掩码内的高斯，其投影覆盖范围仍可能跨入手部，仅按中心筛选更新对象可能遗漏这些贡献。这个待检验的问题不能直接当作模糊成因。原生全身图、固定窗口连续视频与对照条件见[局部运动节点与手部表面](../experiments/37_dynamic_human_reconstruction.md#局部运动节点与手部表面)；手指清晰度与完整场景重建仍未完成。

'''+txt[b:];p.write_text(txt,encoding='utf8')
p=L/'experiments/37_dynamic_human_reconstruction.md';txt=p.read_text(encoding='utf8');assert '### 局部运动节点与手部表面' not in txt;shutil.copy2(p,O/'evidence-before.md')
txt+=r'''

### 局部运动节点与手部表面

检验的问题是：在上一轮局部几何、颜色与运动联合优化之上，增加局部运动自由度，能否让张开的手指从玩偶中分离。32个新增节点取自原模型选中的6330个高斯中心，随原身体与粗层变形移动，再施加局部残差变形；每个高斯连接4个邻近细节点，节点间建立256条有向邻接边。既有22个节点与6330个高斯仍可更新，其余246654个高斯和362个节点保持冻结。没有新增高斯，也没有新增指尖真值。

新增变形的零初始化与原模型渲染精确一致。两支分别仅用图像、再加入法向运动约束，各训练3000步；105张训练照片、抽样、背景、学习率与上一轮相同。此次也加入细层的刚性和变形幅度正则化，因此对比的是完整的细层方案，不能把差异单独归因于节点数量。投影检查已查看13张原生照片拼图：部分节点仍落在衣服或玩偶边缘，它们是原模型中的候选材料位置，不是准确的皮肤节点。

![细节点方案的完整人物对照](../outputs/dynamic-human-enerf-hand-fine-085/comparison/camera04-frame000731-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 原照片、原模型、细节点图像分支、细节点运动约束分支。

120个机位时刻均按1920×1080渲染，360张预测及720张颜色／透明度PNG独立复核。90张留出时间图的前景PSNR分别为24.637、24.637、24.621dB；另外三个机位的18张图为22.699、22.690、22.679dB。相对上一轮同预算粗节点方案，细层图像分支在两组中的平均变化仅为+0.0023、+0.0021dB，尚不足以说明视觉清晰度改善。

![手指分离仍未恢复](../outputs/dynamic-human-enerf-hand-fine-085/comparison/camera04-frame000731-target_right_hand.png)
> 图源：同上 · 原生160×160窗口，无锐化或补画。

6个固定目标手窗口相对原模型平均变化为+0.019、+0.006dB，但12个手与物品窗口为−0.013、−0.052dB；12个脚部窗口完全不变。1527个估计对应的法向误差从1.236降到0.673像素，也没有恢复手指轮廓。已实际查看40张对照图，包含所有两组留出数据的前三差案例、12个手物窗口、6个目标手窗口及覆盖两个机位全部13连续帧的拼图。**两支候选均不替换原模型。**

![抓握到张手过程中的实际渲染](../outputs/dynamic-human-enerf-hand-fine-085/temporal/camera04-comparison.png)
> 图源：同上 · 721、727、731帧 · 每列顺序同前图。

连续检查使用固定窗口，78张预测PNG与数组逐一一致。04、13机位的相邻帧图像差分误差，在图像分支中仅有小幅下降，运动分支则升高；张手后的皮肤与玩偶碎片仍然相互混杂。[04机位连续视频](../outputs/dynamic-human-enerf-hand-fine-085/temporal/camera04-hand-continuity.mp4)、[13机位连续视频](../outputs/dynamic-human-enerf-hand-fine-085/temporal/camera13-hand-continuity.mp4)为13帧、5帧／秒的检查副本，播放速度不是采集帧率。

另将训练完成的细层临时旁路，保留其余参数不变，在105个原训练机位时刻测量细层的实际作用：图像分支与运动分支的选中高斯中心投影位移中位数分别为0.165、0.412像素，95%分位数为0.686、6.025像素。该统计等权计入遮挡和偏离表面的中心，并非表面误差；它只证明细层已经影响几何，不能证明几何正确。继续加节点或加大运动权重，尚无清晰度证据支持。

下一步转向可见贡献检查：原筛选使用中心投影与估计深度，而高斯覆盖的像素范围比中心更大。需先确定被冻结的高斯是否仍大量贡献错误手部像素，再决定是否调整局部优化范围。所有结论仍属于同一场景开发，完整背景与独立新场景验证未完成。

记录：[本轮结论](../outputs/dynamic-human-enerf-hand-fine-085/final-summary.json)、[配对评测](../outputs/dynamic-human-enerf-hand-fine-085/comparison/summary.json)、[同预算粗细节点比较](../outputs/dynamic-human-enerf-hand-fine-085/coarse-versus-fine.json)、[细节窗口](../outputs/dynamic-human-enerf-hand-fine-085/comparison/detail-metrics.json)、[连续画面](../outputs/dynamic-human-enerf-hand-fine-085/temporal/summary.json)、[细层实际作用](../outputs/dynamic-human-enerf-hand-fine-085/fine-effect/summary.json)、[数值核验](../outputs/dynamic-human-enerf-hand-fine-085/numerical-audit.json)、[实际看图记录](../outputs/dynamic-human-enerf-hand-fine-085/render-visual-review.json)。
''';p.write_text(txt,encoding='utf8')
files=['final-summary.json','comparison/summary.json','comparison/detail-metrics.json','coarse-versus-fine.json','temporal/summary.json','fine-effect/summary.json','numerical-audit.json','render-visual-review.json','comparison/camera04-frame000731-full-person.png','comparison/camera04-frame000731-target_right_hand.png','temporal/camera04-comparison.png','temporal/camera04-hand-continuity.mp4','temporal/camera13-hand-continuity.mp4']
p=SITE/'scripts/sync-gs-lab.mjs';txt=p.read_text(encoding='utf8');assert O.name+'/$'+'{name}' not in txt;lines=txt.splitlines(keepends=True);i=next(i for i,l in enumerate(lines) if 'dynamic-human-enerf-hand-joint-084/$'+'{name}' in l)+1;lines.insert(i,'  ...'+json.dumps(files)+".map(name => [\x60outputs/"+O.name+"/$"+"{name}\x60, '局部运动节点与手部表面']),\n");p.write_text(''.join(lines),encoding='utf8');(O/'published-resources.json').write_text(json.dumps(dict(paths={O.name:files},licensed_models_exported=False),indent=2)+'\n',encoding='utf8');shutil.copy2(__file__,O/Path(__file__).name);print('085 theory replaced, evidence added,13resources explicitly allowlisted')
