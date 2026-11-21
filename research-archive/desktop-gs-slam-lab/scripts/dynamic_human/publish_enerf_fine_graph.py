"""Explain graph resolution with actual paired evidence; preserve old sections."""
from pathlib import Path
import json,shutil
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-fine-graph-071';S=Path('D:/Projects/EngineLab/engine-home')
read=lambda p:json.loads(p.read_text(encoding='utf8'))
f=read(R/'final-summary.json');v=read(R/'visual-review.json');assert not f['adopt_as_development_baseline'] and len(v['reviewed_files'])==64
block=r'''
### 运动控制节点的密度与观测支持

![同一画面的原照片、384节点基线与1536节点运动图重建](../outputs/dynamic-human-enerf-fine-graph-071/paired-comparison/camera04-frame000751-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实际训练 · 机位04、帧751 · 原生像素，无增强。

增加变形图节点，可以让相邻小块表面拥有更独立的运动。但控制节点不等于高斯：前者决定局部运动场的自由度，后者决定可渲染几何和颜色的采样。若每个高斯仍由四个邻近节点控制，增加节点会缩短空间支持范围，也会改变上式局部刚性约束中的邻接距离。因此，相同的正则项系数并不代表相同的空间平滑作用。

本地同预算对照将节点从384增至1536，仍使用原生照片、固定身体运动和相同15000次更新。部分衣服图案与前臂边缘略有改善，但手指、抬起的鞋和细衣纹仍偏软。已知机位留出时间的平均前景PSNR提高0.160dB，另外三个机位平均下降0.022dB；更多运动自由度尚未转化为稳定的新视角细节。

![遮挡机位中仍存在的胸前条纹](../outputs/dynamic-human-enerf-fine-graph-071/heldout/paired-comparison/camera06-frame000745-full-person.png)
> 图源：同一公开数据与本站训练 · 机位06、帧745 · 原照片、384节点基线、1536节点对照。

遮挡处的浅色条纹并未因增加节点而消失。运动表示只能调整已有监督下的几何与外观，不能自动补回被掩码删除的观测，也不能把遮挡人物的颜色变成后方表面的颜色。节点密度、观测支持和表面可见性需要分别验证。此次保留384节点模型作为开发基线，[完整条件与负例](../experiments/37_dynamic_human_reconstruction.md#运动控制节点密度的同预算对照)保存在实验附录。

'''
p=L/'knowledge/32_dynamic_human_scene_reconstruction.md';text=p.read_text(encoding='utf8')
if '### 运动控制节点的密度与观测支持' not in text:text=text.replace('### 留出视角中的时间对应与外观验证',block+'### 留出视角中的时间对应与外观验证')
p.write_text(text,encoding='utf8')
evidence=f'''

### 运动控制节点密度的同预算对照

071由067基线重新初始化，只改变运动图的采样密度及相应邻接、绑定距离：每个人物192→768节点，总节点384→1536，运动参数6912→27648。固定105张原生偶数帧、52109个初始高斯、身体运动、时间偏移、随机背景、图像抽样和15000次更新；四节点权重与八邻接数不变，正则项系数不变，但实际空间平滑范围随邻接距离改变。没有采用070的人物归属监督。相同更新预算不是相同计算量。

实际完成15000次更新，最终218579个高斯，训练600.42秒，峰值分配显存1.663GB。所有120张原生渲染完成像素复算，90张已知机位留出时间图平均前景PSNR为26.2857→26.4459dB，77张改善；18张新机位图为23.7577→23.7353dB，7张改善。三个机位00、06、12的平均变化分别为−0.0115、−0.1198、+0.0640dB；最差新机位是06第745帧，变化−0.2220dB。固定手臂区域可能包含遮挡者，不能把区域分数称为隐藏手臂精度。

![固定机位的原生完整对照](../outputs/dynamic-human-enerf-fine-graph-071/paired-comparison/camera04-frame000751-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站训练 · 原照片、067、071。

![新机位中变化最差的完整画面](../outputs/dynamic-human-enerf-fine-graph-071/heldout/paired-comparison/camera06-frame000745-full-person.png)
> 图源：同上 · 机位06、帧745。

{v['narrative_zh']} 已查看64个唯一完整或局部对照图，包含预定区域及两个集合各自最差画面。已知机位75组时间差分残差均值0.032258→0.031527，新机位15组为0.039789→0.039708；7段六帧视频机械解码成功，不宣称完整播放审阅。CPU用SciPy复算全部1536节点、195个身体上下文的旋转与固定512高斯样本的线性混合，最小行列式0.9059，最小奇异值0.9522。该检查只排除所查样本的数值退化，不证明几何正确。

结论：{v['decision_zh']}。单随机种子、同一人物对、重复使用的开发机位与时段，不能推广为普遍方法排名。下一项从067基线隔离原生训练matte中的局部漏标，保持模型容量不变。

报告：[预先固定协议](../outputs/dynamic-human-enerf-fine-graph-071/ablation-protocol.json)、[实际训练条件](../outputs/dynamic-human-enerf-fine-graph-071/training-protocol-audit.json)、[CPU变形检查](../outputs/dynamic-human-enerf-fine-graph-071/trained-graph-independent-audit.json)、[已知机位](../outputs/dynamic-human-enerf-fine-graph-071/paired-comparison/summary.json)、[另外机位](../outputs/dynamic-human-enerf-fine-graph-071/heldout/paired-comparison/summary.json)、[时间变化](../outputs/dynamic-human-enerf-fine-graph-071/temporal-comparison/summary.json)、[另外机位时间变化](../outputs/dynamic-human-enerf-fine-graph-071/heldout/temporal-comparison/summary.json)、[实际看图范围](../outputs/dynamic-human-enerf-fine-graph-071/visual-review.json)、[结论](../outputs/dynamic-human-enerf-fine-graph-071/final-summary.json)。
'''
p=L/'experiments/37_dynamic_human_reconstruction.md';text=p.read_text(encoding='utf8')
if '### 运动控制节点密度的同预算对照' not in text:p.write_text(text+evidence,encoding='utf8')
assets=['ablation-protocol.json','training-protocol-audit.json','trained-graph-independent-audit.json','paired-comparison/summary.json','heldout/paired-comparison/summary.json','temporal-comparison/summary.json','heldout/temporal-comparison/summary.json','visual-review.json','final-summary.json','paired-comparison/camera04-frame000751-full-person.png','heldout/paired-comparison/camera06-frame000745-full-person.png']
assert all((R/x).is_file() for x in assets)
p=S/'scripts/sync-gs-lab.mjs';text=p.read_text(encoding='utf8')
if R.name+'/${name}' not in text:
 lines=text.splitlines(keepends=True);i=next(i for i,line in enumerate(lines) if 'dynamic-human-enerf-actor-supervision-070/${name}' in line)+1
 lines.insert(i,'  ...'+json.dumps(assets)+f".map(name => [`outputs/{R.name}/${{name}}`, '运动节点密度对照']),\n");p.write_text(''.join(lines),encoding='utf8')
shutil.copy2(__file__,R/Path(__file__).name);print('071knowledge/evidence/allowlist prepared. No site build or manifest overwrite.')
