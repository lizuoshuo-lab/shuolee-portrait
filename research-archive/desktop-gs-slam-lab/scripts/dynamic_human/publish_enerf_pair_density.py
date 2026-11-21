from pathlib import Path
import json,shutil
from PIL import Image
L=Path(__file__).resolve().parents[2];SITE=Path('D:/Projects/EngineLab/engine-home');O=L/'outputs/dynamic-human-enerf-hand-adaptive-094';read=lambda p:json.loads(p.read_text(encoding='utf8'))
for r in ['dynamic-human-enerf-hand-two-view-093','dynamic-human-enerf-hand-adaptive-094']:assert not read(L/'outputs'/r/'final-summary.json')['adopt_as_development_baseline']
im=Image.open(O/'comparison/camera06-frame000732-full-person.png');assert im.width%4==0;im.crop((0,0,im.width//2,im.height)).save(O/'comparison/current-baseline-full-person.png')
p=L/'knowledge/32_dynamic_human_scene_reconstruction.md';s=p.read_text(encoding='utf8');shutil.copy2(p,O/'knowledge-before.md');a=s.index('本地对照固定其他身体部分，分别进行单视角');b=s.index('### 相机与人物必须对应同一个拍摄时刻',a)
s=s[:a]+'''本地对照固定其他身体部分，分别进行单视角、双视角、同一名义帧的多视角及多个时刻的多视角拟合。两个参与拟合的机位可以同时出现清楚的手形，其他方向却仍暴露碎片，说明**两张图都清楚也不足以验证三维表面**。核验对应点时，还要确认它们属于同一个可见表面：相近的曝光时刻不代表相近的观察方向，玩偶上的纹理匹配也不能证明手指几何正确。

![两个拟合机位的清晰手形与其他机位的失真](../outputs/dynamic-human-enerf-hand-two-view-093/comparison/two-view-versus-consistency.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 从上到下为04、17、08机位 · 原照片、原模型、单视角拟合、双视角拟合 · 原生像素。

图中04和17机位参与双视角拟合，08机位没有参与该次拟合。它们沿用各自估计拍摄时刻的运动状态，没有把两张图当作完全同步的几何真值。前两排手形恢复，第三排却出现皮肤色碎片和玩偶变形。相邻机位的原图匹配主要覆盖玩偶纹理，仅有少量手背或前臂候选点，尚不能据此建立可靠的手指表面。

另一项条件独立的实验恢复全部15个训练机位，并让局部高斯按投影位置梯度克隆或分裂。高斯数量和分布实际发生变化，一些轮廓有所改善，但指间粘连与局部破碎仍然存在。**密度控制提供更多表面表示能力，无法单独保证观测对应、局部运动与几何一致性**。这并不证明密度控制无用，而是说明当前条件下它还不足以解决问题。强制统一手部时刻也未通过画面验证，因此现有证据不支持直接清零相机时间偏移。

![当前开发对照的全身轮廓与衣物表现](../outputs/dynamic-human-enerf-hand-adaptive-094/comparison/current-baseline-full-person.png)
> 图源：同上 · 06机位、732帧 · 左为原照片，右为保留的开发对照模型；灰色区域为人物单独渲染的背景，未表示完成大场景背景联合重建。

全身图能看清躯干、衣物和步态轮廓，局部原生窗口则显示尚未恢复的手指。这两个尺度应同时展示：既不能只用全身缩略图掩盖局部模糊，也不能把单个拟合窗口变清楚当作人物整体、跨视角和连续运动都已通过。下一步围绕手部表面与局部连续运动的耦合继续验证，保留跨视角与跨时刻检查。[对照图、实验条件与验收结果](../experiments/37_dynamic_human_reconstruction.md#双视角拟合与局部密度控制)集中保留在实验记录中。

'''+s[b:];p.write_text(s,encoding='utf8')
p=L/'experiments/37_dynamic_human_reconstruction.md';s=p.read_text(encoding='utf8');shutil.copy2(p,O/'evidence-before.md');old='后续先检查近同步训练机位04／17的可见手部对应与射线交会，再决定局部表面的初始化及几何约束。';assert old in s;s=s.replace(old,'近同步机位对应与双视角、局部密度对照现已完成，结果见下一节。');assert '### 双视角拟合与局部密度控制' not in s
s+='''

### 双视角拟合与局部密度控制

**几何对应的支持范围。**04／17机位在沿用的时间模型中只差0.00891帧，但相机中心相距约4.063场景长度单位，原图显示不同的手掌表面。原生分辨率SIFT、双向比率0.7匹配没有得到手部窗口对应点，不能据此初始化手指。另检查8组邻近训练机位，窗口内虽有通过极线、正深度、重投影和交会角筛选的点，实际查看的样本大多是玩偶纹理，只有少量手背／前臂候选。匹配筛选先于极线门限；9组基本矩阵与重投影已用另一种矩阵推导复核，但这不等于语义身份或指尖真值得到确认。[几何检查与限制](../outputs/dynamic-human-enerf-hand-pair-geometry-092/final-summary.json)。

**双视角拟合恢复两张图，却未恢复其他方向。**从同一开发对照出发，仅使用04／17机位732帧，更新6330个既有局部高斯3000步，运动、其他246654个高斯及全部绑定冻结。初始模型、损失、学习率、背景采样和预算与前面的单视角对照一致，只改变机位采样。

![双视角恢复与跨视角失败的原生图像对照](../outputs/dynamic-human-enerf-hand-two-view-093/comparison/two-view-versus-consistency.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 04、17、08机位 · 原照片、原模型、单视角拟合、双视角拟合。

两个拟合机位的手部窗口平均PSNR从21.460提升到23.781dB，手形与手指轮廓明显恢复；其他13个已知机位却从20.918降到17.443dB，出现拉丝、碎片和玩偶变形。三个原留出机位的全前景均值从22.804降到21.788dB。此处窗口包含手、衣物和玩偶，分数并非指尖精度。18机位108张颜色／透明度PNG与指标已复核，全部15个手部窗口及3个留出机位全身图已查看。分支不采用，不宣称跨时间验证完成。[双视角结论](../outputs/dynamic-human-enerf-hand-two-view-093/final-summary.json)、[看图记录](../outputs/dynamic-human-enerf-hand-two-view-093/render-visual-review.json)。

**局部密度控制有实际更新，但清晰度仍不足。**恢复732帧15机位，与固定密度对照使用完全一致的3000步相机和背景采样、学习率、初始参数及损失。400至2000步每200步，对局部高斯按平均归一化投影位置梯度执行克隆／分裂，并剔除低不透明度点。梯度阈值为0.0002，大小分界为场景尺度的0.01倍，分裂尺度除以1.6，局部不透明度剔除阈值0.005，无不透明度重置。这是采用gsplat操作的局部研究改动，不是完整官方3DGS复现。

![局部自适应密度与固定密度的手部对照](../outputs/dynamic-human-enerf-hand-adaptive-094/comparison/adaptive-density-hand.png)
> 图源：同上 · 04、17机位 · 原照片、原模型、固定密度15视角拟合、自适应密度15视角拟合。

实际发生2006次克隆、130次分裂、1384次剔除，局部高斯由6330变为7082；其余246654个高斯完全不变，子高斯继承原绑定，全部384个运动节点冻结。相对固定密度对照，04机位手部窗口提高0.068dB，其他14个训练机位均值提高0.169dB，三个留出机位全前景均值则由22.813变为22.811dB。轮廓有小幅改善，但手指仍不完整。18机位108张PNG与指标复核、全部15个手部窗口及3个留出全身图查看后，该分支仍未通过清晰度验收，不替换开发对照。[密度实验结论](../outputs/dynamic-human-enerf-hand-adaptive-094/final-summary.json)、[密度与冻结边界核验](../outputs/dynamic-human-enerf-hand-adaptive-094/density-audit.json)、[看图记录](../outputs/dynamic-human-enerf-hand-adaptive-094/render-visual-review.json)。

当前保留原连续模型作为开发对照。下一步检查局部表面与连续运动的耦合约束；这些同一场景开发实验没有完成独立新场景验收，也没有完成静态背景联合重建。不会通过增加采样机位数、训练步数或高斯数量来代替真实画面验收。
''';p.write_text(s,encoding='utf8')
paths={'dynamic-human-enerf-hand-pair-geometry-092':['final-summary.json'],'dynamic-human-enerf-hand-two-view-093':['final-summary.json','render-visual-review.json','comparison/two-view-versus-consistency.png'],'dynamic-human-enerf-hand-adaptive-094':['final-summary.json','density-audit.json','render-visual-review.json','comparison/adaptive-density-hand.png','comparison/current-baseline-full-person.png']}
p=SITE/'scripts/sync-gs-lab.mjs';s=p.read_text(encoding='utf8');lines=s.splitlines(keepends=True);i=next(i for i,l in enumerate(lines) if 'dynamic-human-enerf-hand-snapshot-091/${name}' in l)+1
for root,files in reversed(list(paths.items())):
 assert root+'/${name}' not in s;lines.insert(i,'  ...'+json.dumps(files)+'.map(name => [`outputs/'+root+'/${name}`, \'双视角与局部密度对照\']),\n')
p.write_text(''.join(lines),encoding='utf8');(O/'published-resources.json').write_text(json.dumps(dict(paths=paths,licensed_models_exported=False),indent=2)+'\n',encoding='utf8');shutil.copy2(__file__,O/Path(__file__).name);print('Theory updated;9assets explicitly allowlisted.')
