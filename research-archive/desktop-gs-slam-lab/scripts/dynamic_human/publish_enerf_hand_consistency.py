from pathlib import Path
import json,shutil
L=Path(__file__).resolve().parents[2];SITE=Path('D:/Projects/EngineLab/engine-home');O=L/'outputs/dynamic-human-enerf-hand-snapshot-091';read=lambda p:json.loads(p.read_text(encoding='utf8'))
for root in ['dynamic-human-enerf-hand-time-scope-088','dynamic-human-enerf-hand-view-scope-089','dynamic-human-enerf-hand-snapshot-091']:assert read(L/'outputs'/root/'final-summary.json')['adopt_as_development_baseline'] is False
p=L/'knowledge/32_dynamic_human_scene_reconstruction.md';txt=p.read_text(encoding='utf8');shutil.copy2(p,O/'knowledge-before.md');a=txt.index('新增变形已经实际生效，但不足以恢复手指。');b=txt.index('### 相机与人物必须对应同一个拍摄时刻',a)
txt=txt[:a]+r'''新增变形已经实际生效，但不足以恢复手指。还需看高斯对手部像素的**实际可见贡献**：中心不在手部掩码内的高斯，其投影覆盖范围仍可能跨入手部。当前检查中，可更新高斯平均承担估计手部区域约81.5%的贡献；手部区域对这些参数的图像梯度也没有被其余像素淹没。因此，简单扩大更新范围或提高局部损失权重，并没有得到当前证据支持。估计掩码会漏掉手指，也不能充当表面真值。原生全身图与连续对照见[局部运动节点与手部表面](../experiments/37_dynamic_human_reconstruction.md#局部运动节点与手部表面)。

#### 为什么单视角拟合不等于三维重建

![同一手部在拟合视角与其他视角中的实际重建](../outputs/dynamic-human-enerf-hand-view-scope-089/comparison/single-view-versus-consistency.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 原生窗口 · 原照片、原模型、15视角拟合、单视角拟合。

上排是参与单视角拟合的04机位，下排是没有参与该次拟合的13机位。只用04机位的一张照片优化后，手掌和手指轮廓明显恢复；同一模型从13机位看，却出现拉长的皮肤色碎片。这些都是模型的实际渲染，未经过锐化或补画。**图像变清楚只能说明该视角的拟合能力，不能证明三维表面正确。**

对于相机中心 $\mathbf o_c$ 和像素 $\mathbf p$，针孔模型对应一条空间射线：

$$
\mathbf X(\lambda)=\mathbf o_c+\lambda\mathbf d_c(\mathbf p),\qquad
\lambda>0,\qquad
\pi_c\!\left(\mathbf X(\lambda)\right)=\mathbf p.
$$

$\mathbf d_c(\mathbf p)$ 为世界坐标系下的单位射线方向，$\lambda$ 是沿射线的距离，与场景坐标使用相同长度单位；$\pi_c$ 为相机投影。同一条射线上的许多三维位置具有相同投影，单张照片不能唯一确定深度。高斯还可以联合调整位置、协方差、不透明度与视角相关颜色，使一张图像看起来正确，却在其他方向暴露错误形状。

多视角训练要求同一组材料参数同时解释不同相机的观测；动态场景还要使用各张照片真实拍摄时刻对应的形变。这里的两个条件必须分别核验：**共享的局部表面是否合理，以及这个表面在各次曝光时刻的位置是否正确**。单个名义帧编号并不保证各相机完全同时曝光。

本地对照固定其他身体部分，分别进行单视角、同一名义帧的多视角、多个时刻的多视角拟合。单视角能够恢复局部轮廓，但跨视角失真；单时刻拟合的局部改善也不能稳定传递到其他时刻。进一步强制所有相机使用同一个手部世界坐标状态，并未恢复一致的手指。这些结果没有给出唯一成因，但排除了把“只让一张图更清楚”当作动态重建成功的做法；也不足以支持直接清零时间偏移。下一步需通过可见性明确的多视角对应检查局部几何，再决定表面初始化和约束方式。[对照图、实验条件与验收结果](../experiments/37_dynamic_human_reconstruction.md#局部拟合-时间共享与多视角一致性)集中保留在实验记录中。

'''+txt[b:];p.write_text(txt,encoding='utf8')
p=L/'experiments/37_dynamic_human_reconstruction.md';txt=p.read_text(encoding='utf8');shutil.copy2(p,O/'evidence-before.md')
old='此项没有训练新模型。下一步检查局部手部图像与其余画面对共享高斯参数的梯度是否稀释或冲突，再决定局部表面优化方式。'
new='此项没有训练新模型。后续梯度分解和局部拟合已完成，见下节；结果不支持手部梯度被其余画面淹没的解释。'
assert old in txt;txt=txt.replace(old,new);assert '### 局部拟合、时间共享与多视角一致性' not in txt
txt+=r'''

### 局部拟合、时间共享与多视角一致性

这组对照检验三件事：手部图像是否真正驱动局部参数，现有高斯是否能拟合清晰手指，以及局部改善能否同时满足其他视角和其他时刻。输入始终采用原生1920×1080照片、原相机和原前景掩码；局部窗口包含手、玩偶和衣物，不是准确的人体部位真值。模型是本地研究扩展，以下结果不代表论文官方成绩。

**梯度没有被其余画面淹没。**对原模型和细节点图像分支，各检查105张训练照片，按同一全图归一化系数拆分手部窗口与其余区域；SSIM完整邻域按中心归属，正则项另计。窗口平均只占人物裁图4.61%，但选中高斯位置的聚合梯度范数约为其余区域的198倍和131倍，六类高斯参数的局部梯度与总梯度方向均高度一致。这里比较的是同一参数族的梯度，不混加不同单位，也不等同于Adam实际更新。此项只做诊断，[完整分解与限制](../outputs/dynamic-human-enerf-hand-gradient-087/final-summary.json)保留备查。

**固定运动，单时刻与多时刻表面拟合仍不能通过。**两支均从同一模型出发，只更新6330个候选高斯的位置、尺度、旋转、不透明度与球谐系数；其余246654个高斯、全部384个运动节点及绑定保持不变。两支各3000步，机位抽样、背景和学习率一致；一支只见732帧的15个训练机位，另一支见720至732之间7个偶数帧。没有增加高斯或修改掩码。相同总步数意味着每张照片被采样次数不同，因此这是拟合范围的诊断，不能把全部差异归结为时间冲突。

![单时刻拟合与多时刻拟合在留出时间中的实际表现](../outputs/dynamic-human-enerf-hand-time-scope-088/temporal/camera04-comparison.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 721、727、731帧 · 原照片、原模型、只拟合732帧、拟合7个时刻。

单时刻分支在732帧的15个手部窗口平均提高0.515dB，但手指仍有碎片；90张其他时刻图像的前景PSNR下降0.392dB，另外三个机位的18张图下降0.232dB。多时刻分支在两组中分别下降0.011和0.017dB，手指也未稳定恢复。134个机位时刻、402张预测的颜色和透明度PNG，以及两个机位各13连续帧，共882次PNG数组核验已完成；实际查看33张拼图或单图，覆盖全部15个同帧手部窗口和两段完整连续序列。两支均不采用。[单／多时刻结论](../outputs/dynamic-human-enerf-hand-time-scope-088/final-summary.json)、[数值核验](../outputs/dynamic-human-enerf-hand-time-scope-088/numerical-audit.json)、[看图记录](../outputs/dynamic-human-enerf-hand-time-scope-088/render-visual-review.json)、[04机位连续视频](../outputs/dynamic-human-enerf-hand-time-scope-088/temporal/camera04-hand-continuity.mp4)、[13机位连续视频](../outputs/dynamic-human-enerf-hand-time-scope-088/temporal/camera13-hand-continuity.mp4)保留在此；视频为5帧／秒慢速检查副本。

**单视角能拟合清楚，其他视角却失真。**再用相同初始模型、6330个高斯、损失、学习率、背景和3000步预算，只拟合04机位的732帧。该机位手部窗口PSNR由22.818提升至25.164dB，手掌与手指轮廓明显恢复；其他14个已知机位的手部窗口却由20.860降至17.523dB，三个原留出机位的全前景均值由22.804降至22.171dB。

![拟合视角变清楚，但其他视角出现几何碎片](../outputs/dynamic-human-enerf-hand-view-scope-089/comparison/single-view-versus-consistency.png)
> 图源：同上 · 上排04、下排13机位 · 原照片、原模型、15视角拟合、单视角拟合。

18个机位的54张预测、108张颜色／透明度PNG独立复核；15个手部窗口及04、00、06、12机位全身图实际查看。该分支只能说明单图拟合能力，不能作为三维或连续模型交付。[单视角诊断](../outputs/dynamic-human-enerf-hand-view-scope-089/final-summary.json)、[完整18视角评测](../outputs/dynamic-human-enerf-hand-view-scope-089/comparison/summary.json)、[实际看图记录](../outputs/dynamic-human-enerf-hand-view-scope-089/render-visual-review.json)。

**拍摄时刻有影响，但统一时刻不是已验证的修复。**沿用原估计时间偏移，18个机位的名义732帧实际在模型中对应730.604至733.767的有效帧索引。以04机位的状态为参照，02机位选中高斯的投影位移中位数约24.58像素，17机位则仅约0.098像素；这些是模型预测的几何变化，绝不是观测误差。17机位在单视角拟合后同样明显退化，因此不能把问题全部归因于时间差。[18机位投影与15个可见区域的独立核验](../outputs/dynamic-human-enerf-hand-capture-geometry-090/final-summary.json)。

进一步保持15个训练机位、732帧、同一初始参数和3000步抽样不变，只让这6330个高斯在所有相机中采用04机位的同一个世界坐标状态、协方差与颜色参考方向，其余身体保持原渲染。参考机位的初始RGB／透明度精确复现，其他高斯几何精确不变；候选手部的初始世界几何则因干预而改变。结果中，其他14个训练机位手部窗口相对原捕获时刻拟合由21.383降至20.901dB，三个留出机位全前景由22.813降至22.734dB，仍有指间混合与碎片。所有18视角和108张PNG已复核，12张实际拼图／单图已查看。**此方案也不采用，原标定和时间偏移保持原值。**[统一手部状态对照](../outputs/dynamic-human-enerf-hand-snapshot-091/final-summary.json)、[实际看图记录](../outputs/dynamic-human-enerf-hand-snapshot-091/render-visual-review.json)、[04机位原生窗口](../outputs/dynamic-human-enerf-hand-snapshot-091/comparison/camera04-frame000732-hand.png)。

当前结论是：单个视角有清晰拟合能力，但正确的多视角手部表面尚未建立；单纯加运动自由度、扩大损失权重或强制统一时刻，都没有通过现有画面验证。后续先检查近同步训练机位04／17的可见手部对应与射线交会，再决定局部表面的初始化及几何约束。既有估计指尖和掩码仍不作为真值。上述均是同一场景的开发对照；独立新场景、完整背景联合重建和移动人体清晰度验收仍未完成。
''';p.write_text(txt,encoding='utf8')
paths={
'dynamic-human-enerf-hand-gradient-087':['final-summary.json'],
'dynamic-human-enerf-hand-time-scope-088':['final-summary.json','numerical-audit.json','render-visual-review.json','temporal/camera04-comparison.png','temporal/camera04-hand-continuity.mp4','temporal/camera13-hand-continuity.mp4'],
'dynamic-human-enerf-hand-view-scope-089':['final-summary.json','comparison/summary.json','render-visual-review.json','comparison/single-view-versus-consistency.png'],
'dynamic-human-enerf-hand-capture-geometry-090':['final-summary.json'],
'dynamic-human-enerf-hand-snapshot-091':['final-summary.json','render-visual-review.json','comparison/camera04-frame000732-hand.png']}
p=SITE/'scripts/sync-gs-lab.mjs';txt=p.read_text(encoding='utf8');lines=txt.splitlines(keepends=True);i=next(i for i,l in enumerate(lines) if 'dynamic-human-enerf-hand-contributions-086/${name}' in l)+1
for root,files in reversed(list(paths.items())):
 assert root+'/${name}' not in txt;lines.insert(i,'  ...'+json.dumps(files)+'.map(name => [`outputs/'+root+'/${name}`, \'局部拟合与多视角一致性\']),\n')
p.write_text(''.join(lines),encoding='utf8');(O/'published-resources.json').write_text(json.dumps(dict(paths=paths,licensed_models_exported=False),indent=2)+'\n',encoding='utf8');shutil.copy2(__file__,O/Path(__file__).name);print('Knowledge updated with theory and native comparison;15assets allowlisted.')
