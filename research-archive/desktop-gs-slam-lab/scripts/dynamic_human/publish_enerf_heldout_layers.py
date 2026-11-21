from pathlib import Path
from datetime import datetime,timezone
import json,shutil
L=Path(__file__).resolve().parents[2];S=Path('D:/Projects/EngineLab/engine-home')
H=L/'outputs/dynamic-human-enerf-heldout-cameras-068';D=L/'outputs/dynamic-human-enerf-actor-layers-069'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,x:p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
h=read(H/'final-summary.json');d=read(D/'final-summary.json')
assert read(H/'independent-image-audit.json')['images']==54
assert len(read(D/'independent-audit.json')['records'])==9
assert not h['goal_complete'] and not d['goal_complete']

addition=r'''
### 留出视角中的时间对应与外观验证

![未参与本轮外观训练的机位：原照片、头颈运动基线与局部变形](../outputs/dynamic-human-enerf-heldout-cameras-068/paired-comparison/camera12-frame000751-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实际渲染 · 机位12、帧751。

留出相机用于检查学到的三维表示能否从其他方向渲染。动态场景还需要明确时间：不同相机的相同帧编号未必对应完全相同的人体姿态。设相机 $k$ 的连续时间索引为

$$
t_k=n+\delta_k,
$$

其中 $n$ 是图像文件的整数帧索引，$\delta_k$ 是相对参考相机的时间偏移，以帧为单位。固定已学到的关节轨迹 $\mathbf X_j(t)$ 后，可以只用校准帧的二维关节预测 $\mathbf u_{k,n,j}$ 估计偏移：

$$
\widehat\delta_k=\underset{|\delta_k|\leq 2}{\arg\min}
\sum_{n\in\mathcal C}\sum_{j\in\mathcal J}\sum_{d\in\{u,v\}}
\rho_s\!\left(\sqrt{q_{k,n,j}}\,[\pi_k(\mathbf X_j(n+\delta_k))-\mathbf u_{k,n,j}]_d\right)
+\rho_s(\beta\delta_k).
$$

$\pi_k$ 是使用给定相机内外参的透视投影，$q_{k,n,j}$ 为截断到1的预测置信度，$d$ 分别取两个图像坐标分量。残差以像素计，稳健损失为 $\rho_s(r)=s^2(\sqrt{1+(r/s)^2}-1)$，本地取 $s=3$ 像素、$\beta=0.1$ 像素/帧。本地设置只使用身体与足部预测，并在拟合前按置信度和与固定轨迹的相容性筛选。随后冻结偏移与所有模型参数，在不属于 $\mathcal C$ 的照片上检查外观。原照片不进行时间插值，只有三维运动在连续时刻求值。

实际对照使用3个未参与本轮高斯训练、初始化和姿态拟合的机位：21张偶数帧只校准时间，18张奇数帧评价外观。局部变形模型的平均前景PSNR为23.76dB，较头颈基线提高0.87dB；图中玩偶斑纹和轮廓更清楚，但手指和鞋部仍偏软。这些相机在更早的开发中使用过，因此属于开发验证，不能作为从未接触的最终测试。二维关节预测也会受遮挡干扰，估计偏移不能替代真实同步标定。

### 不透明度、人物归属与遮挡要分别检查

![固定模型的逐人物渲染：原照片、联合结果、黑衣人物层、白衣人物层](../outputs/dynamic-human-enerf-actor-layers-069/camera06-frame000745-actors.png)
> 图源：同一公开数据与本站实际渲染 · 机位06、帧745 · 单独人物层仅改变参与渲染的高斯集合。

两个人共同渲染时，前方的人会遮住后方人物层的错误。上图将同一个冻结模型分别渲染：黑衣人物层胸前存在缺口和条纹，白衣人物层边缘存在颜色残留。联合画面上的低误差不能证明每个人物已经独立、完整地重建。单独显示的遮挡表面没有这个机位的无遮挡真值，不能据此断言其真实形状。

对某个像素，将全部人物高斯按深度排序。第 $i$ 个高斯的像素不透明度为 $\alpha_i$，到达它之前的透射率为

$$
T_i=\prod_{j<i}(1-\alpha_j).
$$

人物 $a$ 在联合渲染中的颜色与不透明度贡献为

$$
\mathbf C_a=\sum_{i\in\mathcal G_a}T_i\alpha_i\mathbf c_i,
\qquad W_a=\sum_{i\in\mathcal G_a}T_i\alpha_i.
$$

这里的 $T_i$ 包含所有人物的遮挡。若另把人物 $a$ 单独渲染，其他人物被移除，透射率也随之改变，所以单独渲染的不透明度与 $W_a$ 不相同。完整前景满足

$$
\mathbf C=\sum_a\mathbf C_a,\qquad A=\sum_a W_a,
\qquad \mathbf I=\mathbf C+(1-A)\mathbf b,
$$

$\mathbf b$ 是背景颜色。$W_a$ 和 $A$ 无量纲；它们说明当前模型怎样生成图像，不是真实人物分割。

![联合不透明度、逐人物可见贡献与单独人物的期望深度比较](../outputs/dynamic-human-enerf-actor-layers-069/camera06-frame000745-layers.png)
> 图源：本站同一冻结模型的原生分辨率输出 · 橙色为黑衣人物层，蓝色为白衣人物层。

左图白色表示不透明度接近1；中图展示每个人物在联合排序后实际留下的贡献；右图仅在两个单独人物层均有较强不透明度的位置比较期望深度，颜色表示较近的人物层，深度差以场景长度单位计，显示上限为0.25。期望深度是高斯中心深度的加权平均，并非真实表面。

九组画面的前景内部像素中，仅约2.39%的联合不透明度低于0.9；较大颜色误差像素中，约81.27%仍有至少0.9的不透明度。因此，当前残差不能单靠整体提高不透明度解释或修复。接下来需要用训练视角检查人物归属、可见表面支持与局部运动约束，再比较对应训练改动。详细条件和失败画面保存在[留出相机与人物层诊断](../experiments/37_dynamic_human_reconstruction.md#留出相机与人物层诊断)。

'''
p=L/'knowledge/32_dynamic_human_scene_reconstruction.md';s=p.read_text(encoding='utf8')
if '### 留出视角中的时间对应与外观验证' not in s:s=s.replace('### 连续运动中的同预算对照',addition+'### 连续运动中的同预算对照')
p.write_text(s,encoding='utf8')

p=L/'experiments/37_dynamic_human_reconstruction.md';s=p.read_text(encoding='utf8')
addition='''

### 留出相机与人物层诊断

068冻结066头颈基线与067局部变形模型，增加00、06、12三个相机的外观验证。这些相机未参与064—067的高斯初始化、训练与姿态拟合，但出现在更早的开发材料中。本轮不是全新最终测试。帧744—756中，21张偶数帧只用于拟合每个相机一个时间偏移，18张奇数帧用于外观评价；全部RGB保持原生1920×1080，按既有float32标定去畸变。新相机内外参沿用数据集，没有优化。

以固定身体和足部三次轨迹为参照，偏移估计为00：−0.488929帧、06：−0.281492帧、12：−0.705588帧。仅采用偶数帧RTMW身体与足部预测，置信度至少0.4、每个关节至少4个支持观测、零偏移中位投影误差不超过40像素；稳健损失尺度3像素，偏移限制±2帧。CPU独立标量最小化与记录结果最大差约3.39×10⁻⁷帧。遮挡导致部分关节预测混入另一人物，尤其机位06，验证预测的RMSE略变差；不能据此宣称完成真实同步标定。

三个分支各18张原生渲染：头颈基线加估计时间、局部变形加估计时间、局部变形加零偏移。模型检查点前后哈希一致，没有外观、身体或变形图更新；原195个身体上下文逐项保留。54张像素结果、216张PNG和54个原生人物对照均独立复算。

![留出机位实际全人物对照](../outputs/dynamic-human-enerf-heldout-cameras-068/paired-comparison/camera12-frame000751-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实际渲染 · 原照片、头颈基线、局部变形。

局部变形相对头颈基线，18张前景PSNR均改善，平均22.8839→23.7577dB（+0.8738dB）。每机位平均变化：00为+0.3628dB、06为+0.7647dB、12为+1.4939dB。固定投影前臂框的平均变化为黑衣+1.0987dB（7/9改善）、白衣+1.7789dB（9/9改善）。这些框包括背景与遮挡者；机位06的部分“黑衣前臂框”主要显示白衣衣服，不能把变化当作目标前臂清晰度。

![前景指标改善最小的画面](../outputs/dynamic-human-enerf-heldout-cameras-068/paired-comparison/camera00-frame000753-full-person.png)
> 图源：同上 · 机位00、帧753，前景PSNR变化+0.0062dB；它是最小改善，不是全前景退化。

![时间偏移的冻结模型对照](../outputs/dynamic-human-enerf-heldout-cameras-068/paired-comparison/camera12-frame000751-clock-comparison.png)
> 图源：同上 · 原照片、零时间偏移、估计时间偏移。

在同一个冻结局部变形模型上，估计时间相对零偏移的平均前景PSNR提高2.3411dB，但机位06的753、755帧没有改善。时间差分残差均值0.04647→0.03979，共15个相邻奇数帧组合；它包含运动、遮挡与掩码误差，并非运动补偿后的闪烁。三段六帧视频全部解码检查，4fps只用于展示：[00](../outputs/dynamic-human-enerf-heldout-cameras-068/temporal-comparison/camera00-reserved-time-comparison.mp4)、[06](../outputs/dynamic-human-enerf-heldout-cameras-068/temporal-comparison/camera06-reserved-time-comparison.mp4)、[12](../outputs/dynamic-human-enerf-heldout-cameras-068/temporal-comparison/camera12-reserved-time-comparison.mp4)。实际目视范围为9张预定全人物图、18个预定细节框、最小改善图以及3张751帧时间对照，没有声称完整视频播放检查。手指、鞋部和严重遮挡仍未通过清晰度验收。

068记录：[协议](../outputs/dynamic-human-enerf-heldout-cameras-068/protocol.json)、[时间输入独立检查](../outputs/dynamic-human-enerf-heldout-cameras-068/clock-input-independent-audit.json)、[二维观察局限](../outputs/dynamic-human-enerf-heldout-cameras-068/observation-visual-review.json)、[原生像素复算](../outputs/dynamic-human-enerf-heldout-cameras-068/independent-image-audit.json)、[成对指标](../outputs/dynamic-human-enerf-heldout-cameras-068/paired-comparison/summary.json)、[连续检查](../outputs/dynamic-human-enerf-heldout-cameras-068/temporal-comparison/summary.json)、[实际看图范围](../outputs/dynamic-human-enerf-heldout-cameras-068/visual-review.json)、[结论](../outputs/dynamic-human-enerf-heldout-cameras-068/final-summary.json)。

069继续冻结067模型，在同样三个相机的745、751、755帧共9个上下文分别输出联合颜色、每个人物的联合颜色与不透明度贡献、单独人物颜色与不透明度、期望深度。身份沿用最近人体顶点绑定，非实例真值。特征渲染与068原始RGB、alpha逐像素一致；颜色贡献之和与总颜色、不透明度贡献之和与总不透明度的最大绝对差均小于5×10⁻⁶。保存数组、18张对照图及统计均在CPU独立复算。

![冻结模型中的实际人物层](../outputs/dynamic-human-enerf-actor-layers-069/camera06-frame000745-actors.png)
> 图源：同上 · 原照片、联合结果、黑衣人物层、白衣人物层。

按释放前景matte>0.95并腐蚀5×5像素得到的1600009个内部像素中，38294个联合alpha<0.9（2.3934%）；RGB平均绝对误差>0.08的119058个内部像素中，96758个仍有alpha≥0.9（81.2696%）。阈值在分层渲染前固定。这支持“残差并非全部由低alpha造成”，不证明特定几何或人物归属是唯一原因。

机位06的黑衣胸前缺口、白衣层边缘残留，以及00/12机位遮挡手部的肤色散点，是后续可见性与归属约束的检查对象。单独人物层展示了被遮挡的预测区域，但没有这个机位的无遮挡真值；不会据此强行补满透明度或按留出RGB重新指定人物。目视检查5张人物层图、3张机位06贡献图，其余图只做像素和统计检查。

069记录：[固定诊断条件](../outputs/dynamic-human-enerf-actor-layers-069/protocol.json)、[逐图贡献统计](../outputs/dynamic-human-enerf-actor-layers-069/render-summary.json)、[独立像素复算](../outputs/dynamic-human-enerf-actor-layers-069/independent-audit.json)、[看图与下一步](../outputs/dynamic-human-enerf-actor-layers-069/visual-review.json)、[结论](../outputs/dynamic-human-enerf-actor-layers-069/final-summary.json)。保留067为开发基线；下一步仅依据训练数据分析归属与可见支持，再进行同预算训练对照。
'''
if '### 留出相机与人物层诊断' not in s:p.write_text(s+addition,encoding='utf8')

assets={H.name:['protocol.json','clock-input-independent-audit.json','observation-visual-review.json','independent-image-audit.json','paired-comparison/summary.json','temporal-comparison/summary.json','visual-review.json','final-summary.json','paired-comparison/camera12-frame000751-full-person.png','paired-comparison/camera00-frame000753-full-person.png','paired-comparison/camera12-frame000751-clock-comparison.png']+[f'temporal-comparison/camera{c}-reserved-time-comparison.mp4' for c in ['00','06','12']],
D.name:['protocol.json','render-summary.json','independent-audit.json','visual-review.json','final-summary.json','camera06-frame000745-actors.png','camera06-frame000745-layers.png']}
p=S/'scripts/sync-gs-lab.mjs';s=p.read_text(encoding='utf8')
for root,names in assets.items():
 for name in names:assert (L/'outputs'/root/name).is_file()
 if root+'/${name}' not in s:
  lines=s.splitlines(keepends=True);i=next(i for i,line in enumerate(lines) if 'dynamic-human-enerf-residual-graph-067/${name}' in line)+1
  lines.insert(i,'  ...'+json.dumps(names)+f".map(name => [`outputs/{root}/${{name}}`, '留出视角与人物层验证']),\n");s=''.join(lines)
p.write_text(s,encoding='utf8')
p=L/'manifests/dynamic-human-work.json';m=read(p)
m.update(status='heldout_camera_and_actor_layer_diagnostics_complete_clarity_unaccepted',current_runs=[],last_recorded_utc=datetime.now(timezone.utc).isoformat(),
 outdoor_heldout_cameras='outputs/'+H.name+'/final-summary.json',outdoor_actor_layers='outputs/'+D.name+'/final-summary.json',
 outdoor_heldout_website_qa='outputs/'+H.name+'/website-qa.json',outdoor_actor_layers_website_qa='outputs/'+D.name+'/website-qa.json',
 goal_turn_classification='Progress:3heldoutappearancecameras/54nativeimages/3clocks independently checked;9frozenactorlayercontexts/18panels independently checked; actualvisualreview finds occludedlayerdefects. No model updates this turn;067 retained. Overallclarity notaccepted.',
 remaining_quality_issue='Hands/fingers/shoes remain soft; camera06 overlaps have isolated chest holes and adjacent actor-layer color remnants. Positive full-foreground PSNR is not sufficient for clarity.',
 next_actions=['Inspect067 training-only per-actor visibility and canonical anchor displacement at occluding boundaries; quantify support without using heldout RGB to assign labels.',
 'Run a matched-budget training ablation of targeted surface/ownership constraints supported by that diagnosis; preserve067 checkpoint and negative cases.',
 'Check fingers, shoes, occlusion and continuous videos on fixed development views, then a fresh time segment before any large-scene completion claim.'])
entry='068frozen3camera54nativepixelchecks21evenclockcalibration18oddappearance;069nineactorlayers18panelsCPUchecked;clarityunaccepted'
if entry not in m['completed']:m['completed'].append(entry)
save(p,m);shutil.copy2(__file__,H/Path(__file__).name)
print(json.dumps(dict(assets={k:len(v) for k,v in assets.items()},status=m['status'])))
