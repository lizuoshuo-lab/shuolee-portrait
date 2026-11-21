"""Publish only after numeric and visual evaluation have a recorded decision."""
from pathlib import Path
from datetime import datetime,timezone
import json,shutil
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-actor-supervision-070';S=Path('D:/Projects/EngineLab/engine-home')
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
f=read(R/'final-summary.json');v=read(R/'visual-review.json');assert f['goal_complete'] is False
known=f['known_camera_validation'];novel=f['heldout_camera_validation'];ownership=f['training_ownership'];protocol=read(R/'training-protocol-audit.json')
knowledge=r'''
### 人物归属监督怎样作用于联合渲染

![训练照片与保守人物内部标签，橙色表示黑衣人物、蓝色表示白衣人物](../outputs/dynamic-human-enerf-actor-supervision-070/masks/camera07-labels-review.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap；SAM 2.1分割与本站标签筛选 · 原生训练机位07、帧750。

外观损失只约束合成后的颜色。如果某个人物的高斯替邻近人物解释了部分像素，总颜色仍可能接近照片。人物归属监督直接约束联合渲染中每个人物的可见贡献，而不是仅检查高斯在三维空间中离哪个人体更近。

沿用上节的人物不透明度贡献 $W_a(\mathbf p)$ 与总不透明度 $A(\mathbf p)$，定义归一化贡献

$$
P_a(\mathbf p)=\frac{W_a(\mathbf p)+\varepsilon}{A(\mathbf p)+2\varepsilon},
\qquad \varepsilon=10^{-6}.
$$

两个人物的贡献之和等于总不透明度。$P_a$ 是当前渲染模型归一化后的贡献，不是真实分割概率。设 $\mathcal M_a$ 为保守筛选后属于人物 $a$ 的训练像素集合，按人物分别平均的损失为

$$
\mathcal L_{\mathrm{inst}}
=-\frac12\sum_{a=0}^{1}\frac{1}{|\mathcal M_a|}
\sum_{\mathbf p\in\mathcal M_a}\log P_a(\mathbf p).
$$

分别平均可避免被遮挡人物仅有少量像素时，其监督完全被前景人物覆盖。本地对照在既有颜色、透明度与局部变形损失之外，增加权重0.05的该项；只有经过看图接受的13个训练机位的第750帧使用人物标签，其他照片保持原监督。分割内部腐蚀7×7像素，另一人物及手持物分割膨胀7×7后排除，并与原前景matte大于0.95的有效像素取交集。不确定位置被忽略，不当作背景，也不据此填补被遮挡表面。

__RESULT__

![同一原生画面的基线与人物归属监督对照](../outputs/dynamic-human-enerf-actor-supervision-070/heldout/paired-comparison/camera06-frame000745-full-person.png)
> 图源：同一公开数据与本站实际训练 · 原照片、局部变形基线、增加人物归属监督。

分割是额外的弱监督，不能当作实例真值；整体误差、人物层归属和肉眼可辨的细节必须分别检查。完整协议、负例与检查范围见[人物归属训练对照](../experiments/37_dynamic_human_reconstruction.md#人物归属监督的同预算训练对照)。

'''
result=f"相同105张训练照片、初始高斯、视图抽样与15000次更新下，原训练机位的90张留出时间图前景PSNR平均变化{known['mean_delta_db']:+.3f}dB，另外三个机位的18张图变化{novel['mean_delta_db']:+.3f}dB。{v['narrative_zh']} 这些画面已用于开发验证，本轮没有完成大场景或清晰人体的最终验收。"
knowledge=knowledge.replace('__RESULT__',result)
p=L/'knowledge/32_dynamic_human_scene_reconstruction.md';text=p.read_text(encoding='utf8')
if '### 人物归属监督怎样作用于联合渲染' not in text:text=text.replace('### 连续运动中的同预算对照',knowledge+'### 连续运动中的同预算对照')
p.write_text(text,encoding='utf8')
worst=read(R/'heldout/paired-comparison/summary.json')['worst_change'];worstfile=f"camera{worst['camera']}-frame{worst['frame']:06d}-full-person.png"
evidence=f'''

### 人物归属监督的同预算训练对照

070以067局部变形模型为基线，从相同初始化重新训练15000次更新。原始105张照片、matte、固定身体与时间模型、384节点图、随机背景、抽样次序及学习率完全一致。额外监督来自第750帧的SAM2.1 Hiera Tiny人物分割：复用旧训练照片的人工提示，在当前精确去畸变RGB上重新推理；15张标签图逐一查看，排除13、17机位的跨人物小区域误标，13个机位共2117574个保守内部像素用于归属损失。不是增加13张照片，而是对已有13张训练照片增加弱标签。手持物和不确定交界不参与该损失。

原始绑定距离检查发现，仅约0.0273%的高斯比自身绑定人物更接近另一人体至少0.05场景单位，因此没有把全体高斯重新分配身份。训练机位的联合贡献检查仍显示重叠机位存在错误归属；三维最近距离不能代替可见像素归属。

实际完成{protocol['ownership_supervised_updates']}次含归属损失的更新，总更新15000次，权重0.05。新增特征通道的RGB与alpha在实际CUDA测试中与原渲染器一致，逐人物贡献之和与总alpha误差小于5×10⁻⁶；位置、透明度与变形参数梯度非零。训练后CPU独立复算全部195个身体上下文的节点旋转、固定样本高斯的线性变换，保持正行列式。最终{protocol['candidate_points']:,}个高斯，基线为{protocol['baseline_points']:,}；相同更新次数不等于相同运算量。SAM预测、许可人体派生数组及检查点不作为网站下载资源。

![用于弱监督的实际标签](../outputs/dynamic-human-enerf-actor-supervision-070/masks/camera07-labels-review.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap、SAM2.1及本站筛选。

90张已知训练机位的留出时间图，前景PSNR均值{known['before_mean_psnr_db']:.4f}→{known['after_mean_psnr_db']:.4f}dB，{known['improved_images']}/90张改善；18张另外三个机位的奇数帧为{novel['before_mean_psnr_db']:.4f}→{novel['after_mean_psnr_db']:.4f}dB，{novel['improved_images']}/18张改善。这三个机位保持068的时间估计，没有为新模型重拟合。全部120张原生结果逐项复算。

![固定遮挡机位对照](../outputs/dynamic-human-enerf-actor-supervision-070/heldout/paired-comparison/camera06-frame000745-full-person.png)
> 图源：同上 · 原照片、067基线、070归属监督。

{v['narrative_zh']} 采用结论：{v['decision_zh']}。

![留出机位中变化最差的完整画面](../outputs/dynamic-human-enerf-actor-supervision-070/heldout/paired-comparison/{worstfile})
> 图源：同上 · 机位{worst['camera']}、帧{worst['frame']}，前景变化{worst['foreground_delta_db']:+.4f}dB。

与训练弱标签不一致的主导人物贡献像素从{ownership['before_wrong_pixels']}变为{ownership['after_wrong_pixels']}，不能直接解释为真实实例精度。固定模型的逐人物渲染仍需目视对照；被遮挡表面没有该机位的无遮挡真值。时间检查保留75组已知机位、15组另外机位的图像差分残差，以及7段六帧视频；视频4fps只用于展示，全部机械解码，实际看图范围以记录为准，不宣称完整视频播放审阅。

报告：[输入与标签规则](../outputs/dynamic-human-enerf-actor-supervision-070/mask-protocol.json)、[标签看图筛选](../outputs/dynamic-human-enerf-actor-supervision-070/mask-visual-review.json)、[标签独立检查](../outputs/dynamic-human-enerf-actor-supervision-070/mask-independent-audit.json)、[绑定距离](../outputs/dynamic-human-enerf-actor-supervision-070/anchor-diagnostic.json)、[CUDA检查](../outputs/dynamic-human-enerf-actor-supervision-070/actor-runtime-smoke.json)、[实际训练条件](../outputs/dynamic-human-enerf-actor-supervision-070/training-protocol-audit.json)、[已训练变形复算](../outputs/dynamic-human-enerf-actor-supervision-070/trained-graph-independent-audit.json)、[已知机位比较](../outputs/dynamic-human-enerf-actor-supervision-070/paired-comparison/summary.json)、[另外机位比较](../outputs/dynamic-human-enerf-actor-supervision-070/heldout/paired-comparison/summary.json)、[已知机位连续变化](../outputs/dynamic-human-enerf-actor-supervision-070/temporal-comparison/summary.json)、[另外机位连续变化](../outputs/dynamic-human-enerf-actor-supervision-070/heldout/temporal-comparison/summary.json)、[人物层独立核验](../outputs/dynamic-human-enerf-actor-supervision-070/actor-layers/independent-audit.json)、[看图范围](../outputs/dynamic-human-enerf-actor-supervision-070/visual-review.json)、[最终判断](../outputs/dynamic-human-enerf-actor-supervision-070/final-summary.json)。
'''
p=L/'experiments/37_dynamic_human_reconstruction.md';text=p.read_text(encoding='utf8')
if '### 人物归属监督的同预算训练对照' not in text:p.write_text(text+evidence,encoding='utf8')
assets=['mask-protocol.json','mask-visual-review.json','mask-independent-audit.json','anchor-diagnostic.json','actor-runtime-smoke.json','training-protocol-audit.json','trained-graph-independent-audit.json','paired-comparison/summary.json','heldout/paired-comparison/summary.json','temporal-comparison/summary.json','heldout/temporal-comparison/summary.json','actor-layers/independent-audit.json','visual-review.json','final-summary.json','masks/camera07-labels-review.png','heldout/paired-comparison/camera06-frame000745-full-person.png','heldout/paired-comparison/'+worstfile]
assets=list(dict.fromkeys(assets))
for name in assets:assert (R/name).is_file()
p=S/'scripts/sync-gs-lab.mjs';text=p.read_text(encoding='utf8')
if R.name+'/${name}' not in text:
 lines=text.splitlines(keepends=True);i=next(i for i,line in enumerate(lines) if 'dynamic-human-enerf-actor-layers-069/${name}' in line)+1;lines.insert(i,'  ...'+json.dumps(assets)+f".map(name => [`outputs/{R.name}/${{name}}`, '人物归属监督训练']),\n");p.write_text(''.join(lines),encoding='utf8')
p=L/'manifests/dynamic-human-work.json';m=read(p);m.update(status='actor_ownership_training_and_evaluation_complete_clarity_unaccepted',current_runs=[],last_recorded_utc=datetime.now(timezone.utc).isoformat(),outdoor_actor_supervision='outputs/'+R.name+'/final-summary.json',outdoor_actor_supervision_website_qa='outputs/'+R.name+'/website-qa.json',remaining_quality_issue=v['narrative_zh'],next_actions=v['next_actions'],goal_turn_classification='Progress:070actual15000updates with13reviewedweaklabels,120nativeimages,195contextCPUaudit,9actorlayers and recordedvisualreviews complete. Nooverallclaritycompletion.')
if v['adopt_as_development_baseline']:m['current_best_continuous_control']='outputs/'+R.name+'/residual_graph'
entry='070actual15000updates1799weakownershipsteps,120nativeimages,9actorlayers,90temporaltransitions;clarityunaccepted'
if entry not in m['completed']:m['completed'].append(entry)
save(p,m);shutil.copy2(__file__,R/Path(__file__).name);print(json.dumps(dict(assets=len(assets),adopt=v['adopt_as_development_baseline'])))
