from pathlib import Path
from datetime import datetime,timezone
import json,shutil,numpy as np
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-residual-graph-067';SITE=Path('D:/Projects/EngineLab/engine-home')
read=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,x:p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
pair=read(R/'paired-comparison/summary.json');temporal=read(R/'temporal-comparison/summary.json');visual=read(R/'visual-review.json');trained=read(R/'residual_graph/completion.json');audit=read(R/'trained-graph-independent-audit.json')
assert trained['step']==15000 and read(R/'residual_graph/native-evaluation/independent-image-audit.json')['images']==102
assert len(visual['full_person'])==12 and len(visual['details'])==24 and visual['additional_worst']
v=pair['validation_foreground'];tm=temporal['mean_temporal_residual'];worst=pair['worst_change'];worstfile=f"camera{worst['camera']}-frame{worst['frame']:06d}-full-person.png"
assert visual['additional_worst']==worstfile
detail={n:dict(mean_psnr_delta_db=float(np.mean([r['delta_db'] for r in pair['details'] if r['region']==n])),improved=sum(r['delta_db']>0 for r in pair['details'] if r['region']==n),count=6) for n in ['dark_head','dark_forearm','light_forearm','dark_shoe']}
save(R/'final-summary.json',dict(status='15000update_graph_reconstruction102native_images_and_independent_graph_checks_complete',validation_foreground=v,fixed_detail_regions=detail,temporal_difference_residual=tm,visual_findings=visual['findings'],development_baseline_adopted=visual['adopt_as_development_baseline'],qualification='Local research extension; single seed; sameinitial105images15000updates, notsameGaussiancount orcompute. Repeated development-time validation, not fresh final tests. Mattes/keypoints notGT. No newcamera/generalization/large-scene completion claim.',goal_complete=False))
p=L/'knowledge/32_dynamic_human_scene_reconstruction.md';text=p.read_text(encoding='utf8')
addition=r'''

### 骨架之外的局部变形如何进入渲染

![相同原生画面的骨架运动与局部变形重建对照](../outputs/dynamic-human-enerf-residual-graph-067/paired-comparison/camera04-frame000745-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实际训练 · 原照片、头颈修正基线、增加局部变形。

骨架适合描述身体的大体姿态，衣物、手指和手持物还可能相对骨架运动。变形图（deformation graph）用一组邻近控制节点描述这部分变化。本地扩展先应用已有骨架运动，再为每个高斯关联四个控制节点；节点权重随该高斯继承，训练中的复制与分裂也保留这种关联。这是本站的研究对照实现，并非某篇论文官方实现的复现结论。

设骨架变换后的高斯中心与协方差为 $\mathbf x_i$、$\mathbf C_i$，节点位置为 $\mathbf g_k$，节点的残余旋转和平移为 $\mathbf R_k(t)$、$\mathbf d_k(t)$，固定权重满足 $w_{ik}\geq0$、$\sum_k w_{ik}=1$。进一步变形后的中心为

$$
\widetilde{\mathbf x}_i(t)=\sum_{k\in\mathcal N_i}w_{ik}
\left[\mathbf R_k(t)(\mathbf x_i-\mathbf g_k)+\mathbf g_k+\mathbf d_k(t)\right].
$$

中心和位移使用同一场景长度单位；旋转矩阵与权重无量纲。对于权重固定的单个高斯，这个变换的线性部分及其协方差传播为

$$
\mathbf L_i(t)=\sum_{k\in\mathcal N_i}w_{ik}\mathbf R_k(t),
\qquad
\widetilde{\mathbf C}_i(t)=\mathbf L_i(t)\mathbf C_i\mathbf L_i(t)^{\mathsf T}.
$$

因此，局部运动既改变中心，也改变投影前的形状与方向。球谐颜色的观察方向也要逆变换回规范坐标，才能保持外观与表面运动的对应。节点旋转采用有界轴角参数，避免加权线性变换接近奇异。

为使表示在参考时刻保持一致，本地模型的残余旋转向量和平移均使用无常数项的三次多项式：

$$
\boldsymbol\omega_k(\tau)=\sum_{q=1}^{3}\mathbf a_{kq}\tau^q,
\qquad
\mathbf d_k(\tau)=\sum_{q=1}^{3}\mathbf b_{kq}\tau^q,
\qquad \tau=\frac{t-750}{6}.
$$

$t$ 是连续帧索引，$\mathbf a_{kq}$ 以弧度计，$\mathbf b_{kq}$ 使用场景长度单位。在 $t=750$ 时，残余旋转为单位矩阵、平移为零。相邻节点另受局部刚性约束，减少无规律扭曲；这类约束不会自动赋予模型真实手指关节或独立物体结构。

__RESULT_PARAGRAPH__

'''
result=f"实际相同更新预算对照中，90张开发时间验证图的前景PSNR均值从 {v['before_mean_psnr_db']:.2f} 变为 {v['after_mean_psnr_db']:.2f} dB，{v['improved_images']}张改善。{visual['narrative_zh']} 具体条件、细节与负例见[局部变形实验](../experiments/37_dynamic_human_reconstruction.md#骨架之后的局部变形对照)。这些图已用于开发，仍需要新的时间片段和相机检验泛化。"
addition=addition.replace('__RESULT_PARAGRAPH__',result)
if '### 骨架之外的局部变形如何进入渲染' not in text:text=text.replace('### 连续运动中的同预算对照',addition+'### 连续运动中的同预算对照')
p.write_text(text,encoding='utf8')
p=L/'experiments/37_dynamic_human_reconstruction.md';text=p.read_text(encoding='utf8')
addition=f'''

### 骨架之后的局部变形对照

在066的固定身体与头颈运动上，加入384个局部节点，每个人192个；节点由仅训练数据构成的初始视觉外壳确定，每个高斯关联4个同一人物的节点。3072条有向邻接边约束局部刚性。两个三维三次系数张量共6912个可训练标量，参考物理帧索引750处的残余变形严格为恒等。身份关联来自最近人体顶点，是近似绑定，并非实例真值；玩偶尚无独立节点类别。

RGB与透明度损失、105张训练照片、高斯初值、抽样、随机背景及15000次更新均同066一致；增加的归一化局部刚性、旋转幅度、平移幅度正则权重分别为10、0.001、0.1，训练前确定。图参数Adam初始学习率0.003，指数降至初值0.1倍。最终有{trained['points']:,}个高斯；相同更新次数不代表相同计算量或最终高斯数量。模型是本地研究扩展，未冒充作者官方基线。

训练前验证了恒等变换、刚体等变性、有限差分梯度、实际CUDA反向和复制／分裂绑定。训练后在CPU以SciPy独立复算全部384节点、195时刻组合的旋转，并在每个组合抽查512个高斯的加权线性变换；最小奇异值为{audit['sampled_min_singular_value']:.6f}。完整102张渲染保存像素和指标均另行复核。

![相同原生画面的全人物比较](../outputs/dynamic-human-enerf-residual-graph-067/paired-comparison/camera04-frame000745-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站训练 · 原照片、066、067。

90张开发时间验证图的前景PSNR从{v['before_mean_psnr_db']:.4f}变为{v['after_mean_psnr_db']:.4f}dB，变化{v['mean_delta_db']:+.4f}dB，{v['improved_images']}/90张改善。固定区域平均变化：头部{detail['dark_head']['mean_psnr_delta_db']:+.4f}dB、黑衣前臂{detail['dark_forearm']['mean_psnr_delta_db']:+.4f}dB、白衣前臂{detail['light_forearm']['mean_psnr_delta_db']:+.4f}dB、鞋部{detail['dark_shoe']['mean_psnr_delta_db']:+.4f}dB。每类6个区域，包含背景和遮挡，不能当作精确部位分割指标。

{visual['narrative_zh']} 实际查看了12个固定全图、24个细节区域及全前景退化最大的样例，**清晰人物重建仍未通过**。

![最差变化的完整样例](../outputs/dynamic-human-enerf-residual-graph-067/paired-comparison/{worstfile})
> 图源：同上 · 机位{worst['camera']}、第{worst['frame']}帧，前景PSNR变化{worst['foreground_delta_db']:+.4f}dB。

75组连续差分残差均值从{tm['before']:.5f}变为{tm['after']:.5f}。它包含运动、遮挡及掩码误差，不等同于真实运动或闪烁。四段六帧视频全部解码检查，目视范围为上述离散画面；4帧/秒仅用于展示：[01](../outputs/dynamic-human-enerf-residual-graph-067/temporal-comparison/camera01-reserved-time-comparison.mp4)、[04](../outputs/dynamic-human-enerf-residual-graph-067/temporal-comparison/camera04-reserved-time-comparison.mp4)、[08](../outputs/dynamic-human-enerf-residual-graph-067/temporal-comparison/camera08-reserved-time-comparison.mp4)、[17](../outputs/dynamic-human-enerf-residual-graph-067/temporal-comparison/camera17-reserved-time-comparison.mp4)。

报告：[结论](../outputs/dynamic-human-enerf-residual-graph-067/final-summary.json)、[节点准备](../outputs/dynamic-human-enerf-residual-graph-067/graph-preparation.json)、[数学检查](../outputs/dynamic-human-enerf-residual-graph-067/math-audit.json)、[实际CUDA检查](../outputs/dynamic-human-enerf-residual-graph-067/runtime-smoke.json)、[训练条件](../outputs/dynamic-human-enerf-residual-graph-067/residual_graph/graph-protocol.json)、[训练后独立复算](../outputs/dynamic-human-enerf-residual-graph-067/trained-graph-independent-audit.json)、[图像复算](../outputs/dynamic-human-enerf-residual-graph-067/residual_graph/native-evaluation/independent-image-audit.json)、[成对指标](../outputs/dynamic-human-enerf-residual-graph-067/paired-comparison/summary.json)、[连续检查](../outputs/dynamic-human-enerf-residual-graph-067/temporal-comparison/summary.json)、[看图范围](../outputs/dynamic-human-enerf-residual-graph-067/visual-review.json)。
'''
if '### 骨架之后的局部变形对照' not in text:p.write_text(text+addition,encoding='utf8')
p=SITE/'scripts/sync-gs-lab.mjs';text=p.read_text(encoding='utf8')
assets=['final-summary.json','graph-preparation.json','math-audit.json','runtime-smoke.json','residual_graph/graph-protocol.json','trained-graph-independent-audit.json','residual_graph/native-evaluation/independent-image-audit.json','paired-comparison/summary.json','temporal-comparison/summary.json','visual-review.json','paired-comparison/camera04-frame000745-full-person.png','paired-comparison/'+worstfile]+[f'temporal-comparison/camera{c}-reserved-time-comparison.mp4' for c in ['01','04','08','17']]
if 'dynamic-human-enerf-residual-graph-067/${name}' not in text:
 lines=text.splitlines(keepends=True);i=next(i for i,l in enumerate(lines) if 'dynamic-human-enerf-head-motion-066/${name}' in l)+1
 lines.insert(i,'  ...'+json.dumps(assets)+".map(name => [`outputs/dynamic-human-enerf-residual-graph-067/${name}`, '骨架之外的局部运动']),\n");p.write_text(''.join(lines),encoding='utf8')
p=L/'manifests/dynamic-human-work.json';m=read(p);m.update(status='residual_graph_training_evaluation_and_visual_review_complete_clarity_unaccepted',current_runs=[],outdoor_residual_graph='outputs/dynamic-human-enerf-residual-graph-067/final-summary.json',outdoor_head_website_qa='outputs/dynamic-human-enerf-head-motion-066/website-qa.json',outdoor_residual_graph_website_qa='outputs/dynamic-human-enerf-residual-graph-067/website-qa.json',last_recorded_utc=datetime.now(timezone.utc).isoformat(),remaining_quality_issue=visual['narrative_zh'],goal_turn_classification='Progress:066isolated head motion and067actual residualgraph15000updates/102nativeimages/CPUgraphaudit/fixedvisualreview completed. Overallclarity notpassed.',next_actions=visual['next_actions'])
if visual['adopt_as_development_baseline']:m['current_best_continuous_control']='outputs/dynamic-human-enerf-residual-graph-067/residual_graph'
entry='067actual15000update residualgraph,102nativepixel audits,195contextCPUcheck,12full24detailworstreview,75temporalchecks4decodedvideos; claritynotpassed'
if entry not in m['completed']:m['completed'].append(entry)
save(p,m);shutil.copy2(__file__,R/Path(__file__).name);print(json.dumps(dict(validation=v,details=detail,temporal=tm),ensure_ascii=False))
