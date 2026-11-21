from pathlib import Path
import json
from datetime import datetime,timezone
LAB=Path(__file__).resolve().parents[2];SITE=Path('D:/Projects/EngineLab/engine-home')
p=LAB/'knowledge/32_dynamic_human_scene_reconstruction.md';text=p.read_text(encoding='utf8')
addition=r'''### 掩码中的未知区域与几何观测覆盖

![原照片、完整掩码监督与忽略掩码冲突区域的重建对照](../outputs/dynamic-human-enerf-mask-uncertainty-053/paired-comparison/camera08-frame000750-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站运行 · 训练诊断机位 08 · 原生像素，无增强。

前景掩码漏掉一段前臂时，把该处当成背景会错误地惩罚真实人体；补全掩码若把墙面也包含进来，又会让模型学习背景纹理。上图右列在一部分手臂附近减少了墙面残留，但手臂拉伸并未消失。掩码处理解决的是监督像素的可靠性，不能直接保证三维表面正确。

设 $H(p)$ 是补全后的二值前景掩码，$S(p)$ 是另一分割模型的二值前景结果，$V(p)$ 表示图像去畸变后是否有效。两者冲突的区域可以暂作未知观测，定义损失权重

$$
\omega(p)=V(p)\begin{cases}
0,&H(p)=1\ \land\ S(p)=0,\\
1,&\text{其他情况}.
\end{cases}
$$

未知区域不会被强制标成背景。这种做法也会移除一部分真实皮肤和衣服的监督，因此需要其他相机提供支持。对逐像素颜色项，可用 $\omega(p)$ 加权；SSIM 则依赖局部窗口，直接把无效像素置零还会改变邻域统计，必须说明采用的掩码方式。这里的本地对照沿用既有掩码图像损失，不能解释为所有监督误差均被排除。

几何检查也有覆盖范围。三台以上相机中的纹理匹配可以三角化为空间点，再与渲染深度比较；但鞋带和玩具上的匹配很准确，并不说明缺少纹理的前臂和手指也正确。检查时既要看重投影误差，也要看观测点落在身体哪些位置。对应的掩码与几何检查见[实验记录](../experiments/37_dynamic_human_reconstruction.md#掩码冲突与稀疏表面检查)。

'''
if '### 掩码中的未知区域与几何观测覆盖' not in text:
 anchor='### 增密梯度与细节恢复';assert text.count(anchor)==1;text=text.replace(anchor,addition+anchor);p.write_text(text,encoding='utf8')
p=LAB/'experiments/37_dynamic_human_reconstruction.md';text=p.read_text(encoding='utf8')
addition='''

### 掩码冲突与稀疏表面检查

以补全掩码为前景、SAM 掩码为非前景的像素作为未知，只关闭该处的损失有效性，不改原始 RGB、目标掩码、相机、初始高斯或 6,000 次采样序列。15 个训练机位总计忽略 46,344 个像素，分别占各自前景的 0.30%～4.86%。最大的 12 个冲突区域和三张完整输入图已查看：这些区域同时包含墙面和真实手臂、衣服，不能统一改成背景。

完成 6,000 次更新后得到 137,951 个高斯。三个留出视角 00、06、12 的前景 PSNR 变化分别为 −0.174、+0.071、+0.037 dB。六张完整人物对照与预先确定的 17 个细节区域均已查看，机位 08 的部分墙面残留减少，但手臂、手指和交叉遮挡仍不合格。见[训练结论](../outputs/dynamic-human-enerf-mask-uncertainty-053/final-summary.json)、[输入与成对状态核验](../outputs/dynamic-human-enerf-mask-uncertainty-053/paired-training-state-audit.json)、[原生图像核验](../outputs/dynamic-human-enerf-mask-uncertainty-053/native-evaluation/independent-image-audit.json)和[图像审阅](../outputs/dynamic-human-enerf-mask-uncertainty-053/visual-review.json)。

![原照片、原监督与忽略冲突区域的训练机位 08 效果](../outputs/dynamic-human-enerf-mask-uncertainty-053/paired-comparison/camera08-frame000750-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站运行 · 无增强。

为检验表面位置，从已有训练相机互匹配中建立跨三台以上相机的纹理轨迹；拒绝含重复相机的冲突轨迹，要求多视角三角化后每个观测重投影误差小于 1 像素、最大三角化夹角大于 2°、深度均为正。最后保留 90 条轨迹、375 个观测，未使用留出照片。

这些点与同帧 3DGS 基线的期望深度相比，相对差异中位数为 0.326%，第 90 百分位为 1.001%；使用模型深度反投影后再投到其他观测机位，误差中位数为 1.358 像素。15 张深度图、375 个采样和投影关系已重算，12 组三视角原生照片／渲染局部图及三个机位的覆盖图已查看。匹配明显集中在着地鞋、玩具、面部和少量衣服位置，裸露前臂、手指、抬起的鞋缺少直接支持。这不是独立深度真值，也不能作为人体几何已经恢复的证明。见[几何结论](../outputs/dynamic-human-enerf-surface-tracks-054/final-summary.json)、[独立重算](../outputs/dynamic-human-enerf-surface-tracks-054/independent-audit.json)和[对应与覆盖审阅](../outputs/dynamic-human-enerf-surface-tracks-054/visual-review.json)。
'''
if '### 掩码冲突与稀疏表面检查' not in text:p.write_text(text+addition,encoding='utf8')
p=SITE/'scripts/sync-gs-lab.mjs';text=p.read_text(encoding='utf8')
if 'dynamic-human-enerf-mask-uncertainty-053/${name}' not in text:
 lines=text.splitlines(keepends=True);idx=next(i for i,l in enumerate(lines) if 'dynamic-human-enerf-absgrad-052/${name}' in l)+1
 extra="  ...['final-summary.json','paired-training-state-audit.json','native-evaluation/independent-image-audit.json','visual-review.json','paired-comparison/camera08-frame000750-full-person.png'].map(name => [`outputs/dynamic-human-enerf-mask-uncertainty-053/${name}`, '掩码冲突监督对照']),\n"
 extra+="  ...['final-summary.json','independent-audit.json','visual-review.json'].map(name => [`outputs/dynamic-human-enerf-surface-tracks-054/${name}`, '训练相机稀疏表面几何检查']),\n"
 lines.insert(idx,extra);p.write_text(''.join(lines),encoding='utf8')
p=LAB/'manifests/dynamic-human-work.json';m=json.loads(p.read_text(encoding='utf8'))
m['current_runs']=[dict(name='055surface-only then surface-regularized',exec_session=3786,state='training_running_verified_step3000_output')]
m['last_recorded_utc']=datetime.now(timezone.utc).isoformat();p.write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
print('053054knowledge/evidence ready for sync;055live training recorded')
