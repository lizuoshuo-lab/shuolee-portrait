"""Update existing theory and evidence pages with bounded new-interval findings."""
from pathlib import Path
import json,shutil
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-second-interval-079';M=L/'outputs/dynamic-human-enerf-matte-diagnosis-080';S=Path('D:/Projects/EngineLab/engine-home')
read=lambda p:json.loads(p.read_text(encoding='utf8'))
assert read(R/'final-summary.json')['actual_panels_reviewed']==52
assert not read(R/'final-summary.json')['clarity_accepted']
assert read(M/'summary.json')['training_changed'] is False
p=L/'knowledge/32_dynamic_human_scene_reconstruction.md';s=p.read_text(encoding='utf8')
if not (R/'knowledge-before.md').exists():shutil.copy2(p,R/'knowledge-before.md')
old='![原照片、身体运动基线与保留的局部变形模型在新机位的实际效果](../outputs/dynamic-human-enerf-heldout-cameras-068/paired-comparison/camera12-frame000751-full-person.png)\n> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实际训练 · 机位12、帧751 · 原生像素，无增强。'
new='![另一运动片段的原照片、全局三次时间模型与局部时间模型](../outputs/dynamic-human-enerf-second-interval-079/comparison/camera04-frame000731-full-person.png)\n> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 机位04、留出帧731 · 原生像素，无增强。'
if old in s:s=s.replace(old,new,1)
old='当前保留的局部变形模型已经恢复人物主体、衣物图案和手持物的大体外观，但手指、鞋部及交叉遮挡还没有达到清晰重建要求。灰色是用于检查人物的合成背景，此处尚未联合重建建筑。近期的[运动节点密度对照](#运动控制节点的密度与观测支持)与[局部掩码修正](#掩码怎样改变颜色监督)分别检验模型容量和监督质量；具体取舍见各自的真实画面对照。'
new='人物主体、衣服图案和腿部已能辨认，张开的手掌、手持物及抬起的鞋仍有模糊和重影。上图来自同一采集场景的另一段运动：两种模型重新初始化并使用相同预算训练，局部时间模型带来小幅改善，尚未达到清晰度要求。灰色是检视人物的合成背景，建筑还没有与人物联合重建。[时间表示](#时间基函数与局部运动)解释怎样增加运动自由度；[人物轮廓与背景变化](#人物轮廓与背景变化怎样区分)解释为什么墙面阴影可能进入错误监督。'
if old in s:s=s.replace(old,new,1)
s=s.replace('令 $\\tau=(t-750)/6$，$t$ 为连续帧索引。对每个节点的旋转向量和平移，在原多项式上增加','令 $\\tau=(t-t_0)/\\Delta t$，$t$ 为连续帧索引，$t_0$ 为参考时刻，$\\Delta t$ 为时间归一化尺度，二者以帧为单位；本地对照采用 $\\Delta t=6$。对每个节点的旋转向量和平移，在原多项式上增加',1)
old='这次相同15000次更新的对照中，原机位留出时刻平均前景PSNR提高0.251dB，另外三个机位提高0.073dB。部分衣物和手持物更集中，但图中的抬脚、手指与交叉遮挡仍偏软。时间自由度的增加没有提供新的可见表面、手指关节或正确的遮挡对应，因此不能仅凭平均误差下降就认定细节已恢复。[局部时间表示的原图对照与条件](../experiments/37_dynamic_human_reconstruction.md#局部时间基函数与人物细节)保留了改进与失败区域。'
new='在两个互不重叠的运动片段中，相同训练预算下的局部时间模型均降低了平均图像误差；这支持保留该表示继续研究，但范围仍是同一场景。下面的张手位置说明了平均误差的局限：衣物主体已经成形，手掌和玩偶边缘仍出现混合的半透明结构。\n\n![张开的手掌与手持物仍有重影](../outputs/dynamic-human-enerf-second-interval-079/comparison/camera04-frame000731-both_hands_and_objects.png)\n> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 机位04、留出帧731 · 原照片、全局三次时间模型、局部时间模型。\n\n时间基函数增加的是已有运动场的表达能力，没有直接增加独立手指关节、正确的表面对应或新出现的可见表面。此例中，后续训练帧的原照片与监督目标仍保留了清楚的手掌，因此还需要检查局部几何、姿态和可见性，不能把失败全部归于输入分辨率。[另一运动片段的真实对照](../experiments/37_dynamic_human_reconstruction.md#另一运动片段中的时间表示)保留了条件、退化机位与手鞋细节。'
if old in s:s=s.replace(old,new,1)
marker='因此，人物训练需要同时检查轮廓监督、原图纹理、相机投影和动态表面的对应关系。'
addition='''![原始训练照片、作者提供的前景透明度与实际训练目标](../outputs/dynamic-human-enerf-matte-diagnosis-080/camera07-frame000726-training-target.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap · 机位07、训练帧726 · 实际输入对照，不是重建图。

同一问题也出现在另一运动片段中。中列白色代表被当作前景的区域：绿色玩偶上方的墙面阴影、手臂下方的墙面和腿间背景被错误保留，右列训练目标因而包含墙面纹理。这直接证明了监督中的背景污染；它对最终残影的影响大小，仍需保持其他条件不变的训练对照。按此前景掩码计算的分数也会受到这些像素影响，不能称为精确的人体语义区域分数。[实际训练目标与张手细节](../experiments/37_dynamic_human_reconstruction.md#训练掩码中的墙面阴影与张手细节)给出输入证据。

'''
if M.name+'/camera07-frame000726-training-target.png' not in s:
 assert marker in s;s=s.replace(marker,addition+marker,1)
p.write_text(s,encoding='utf8')

p=L/'experiments/37_dynamic_human_reconstruction.md';s=p.read_text(encoding='utf8')
if not (R/'evidence-before.md').exists():shutil.copy2(p,R/'evidence-before.md')
s=s.replace('当前结果来自单一随机种子和反复使用的开发片段；下一步先在独立运动片段中保持参数不变做对照，再判断是否替换基线。','这组结果来自单一随机种子和反复使用的开发片段；下文补充了同一场景另一互不重叠运动片段的重新训练对照，尚不构成独立场景验证。',1)
addition='''

### 另一运动片段中的时间表示

在同一 actor5_6 采集中，将帧720至732作为第二段运动，参考时刻为726，与原744至756片段互不重叠。重新估计该片段的人体与头部运动、生成参考几何并训练外观，保留先前的相机时间偏移。两支共享105张偶数帧训练照片、57678个初始高斯、384个空间节点、随机种子、图像抽样、损失与增密日程，各完成15000次更新；唯一表示差别为是否加入局部时间B样条。没有将旧750时刻的身体运动外推到本片段，也没有复用旧外观模型。

原生1920×1080评测共120个相机时刻，每支各渲染一次，共240张。包括90张训练机位的奇数帧、18张另三个机位的奇数帧，以及12张偶数帧诊断图。两支按原始透明度定义的前景PSNR，在前两组分别为24.257→24.637dB、22.491→22.699dB；分别80/90、15/18张改善。机位02的六张留出帧平均下降0.040dB，机位06帧725下降0.224dB。新机位沿用此前另一片段估计的相机时间偏移，这不是完全未知相机条件。

![第二片段的完整人物对照](../outputs/dynamic-human-enerf-second-interval-079/comparison/camera04-frame000731-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 机位04、留出帧731 · 原照片、全局三次时间模型、局部时间模型。

全身、衣服图案和腿部已有明确结构，新增时间项的变化仍然有限。已逐一查看24张预定全身对照、24张手持物或鞋部对照，以及4张额外的最差变化图，共52张。张开的手掌有半透明重影，玩偶边缘破碎，抬起鞋子的边缘和鞋带仍软；人物之间及鞋边有墙面、地面残片。下面保留失败区域，不把平均分数改善视为清晰度验收。

![张手与玩偶局部仍不清晰](../outputs/dynamic-human-enerf-second-interval-079/comparison/camera04-frame000731-both_hands_and_objects.png)
> 图源：同上 · 机位04、帧731 · 相同原生像素范围，无增强。

![鞋底与鞋带仍偏软](../outputs/dynamic-human-enerf-second-interval-079/comparison/camera04-frame000731-both_person_feet.png)
> 图源：同上 · 机位04、帧731。

两组机位的相邻时间差分残差分别由0.03975降至0.03851、0.04541降至0.04434；它包含真实运动、遮挡和掩码误差，不能视作单独的闪烁真值。本轮没有进行视频视觉验收。最终全局模型255371个高斯、局部模型252984个高斯，训练约508秒和543秒，框架记录的显存分配峰值约1.72GB。240张渲染的指标与960张保存PNG均已从实际数组复核。

两个片段的平均改进支持继续保留局部时间表示，但仍是同场景、单一随机种子的开发验证。**张手、鞋部和遮挡尚未达到清晰要求，长序列、独立场景与建筑背景联合重建仍未完成。**

报告：[配对指标](../outputs/dynamic-human-enerf-second-interval-079/comparison/summary.json)、[实际看图范围](../outputs/dynamic-human-enerf-second-interval-079/visual-review.json)、[完整结论](../outputs/dynamic-human-enerf-second-interval-079/final-summary.json)。

### 训练掩码中的墙面阴影与张手细节

检查第二片段机位04、07、13的726和732两帧，共六张实际训练图。原照片、提供的前景透明度与按该透明度生成的训练目标逐像素对应。机位07的玩偶上方墙面阴影被保留；机位13帧732出现大块墙面误前景；机位04的墙面碎片与脚边地面也进入监督。这是在新片段中再次确认此前的掩码问题，不是已经完成修复。

![墙面阴影进入训练目标](../outputs/dynamic-human-enerf-matte-diagnosis-080/camera07-frame000726-training-target.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap · 机位07、训练帧726 · 原照片、前景透明度、灰底训练目标。

![遮挡机位的大块墙面误前景](../outputs/dynamic-human-enerf-matte-diagnosis-080/camera13-frame000732-training-target.png)
> 图源：同上 · 机位13、训练帧732。

手部不能用同一个原因全部解释。机位04帧732的监督目标仍保留清楚的张开手掌与弯曲手指，而相邻留出帧的重建有明显重影。这里还需要局部姿态、表面覆盖和跨视角对应的约束。

![原始训练目标保留了张手结构](../outputs/dynamic-human-enerf-matte-diagnosis-080/camera04-frame000732-hands-training-target.png)
> 图源：同上 · 机位04、训练帧732 · 原照片、前景透明度、实际训练目标；三列都不是重建结果。

下一项对照应只修正训练侧的可靠前景支持，同时保留人物、手持物和脚部，不把未知或被遮挡表面强行标成背景。原评测指标继续保留，另用固定的原照片细节区域检查，避免更换测试掩码后把分数变化混作重建提升。此处只完成输入诊断，没有运行新的修正训练。[诊断范围与结论](../outputs/dynamic-human-enerf-matte-diagnosis-080/summary.json)。
'''
if '### 另一运动片段中的时间表示' not in s:s+=addition
p.write_text(s,encoding='utf8')
resources={R.name:['final-summary.json','comparison/summary.json','visual-review.json','comparison/camera04-frame000731-full-person.png','comparison/camera04-frame000731-both_hands_and_objects.png','comparison/camera04-frame000731-both_person_feet.png'],M.name:['summary.json','camera07-frame000726-training-target.png','camera13-frame000732-training-target.png','camera04-frame000732-hands-training-target.png']}
p=S/'scripts/sync-gs-lab.mjs';s=p.read_text(encoding='utf8')
for root,files in resources.items():
 if root+'/${name}' not in s:
  lines=s.splitlines(keepends=True);i=next(i for i,l in enumerate(lines) if 'dynamic-human-enerf-temporal-basis-078/${name}' in l)+1
  lines.insert(i,'  ...'+json.dumps(files)+f".map(name => [`outputs/{root}/${{name}}`, '动态人物时间验证与监督诊断']),\n");s=''.join(lines)
 for f in files:assert (L/'outputs'/root/f).exists()
p.write_text(s,encoding='utf8')
(R/'published-resources.json').write_text(json.dumps(dict(paths=resources,licensed_models_exported=False),indent=2)+'\n')
print('Updated existing knowledge/evidence pages and ten explicit public resources.')
