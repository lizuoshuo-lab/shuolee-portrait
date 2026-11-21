from pathlib import Path
import json,shutil
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-enerf-support-correction-081';SITE=Path('D:/Projects/EngineLab/engine-home')
read=lambda p:json.loads(p.read_text(encoding='utf8'))
s=read(O/'final-summary.json');assert s['actual_panels_reviewed']==36 and not s['adopt_as_development_baseline']
p=L/'knowledge/32_dynamic_human_scene_reconstruction.md';txt=p.read_text(encoding='utf8');shutil.copy2(p,O/'knowledge-before.md')
marker='因此，人物训练需要同时检查轮廓监督、原图纹理、相机投影和动态表面的对应关系。'
addition='''前景更干净，并不自动意味着手指更清楚。在相同运动模型、初始化和训练预算下，只去掉部分训练图中确认的背景误前景，确实减弱了一些墙面残片，但下图的张手仍然混有半透明重影，玩偶边缘也没有稳定恢复。掩码主要决定哪些像素参与前景颜色与轮廓监督；它不会直接补充手指关节的运动或被遮挡表面的对应关系。这个有限对照支持继续处理局部运动与可见性，不能说明正确掩码不重要。

![原照片与仅改变训练前景支持后的手部重建对照](../outputs/dynamic-human-enerf-support-correction-081/comparison/camera04-frame000731-both_hands_and_objects.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 机位04、留出帧731 · 原照片、原训练支持、保守修正支持 · 原生像素。

这次修正没有替换原开发基线。训练输入中不确定的遮挡人物、手持物与脚部受到保护，仍有背景残留；[相同预算的实际对照](../experiments/37_dynamic_human_reconstruction.md#前景支持修正与手部清晰度)说明了它的有效范围。

'''
assert marker in txt and O.name+'/comparison/camera04-frame000731-both_hands_and_objects.png' not in txt
txt=txt.replace(marker,addition+marker,1)
txt=txt.replace('它对最终残影的影响大小，仍需保持其他条件不变的训练对照。','下方的相同预算训练对照进一步检验了部分修正能否改善最终重建。',1)
txt=txt.replace('灰色是检视人物的合成背景，建筑还没有与人物联合重建。','随后完成的[前景支持修正](#人物轮廓与背景变化怎样区分)减少了部分背景残片，尚未改善张手细节。灰色是检视人物的合成背景，建筑还没有与人物联合重建。',1)
p.write_text(txt,encoding='utf8')
p=L/'experiments/37_dynamic_human_reconstruction.md';txt=p.read_text(encoding='utf8');shutil.copy2(p,O/'evidence-before.md')
old='下一项对照应只修正训练侧的可靠前景支持，同时保留人物、手持物和脚部，不把未知或被遮挡表面强行标成背景。原评测指标继续保留，另用固定的原照片细节区域检查，避免更换测试掩码后把分数变化混作重建提升。此处只完成输入诊断，没有运行新的修正训练。'
new='上述工作只完成输入诊断；下节补充实际的训练侧前景支持对照。评测保留原图、原掩码和既定细节范围，避免更换测试口径。'
assert old in txt;txt=txt.replace(old,new,1)
txt+='''

### 前景支持修正与手部清晰度

相同运动片段720至732中，先对全部105张训练照片生成分割候选，再查看15个机位各7帧。直接使用候选会删掉部分玩偶，以及遮挡机位中的后方人物，因此没有采用这些直接预测。最终仅在9个机位的63张图中保守裁去部分误前景，手、玩偶、头颈和脚部附近保留原监督；另6个机位的42张图完全保留原掩码。去掉的原前景像素累计403794个，不等于获得了完整的人体语义真值。

局部时间B样条模型从相同57678个初始高斯重新训练15000次。身体运动、相机、图像裁剪、随机种子、抽样、随机背景、损失和增密设置均保持一致；唯一输入差别是上述训练侧透明度。约543秒完成训练，最终252373个高斯，框架显存分配峰值约1.72GB。评测使用原来的120个机位时刻，原生1920×1080，未改动测试掩码或进行图像增强。

![保守修正前后的完整人物重建](../outputs/dynamic-human-enerf-support-correction-081/comparison/camera04-frame000731-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 机位04、留出帧731 · 原照片、原训练支持、保守修正支持。

原机位90张留出帧的前景PSNR为24.637→24.643dB，48张改善、42张退化；另外3个机位18张为22.699→22.661dB，6张改善、12张退化。既定12个手与玩偶窗口平均变化−0.081dB，12个脚部窗口为+0.025dB。窗口包含周围区域，原掩码也包含部分背景，因此这些不是精确的人体部位语义分数。

![张开的手掌仍有重影](../outputs/dynamic-human-enerf-support-correction-081/comparison/camera04-frame000731-both_hands_and_objects.png)
> 图源：同上 · 相同原生像素范围。

![抬脚轮廓仍偏软](../outputs/dynamic-human-enerf-support-correction-081/comparison/camera04-frame000731-both_person_feet.png)
> 图源：同上 · 机位04、留出帧731。

逐一查看全部24个预定手物与脚部对照，以及12张全身对照，共36张，覆盖两组机位各自三个最差变化案例。部分墙面残片减弱，但张手、玩偶边缘和抬起的鞋没有出现稳定清晰度改善。机位03帧731下降0.376dB、机位09帧723下降0.297dB；这些退化保留在原记录中。时间差分残差在原机位基本不变，新机位略升，也不能支持时间稳定性提升。

**这轮候选不替换原开发基线。**结论只针对这次不完整的保守前景修正：它不足以解决当前手部模糊，不能据此否定完整且准确的分割。下一步将重点检查原生分辨率手部观测、局部运动自由度和跨视角对应，再引入通过核验的约束。

完整记录：[结论](../outputs/dynamic-human-enerf-support-correction-081/final-summary.json)、[固定口径配对指标](../outputs/dynamic-human-enerf-support-correction-081/comparison/summary.json)、[24个细节窗口](../outputs/dynamic-human-enerf-support-correction-081/comparison/detail-metrics.json)、[实际看图范围](../outputs/dynamic-human-enerf-support-correction-081/render-visual-review.json)。120组数组指标与960张保存PNG均已独立复核，原模型和原始数据完整保留。
'''
p.write_text(txt,encoding='utf8')
files=['final-summary.json','comparison/summary.json','comparison/detail-metrics.json','render-visual-review.json','comparison/camera04-frame000731-full-person.png','comparison/camera04-frame000731-both_hands_and_objects.png','comparison/camera04-frame000731-both_person_feet.png']
p=SITE/'scripts/sync-gs-lab.mjs';txt=p.read_text(encoding='utf8');assert O.name+'/${name}' not in txt;lines=txt.splitlines(keepends=True);i=next(i for i,l in enumerate(lines) if 'dynamic-human-enerf-second-interval-079/${name}' in l)+1;lines.insert(i,'  ...'+json.dumps(files)+f".map(name => [`outputs/{O.name}/${{name}}`, '动态人物前景支持与细节对照']),\n");p.write_text(''.join(lines),encoding='utf8')
(O/'published-resources.json').write_text(json.dumps(dict(paths={O.name:files},licensed_models_exported=False),indent=2)+'\n');print('Updated existing knowledge and evidence; seven explicit resources')
