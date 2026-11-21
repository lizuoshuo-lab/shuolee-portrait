"""Replace the existing motion/clarity example; keep detailed evidence outside theory."""
from pathlib import Path
import json,shutil
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-enerf-hand-joint-084';SITE=Path('D:/Projects/EngineLab/engine-home')
read=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
s=read(O/'final-summary.json');assert s['actual_render_panels_reviewed']==40 and not s['adopt_as_development_baseline']
p=L/'knowledge/32_dynamic_human_scene_reconstruction.md';txt=p.read_text(encoding='utf8')
assert '手部局部联合优化与连续画面' not in txt
shutil.copy2(p,O/'knowledge-before.md')
a=txt.index('![原照片、仅图像续训与增加手部运动约束后的原生重建]')
b=txt.index('### 相机与人物必须对应同一个拍摄时刻',a)
txt=txt[:a]+r'''![张手时的原照片、原模型、局部联合优化及增加运动约束后的原生对照](../outputs/dynamic-human-enerf-hand-joint-084/comparison/camera04-frame000731-target_right_hand.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 机位04、留出帧731 · 四列依次为原照片、原模型、局部图像联合优化、再加入运动约束。

原照片中的弯曲手指与玩偶已经分离，三种重建仍出现皮肤与玩偶边缘混合的碎片。前一次只更新运动节点的对照，已经表明运动残差下降不保证图像更清晰。上图进一步放开局部高斯的位置、协方差、不透明度与颜色，并在每次更新时重算源图可见贡献，但阻断该贡献对运动损失的梯度。加入运动项后，训练观测的法向残差从约1.25降到0.73像素，留出图像和手物区域仍未稳定改善。这说明，在当前观测与表示条件下，局部联合优化仍不足以恢复手指。

还要区分**表面表示的密度**与**运动控制的空间分辨率**。前文[变形图](#骨架之外的局部变形如何进入渲染)中，每个高斯的运动由邻近节点及固定权重共同决定。增加高斯或改变颜色，并不自动增加可独立表达的局部运动；多个手指若继承相近的控制权重，仍可能被相似的变形牵动。对当前手部候选区域检查发现，原有控制节点在该区域内缺乏局部覆盖。这个现象提供了检验更细控制节点的依据，但不能单凭节点距离断言模糊的成因，也不能把新增节点数量当成效果提升。

清晰度最终要由原生图像中的指间分离、边缘和连续运动来判断。当前两种联合优化结果均未替换原模型；训练条件、完整人物、连续13帧视频及空间覆盖诊断见[手部局部联合优化与连续画面](../experiments/37_dynamic_human_reconstruction.md#手部局部联合优化与连续画面)。局部节点方案尚未训练，手指清晰度与完整场景重建仍未完成。

'''+txt[b:]
p.write_text(txt,encoding='utf8')
p=L/'experiments/37_dynamic_human_reconstruction.md';txt=p.read_text(encoding='utf8')
assert '### 手部局部联合优化与连续画面' not in txt;shutil.copy2(p,O/'evidence-before.md')
txt+=r'''

### 手部局部联合优化与连续画面

这次检验的是：同时允许手部几何、外观和运动调整，是否能修复张手时的碎片。仍从原15000步模型出发，两支各续训3000步，采用完全一致的105张训练照片、抽样顺序、背景、学习率与正则化。候选增加权重0.05、尺度2像素的法向运动项，仍使用前次筛出的1527个估计对应；它们主要覆盖手背与手腕，不能视为手指真值。所有留出划分与原始掩码保持不变。

手部范围由13个训练机位的728、730、732帧确定，需通过保守分割内部、正深度、与原模型渲染深度接近及至少两个机位支持的筛选。选出的6330个高斯可更新位置、尺度、旋转、不透明度和球谐颜色；关联的22个现有节点可更新运动，但更新只作用于这些高斯。其余246654个高斯参数和362个节点精确冻结，身体运动、相机、时间偏移保持不变。本轮没有新增或分裂高斯。该范围仍可能包含手物交界的混合结构，不是准确人体分割真值。

![完整人物的四列对照](../outputs/dynamic-human-enerf-hand-joint-084/comparison/camera04-frame000731-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 机位04、帧731 · 原照片、原模型、局部图像联合优化、再加入运动约束。

两支实际完成各3000步，耗时约158、163秒，框架显存分配峰值约1.75GB。120个机位时刻均按1920×1080真实渲染，再与原模型配对比较；360张预测及其720张颜色／透明度PNG已独立数值复核。90张原训练机位的留出时间图，前景PSNR为24.637、24.634、24.616dB；另外三个机位的18张图为22.699、22.688、22.679dB。仅图像联合优化与原模型接近，增加运动项后两组均略有退化。

![正面手掌在留出时刻仍然破碎](../outputs/dynamic-human-enerf-hand-joint-084/comparison/camera04-frame000731-target_right_hand.png)
> 图源：同上 · 保留160×160原生窗口，无锐化或补画。

12个原有手与物品窗口，相对原模型平均变化为−0.017、−0.074dB；12个脚部窗口数值完全不变。额外固定的6个目标手窗口变化为+0.003、−0.012dB，也不足以支持清晰度提升。这些指标使用预先确定的图像窗口与原掩码交集，并非独立标注的手部真值区域。训练观测的法向运动残差则从1.252降到0.731像素，说明约束拟合改善仍未转化成真实手指细节。

![同一固定窗口中的三个连续片段代表时刻](../outputs/dynamic-human-enerf-hand-joint-084/temporal/camera04-comparison.png)
> 图源：同上 · 从上到下为721、727、731帧 · 每列顺序与前图一致，窗口在整个片段保持不变。

另外完整渲染了04、13两个机位的720至732连续13帧，再裁出固定窗口：共78张模型局部预测，全部PNG与数组一致。已逐帧查看26个机位时刻，从抓握转向张手时，皮肤与玩偶碎片持续重叠；13机位的手背与玩偶接触处也没有稳定改善。04机位相邻帧图像差分误差为0.05922、0.05919、0.06006；13机位为0.04969、0.04972、0.05061，没有一致改善。差分误差比较的是预测与原照片的亮度变化，不等同于运动真值误差。

连续对照视频：[04机位](../outputs/dynamic-human-enerf-hand-joint-084/temporal/camera04-hand-continuity.mp4)、[13机位](../outputs/dynamic-human-enerf-hand-joint-084/temporal/camera13-hand-continuity.mp4)。均为13帧、5帧／秒的慢速检查版本；播放速度不是原始采集帧率。

![另一机位的张手与玩偶边缘](../outputs/dynamic-human-enerf-hand-joint-084/comparison/camera12-frame000731-both_hands_and_objects.png)
> 图源：同上 · 留出机位12、帧731。

实际查看范围共40张对照图／连续帧拼图，包含12个全身案例、12个手物案例、6个固定目标手窗口、8张覆盖全部连续帧的检查图及2张代表时刻拼图。全身案例覆盖两支模型在两组留出数据中的全部前三差案例。**两支候选均不替换原模型；手指清晰度仍未达标。**这是同一场景反复开发的验证，不能视为独立新场景验收。

下一项诊断针对运动控制的空间覆盖。选中高斯的规范坐标包围盒最大边长约0.222场景长度单位，22个原有关联节点均不在这个包围盒内；高斯到最近关联节点的距离中位数为0.106，约为最大边长的48%。从同一训练模型的选中高斯中心提出32个局部节点，可把最近距离中位数降到0.0215。**这仅是未训练的空间覆盖方案，不是手指关节标注，也没有证明图像会改善。**还需核对节点在原照片中的投影，验证零初始化不改变原渲染，再做相同条件的训练对照。高斯数量、控制节点数量和可信观测覆盖是不同问题，不能相互替代。

记录：[本轮结论](../outputs/dynamic-human-enerf-hand-joint-084/final-summary.json)、[配对评测](../outputs/dynamic-human-enerf-hand-joint-084/comparison/summary.json)、[细节窗口](../outputs/dynamic-human-enerf-hand-joint-084/comparison/detail-metrics.json)、[连续画面检查](../outputs/dynamic-human-enerf-hand-joint-084/temporal/summary.json)、[节点空间覆盖](../outputs/dynamic-human-enerf-hand-joint-084/spatial-diagnostic/summary.json)、[数值核验](../outputs/dynamic-human-enerf-hand-joint-084/numerical-audit.json)、[实际看图记录](../outputs/dynamic-human-enerf-hand-joint-084/render-visual-review.json)。原始数据与所有旧模型保留；静态背景联合重建和独立场景验证仍未完成。
'''
p.write_text(txt,encoding='utf8')
files=['final-summary.json','comparison/summary.json','comparison/detail-metrics.json','temporal/summary.json','spatial-diagnostic/summary.json','numerical-audit.json','render-visual-review.json','comparison/camera04-frame000731-full-person.png','comparison/camera04-frame000731-target_right_hand.png','comparison/camera12-frame000731-both_hands_and_objects.png','temporal/camera04-comparison.png','temporal/camera04-hand-continuity.mp4','temporal/camera13-hand-continuity.mp4']
p=SITE/'scripts/sync-gs-lab.mjs';txt=p.read_text(encoding='utf8');assert O.name+'/$'+'{name}' not in txt
lines=txt.splitlines(keepends=True);i=next(i for i,l in enumerate(lines) if 'dynamic-human-enerf-open-hand-flow-083/$'+'{name}' in l)+1
lines.insert(i,'  ...'+json.dumps(files)+".map(name => [\x60outputs/"+O.name+"/$"+"{name}\x60, '手部联合优化与连续画面']),\n")
p.write_text(''.join(lines),encoding='utf8')
save(O/'published-resources.json',dict(paths={O.name:files},licensed_models_exported=False))
shutil.copy2(__file__,O/Path(__file__).name);print('Updated existing theory example; appended bounded evidence;13resources allowlisted')

