from pathlib import Path
import json
LAB=Path(__file__).resolve().parents[2];SITE=Path('D:/Projects/EngineLab/engine-home')
p=LAB/'experiments/37_dynamic_human_reconstruction.md';t=p.read_text(encoding='utf8')
addition='''

### 训练照片中的人体关节与粗略人体先验

对第 750 帧的 15 台训练相机，使用 Keypoint R-CNN 提取 12 个躯干和四肢关节，以先前的人物实例掩码提出人物对应；未读取机位 00、06、12 的照片。15 张检测叠加图均已查看：被遮挡的黑衣人物存在漏检，机位 13 甚至把露出的头部当作完整人体，其低关节分数被排除。其他遮挡处仍有左右关节和人物混淆，检测分数不是真实可见性的证明。

用成对三角化和鲁棒重投影拟合检查各关节，仅保留至少四台相机支持、重投影残差小于 8 像素且最大观测夹角超过 2° 的候选。两人共 24 个身体关节中保留 23 个，黑衣人物左肘因缺乏支持保持缺失。220 个支持观测的重投影误差中位数为 3.197 像素，已独立重算；五个机位的三角化投影已查看。这些是预测得到的关节假设，不能充当关节真值或人体表面真值。

据此完成 2,000 次 CPU 优化，拟合两个人的 SMPL 身体姿态、10 个体形参数、整体旋转、平移和尺度。关节正向运动学与完整 SMPL 前向输出一致，输出参数与误差已核验。15 张投影图中已查看机位 01、04、07、08、17：躯干和腿部中心大体对应，但黑衣人物头部过直，缺失左肘导致手臂弯曲错误，手掌与脚的朝向仍不准确。该先验没有衣物、头发和玩具，尚未作为硬几何约束加入高斯训练。

![原训练照片与拟合的粗略 SMPL 人体投影；没有进行外观训练](../outputs/dynamic-human-enerf-body-prior-058/camera04-body-prior.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap，用户授权下载的 SMPL 中性模型与本站拟合 · 人体先验投影，不是颜色重建效果。

见[关节观测结论](../outputs/dynamic-human-enerf-body-observations-057/final-summary.json)、[关节几何重算](../outputs/dynamic-human-enerf-body-observations-057/independent-audit.json)、[观测审阅](../outputs/dynamic-human-enerf-body-observations-057/visual-review.json)、[人体先验结论](../outputs/dynamic-human-enerf-body-prior-058/final-summary.json)、[拟合核验](../outputs/dynamic-human-enerf-body-prior-058/independent-audit.json)和[投影审阅](../outputs/dynamic-human-enerf-body-prior-058/visual-review.json)。SMPL 模板、模型参数和网格均留在本地，不作为网站下载资源。
'''
if '### 训练照片中的人体关节与粗略人体先验' not in t:p.write_text(t+addition,encoding='utf8')
p=SITE/'scripts/sync-gs-lab.mjs';t=p.read_text(encoding='utf8')
if 'dynamic-human-enerf-body-observations-057/${name}' not in t:
 lines=t.splitlines(keepends=True);idx=next(i for i,l in enumerate(lines) if 'dynamic-human-enerf-surface-gs-055/${name}' in l)+1
 lines.insert(idx,"  ...['final-summary.json','independent-audit.json','visual-review.json'].map(name => [`outputs/dynamic-human-enerf-body-observations-057/${name}`, '训练相机人体关节检查']),\n  ...['final-summary.json','independent-audit.json','visual-review.json','camera04-body-prior.png'].map(name => [`outputs/dynamic-human-enerf-body-prior-058/${name}`, '粗略人体先验投影检查']),\n")
 p.write_text(''.join(lines),encoding='utf8')
print('057058 evidence source prepared')
