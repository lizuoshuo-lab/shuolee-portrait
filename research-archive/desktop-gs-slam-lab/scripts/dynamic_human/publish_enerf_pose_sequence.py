from pathlib import Path
import json,shutil
LAB=Path(__file__).resolve().parents[2];SITE=Path('D:/Projects/EngineLab/engine-home');OUT=LAB/'outputs/dynamic-human-enerf-pose-sequence-062'
assert json.loads((OUT/'independent-audit.json').read_text())['native_images']==195
p=LAB/'experiments/37_dynamic_human_reconstruction.md';text=p.read_text(encoding='utf8')
addition='''

### 连续人体关键点中的跨相机时间偏移

前臂约束对照未解决拖影后，改用第 744～756 帧的连续人体观测检查时序。15 台训练相机共有 195 张原生照片、390 次分人全身关键点推理；预先选定七个偶数时刻用于拟合，六个奇数时刻用于检查。第 750 帧的全部 15 张输入及关键点结果与上一轮逐项一致。首轮输入因去畸变的浮点精度约定不同触发断言并停止，随后恢复原训练加载方式，通过全部输入一致性检查，未改动原训练数据。

对身体和脚部的 36 条三维点轨迹分别采用三次多项式，联合估计相机时间偏移；相机 01 定义零偏移，其他偏移限制在正负两帧内。对应支持只来自第 750 帧的鲁棒几何检查，遮挡错误仍由鲁棒损失处理。保持相机几何不变，对比零偏移拟合与允许偏移的拟合，后者没有参数触及偏移边界。

在预留的 2,262 个关节点观测中，重投影误差中位数从 3.228 降为 2.636 像素，90% 分位数从 9.398 降为 8.235 像素；均方根误差仅从 18.461 降到 18.176 像素，表明遮挡或错误对应导致的大残差仍然存在，不能只展示中位数。

随后做跨人物检查：只由黑衣人物估计时间偏移，将这些偏移固定，再拟合白衣人物的训练时刻轨迹；也反向进行同样检查。在另一人的预留时刻，黑衣人物中位误差从 3.114 降到 2.542 像素，白衣人物从 3.297 降到 2.902 像素。两人独立估计的偏移差异均方根为 0.361 帧，部分相机差异较大。这支持把残余时序差异作为下一轮图像训练的候选因素，尚不足以认证真实时钟偏移。

这一轮没有按估计偏移重新取图，也没有据此宣称人物清晰度改善。尤其机位 00、06、12 未参与估计，它们相对相机 01 的拍摄时刻仍未知。下一步应在已获得时间关系的训练相机中，比较连续人体模型对原生预留时刻图像的重建，保持原始照片不做锐化或插帧。

见[连续片段与结论](../outputs/dynamic-human-enerf-pose-sequence-062/final-summary.json)、[时序拟合条件](../outputs/dynamic-human-enerf-pose-sequence-062/timing-protocol.json)、[逐相机结果](../outputs/dynamic-human-enerf-pose-sequence-062/timing-summary.json)、[跨人物检查](../outputs/dynamic-human-enerf-pose-sequence-062/person-transfer-summary.json)、[输入和全部误差复算](../outputs/dynamic-human-enerf-pose-sequence-062/independent-audit.json)、[实际看图范围](../outputs/dynamic-human-enerf-pose-sequence-062/visual-review.json)及[输入约定修复](../outputs/dynamic-human-enerf-pose-sequence-062/input-recovery.json)。
'''
if '### 连续人体关键点中的跨相机时间偏移' not in text:p.write_text(text+addition,encoding='utf8')
p=LAB/'knowledge/32_dynamic_human_scene_reconstruction.md';text=p.read_text(encoding='utf8')
addition='\n人物对应还能提供另一种检查：如果同一组相机存在时间偏移，从一个人物估计的偏移应当有助于解释另一个人物的运动。本地连续片段采用一组时刻拟合、另一组时刻检查，跨人物转移的偏移降低了两个人在预留时刻的关节点误差；但大残差仍在，原生 RGB 的清晰度尚未验证。这比只看同一批拟合残差更有说服力，也仍不能直接等同于时钟标定成功。见[连续人体与跨人物检查](../experiments/37_dynamic_human_reconstruction.md#连续人体关键点中的跨相机时间偏移)。\n\n'
if '从一个人物估计的偏移应当有助于解释另一个人物的运动' not in text:
 anchor='### 连续运动中的同预算对照';assert anchor in text;text=text.replace(anchor,addition+anchor);p.write_text(text,encoding='utf8')
p=SITE/'scripts/sync-gs-lab.mjs';text=p.read_text(encoding='utf8')
if 'dynamic-human-enerf-pose-sequence-062/${name}' not in text:
 lines=text.splitlines(keepends=True);idx=next(i for i,l in enumerate(lines) if 'dynamic-human-enerf-forearm-control-061/${name}' in l)+1
 lines.insert(idx,"  ...['final-summary.json','timing-protocol.json','timing-summary.json','person-transfer-summary.json','independent-audit.json','visual-review.json','input-recovery.json'].map(name => [`outputs/dynamic-human-enerf-pose-sequence-062/${name}`, '连续人体时序检查']),\n");p.write_text(''.join(lines),encoding='utf8')
shutil.copy2(__file__,OUT/Path(__file__).name);print('062 bounded findings prepared for website')
