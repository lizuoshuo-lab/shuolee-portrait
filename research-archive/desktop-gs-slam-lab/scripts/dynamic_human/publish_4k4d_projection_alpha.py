from pathlib import Path
LAB=Path(__file__).resolve().parents[2];page=LAB/'experiments/37_dynamic_human_reconstruction.md';text=page.read_text(encoding='utf8')
addition='''

### 背景配准改善没有直接转化为人体清晰度

从同一个修订掩码模型与 Adam 状态出发，原投影和静态背景偏移两组各再训练 1,000 次，累计均为 2,000 次。修正组仅在解码、裁剪以后改变投影矩阵中的像平面偏移，原照片、源机位、训练预算与初始化保持一致。全部 12 张原生视角渲染及对应指标已核验。

修正组在三个留出机位的前景 PSNR 分别降低 0.82、0.70、0.66 dB，六个检查视角都下降。六组人物对照与三张完整留出渲染中，手、衣服图案和遮挡关系没有恢复，部分肩部与鞋边伪影仍明显。因此保留原投影继续诊断，不采用这组偏移。这只否定当前父模型与预算下的经验偏移处理，不能推广为“相机标定和时间同步不重要”。见[成对结果](../outputs/dynamic-human-4k4d-projection-control-040/final-summary.json)、[实际看图结论](../outputs/dynamic-human-4k4d-projection-control-040/visual-review.json)、[成对模型状态核验](../outputs/dynamic-human-4k4d-projection-control-040/independent-state-audit.json)。

### 合成公式不同，也要测量实际影响

对当前点渲染实现的核对发现，其透明度通道计算为 $\sum_i T_i a_i o_i$，其中 $o_i$ 是点不透明度，$a_i$ 已包含不透明度与径向权重；标准累计透明度则为 $\sum_i T_i a_i=1-\prod_i(1-a_i)$。这段写法在官方 PyTorch3D 与可微 OpenGL 路径中均存在，不能归因于本机兼容路径。

小例子能清楚地区分两个公式，但实际模型的点不透明度均已高于 0.9978。使用同一批真实光栅化片元重算，六个原生视角的最大透明度变化约为 0.00206，前景 PSNR 变化均不足 0.0007 dB。三个留出视角的人物对照已查看，手、衣物与遮挡仍模糊。**公式差异属实，但它不足以解释当前明显的清晰度问题。** 这是固定模型的合成诊断，尚未声称从头训练时两种公式效果相同。见[公式与实测结论](../outputs/dynamic-human-4k4d-alpha-composition-041/final-summary.json)、[独立片元与像素核验](../outputs/dynamic-human-4k4d-alpha-composition-041/independent-audit.json)。
'''
if '### 背景配准改善没有直接转化为人体清晰度' not in text:page.write_text(text+addition,encoding='utf8')
sync=Path('D:/Projects/EngineLab/engine-home/scripts/sync-gs-lab.mjs');text=sync.read_text(encoding='utf8')
if 'dynamic-human-4k4d-projection-control-040/${name}' not in text:
    lines=text.splitlines(keepends=True);idx=next(i for i,l in enumerate(lines) if 'dynamic-human-4k4d-mask-control-039/${name}' in l)+1
    lines.insert(idx,"  ...['final-summary.json', 'visual-review.json', 'independent-state-audit.json'].map(name => [`outputs/dynamic-human-4k4d-projection-control-040/${name}`, '经验投影偏移的成对训练结果；未采用']),\n  ...['final-summary.json', 'independent-audit.json'].map(name => [`outputs/dynamic-human-4k4d-alpha-composition-041/${name}`, '真实片元透明度诊断；非清晰度改善']),\n")
    sync.write_text(''.join(lines),encoding='utf8')
print('040/041 evidence prepared')
