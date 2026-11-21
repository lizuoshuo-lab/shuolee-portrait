from pathlib import Path
LAB=Path(__file__).resolve().parents[2]
page=LAB/'experiments/37_dynamic_human_reconstruction.md';text=page.read_text(encoding='utf8')
addition='''

### 输入修复后，真实渲染改善了哪里

相同 1,000 次更新预算的训练与六个原生视角渲染已完成。与原输入流程相比，三个留出视角的人物前景 PSNR 分别变化 −0.27、+0.10、+0.94 dB；完整有效图像则分别提高 0.46、0.76、1.27 dB。前景指标仍使用有误差的作者透明度，因此同时查看原照片与真实渲染。

![同一原照片、原掩码训练和修订掩码训练的真实渲染](../outputs/dynamic-human-4k4d-mask-control-039/paired-comparison/camera12-frame000750-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站运行 · 机位 12，第 750 帧 · 相同 1,000 次更新，原生像素，无图像增强。

修订输入后，部分前臂连接与鞋边有所改善，但手指、衣物图案、手持物和交叉肢体仍模糊，头肩轮廓仍有黑色点状伪影。六组完整人物对照、一个前臂对照和三张完整留出渲染均已查看。**这是局部恢复，仍未达到清晰人体重建的要求。** 此处的右列是真实模型渲染，与上一节的分割输入合成有明确区别。

模型、优化器与 1,000 次更新状态，以及全部原生图像和指标，均已单独核对。见[结果结论](../outputs/dynamic-human-4k4d-mask-control-039/final-summary.json)、[看图范围](../outputs/dynamic-human-4k4d-mask-control-039/visual-review.json)、[模型状态核验](../outputs/dynamic-human-4k4d-mask-control-039/independent-state-audit.json)、[图像核验](../outputs/dynamic-human-4k4d-mask-control-039/native-evaluation/independent-image-audit.json)。
'''
if '### 输入修复后，真实渲染改善了哪里' not in text:page.write_text(text+addition,encoding='utf8')
knowledge=LAB/'knowledge/32_dynamic_human_scene_reconstruction.md';text=knowledge.read_text(encoding='utf8')
text=text.replace('当前候选仅用于有界训练对照，尚未证明人体重建变清楚；见', '相同预算的训练对照显示部分前臂与轮廓有所恢复，但手指、衣物与遮挡仍未清晰；见')
knowledge.write_text(text,encoding='utf8')
sync=Path('D:/Projects/EngineLab/engine-home/scripts/sync-gs-lab.mjs');text=sync.read_text(encoding='utf8')
anchor="  ['outputs/dynamic-human-4k4d-mask-control-039/protocol.json', '原生人体输入掩码对照条件'],"
if 'dynamic-human-4k4d-mask-control-039/${name}' not in text:
    assert anchor in text
    text=text.replace(anchor,anchor+"\n  ...['final-summary.json', 'visual-review.json', 'independent-state-audit.json', 'native-evaluation/independent-image-audit.json', 'paired-comparison/camera12-frame000750-full-person.png'].map(name => [`outputs/dynamic-human-4k4d-mask-control-039/${name}`, '输入修复后的真实渲染，清晰度未通过']),")
    sync.write_text(text,encoding='utf8')
print('039 result prose and five resources prepared')
