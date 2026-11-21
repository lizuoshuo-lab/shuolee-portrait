from pathlib import Path
LAB=Path(__file__).resolve().parents[2]
knowledge=LAB/'knowledge/32_dynamic_human_scene_reconstruction.md'
section=r'''
### 人物轮廓与背景变化怎样区分

背景减除得到的前景透明度不等于人体分割。人遮住墙面是一种可见性变化；人的影子落在墙上，是同一静态表面的照明变化。两者都可能使当前图像偏离空场景参照，但只有前者需要在墙前放置人体几何。手持物也属于运动前景，不能仅凭是否落在人形范围内就删除。

![原图中的投影阴影与前景透明度中的墙面残留](../outputs/dynamic-human-enerf-matte-background-034/camera08-frame000750-wall-matte-comparison.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap · 原图、空场景参照、作者前景透明度及灰底合成；同一原生像素区域。

从左到右看，原图中人的影子真实地改变了墙面亮度；空场景参照没有这片阴影；作者提供的透明度却还保留了一块墙面纹理，灰底合成后出现人物之外的半透明残片。这类残片不能直接解释成衣物或肢体。

在像素已经几何对齐、颜色以线性 RGB 表示的理想合成模型中，令 $\mathbf C_t(\mathbf u)$ 为运动前景颜色，$\alpha_t(\mathbf u)\in[0,1]$ 为覆盖比例，$\mathbf B_t(\mathbf u)$ 为当前光照下的背景颜色，$\mathbf B_0(\mathbf u)$ 为空场景参照，则

$$
\mathbf I_t-\mathbf B_0
=\alpha_t(\mathbf C_t-\mathbf B_0)
+(1-\alpha_t)(\mathbf B_t-\mathbf B_0).
$$

右侧第二项在没有人体覆盖的位置仍可能非零：阴影、照明变化和曝光差异都能贡献背景残差。若背景图还存在像素位移 $\Delta\mathbf u$，其额外一阶误差近似为 $\nabla\mathbf B_0\,\Delta\mathbf u$；其中图像梯度单位为颜色值每像素，位移单位为像素。墙缝和衣物边缘的梯度较大，几像素错位就会产生明显条纹。这个近似只描述局部小位移，不能用墙面的二维变换代替人体三维几何。

因此，人物训练需要同时检查轮廓监督、原图纹理、相机投影和动态表面的对应关系。提高不透明度只能遮住背景残差，不能保证新生成的是正确人体。实际输入中的阴影和错位示例见[背景与轮廓诊断](../experiments/37_dynamic_human_reconstruction.md#人物透明度中的墙面残留与静态投影检查)。

'''
text=knowledge.read_text(encoding='utf8')
if '### 人物轮廓与背景变化怎样区分' not in text:
    text=text.replace('### 同一帧号不一定对应同一运动状态',section+'### 同一帧号不一定对应同一运动状态')
    knowledge.write_text(text,encoding='utf8')
evidence=LAB/'experiments/37_dynamic_human_reconstruction.md'
extra=r'''

### 背景拟合后的实际纹理检查

actor5_6 的 6,480 个训练视角背景点完成了累计 6,000 次优化。原运行在第 2,401 次更新后超过预设 50 万点数量限制，停止时为 532,803 个高斯；随后保留优化器状态、固定点数继续 3,599 次更新。这是有密度上限恢复的结果，不能称为未经修改的默认密度控制复现。没有发生显存不足，训练记录的 PyTorch 分配峰值约 1.71 GB，不包含其他程序和全部驱动占用。

![背景纹理在训练后仍然存在拉丝](../outputs/dynamic-human-enerf-background-fit-033/density-capped/evaluation/camera06-wall-comparison.png)
> 图源：ENeRF-Outdoor 与本站渲染 · 留出相机 06 · 原图、稀疏初始点、最终背景；原生裁图。

三个留出视角 00、06、12 的有效像素 PSNR 分别由 14.79、15.10、15.24 dB 提高到 22.02、21.36、21.50 dB。完整图和局部图仍显示墙面拉丝、边缘区域重建不足及地面纹理缺失，抽查的训练视角也有同类问题。这个背景没有通过质量检查，尚不能作为可靠的固定背景用于人物联合训练。

独立复算覆盖 15 个训练缓存、12 张 RGB、12 张 alpha 及其指标；实际查看 3 张完整留出图、6 张墙面比较图和 3 张底部比较图。[完整机位 06 输出](../outputs/dynamic-human-enerf-background-fit-033/density-capped/evaluation/camera06-trained.png)、[状态与图像复核](../outputs/dynamic-human-enerf-background-fit-033/density-capped/independent-audit.json)、[检查范围](../outputs/dynamic-human-enerf-background-fit-033/density-capped/visual-review.json)、[阶段结论](../outputs/dynamic-human-enerf-background-fit-033/final-summary.json)。

### 人物透明度中的墙面残留与静态投影检查

在已选的行走片段中，仅使用训练相机 08 的第 740、750、760 帧检查源图。原图包含真实投影阴影，作者前景透明度同时保留了人体以外的部分墙面纹理。这说明背景减除结果不能直接当作语义人体真值，不能把阴影变化全部交给人体几何拟合。

![原始阴影与前景透明度残留的对应](../outputs/dynamic-human-enerf-matte-background-034/camera08-frame000750-wall-matte-comparison.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap · 训练相机 08，第 750 帧；没有清理或补绘原图。

低前景透明度墙面区域的特征对应另按 80% 拟合、20% 检查划分。原空场景参照与当前图像的检查点位移中位数约 2.67～2.88 像素，拟合二维单应性后为 0.21～0.24 像素。这里划分的是同一对图中的对应点，不是新照片测试；墙面单应性也不适用于墙前运动的人体。

再用训练视角三角化的三维点估计 PnP 候选位姿，墙面检查点重投影中位数由 2.42～2.70 像素降到 0.44～0.48 像素。没有参与拟合的底部区域仅找到每帧 10 或 11 个对应点，其中 9 或 10 个改善，中位误差降到 0.36～0.53 像素。点数少且主体几何接近平面，只能支持继续检查标定，不能宣称全序列标定已修复，或把此前另一演员序列的手部模糊都归因于此。

源图、前景透明度及正式相机均未改变；二维配准图仅作诊断，不用作模型效果。实际查看三组原生源图／透明度比较和三组静态配准比较。[二维配准结果](../outputs/dynamic-human-enerf-matte-background-034/static-registration.json)、[PnP 候选](../outputs/dynamic-human-enerf-matte-background-034/static-pose-diagnostic.json)、[独立底部对应检查](../outputs/dynamic-human-enerf-matte-background-034/pose-floor-check.json)、[观察范围](../outputs/dynamic-human-enerf-matte-background-034/visual-review.json)、[阶段结论](../outputs/dynamic-human-enerf-matte-background-034/final-summary.json)。
'''
text=evidence.read_text(encoding='utf8')
if '### 背景拟合后的实际纹理检查' not in text:evidence.write_text(text.rstrip()+extra+'\n',encoding='utf8')
sync=Path('D:/Projects/EngineLab/engine-home/scripts/sync-gs-lab.mjs')
text=sync.read_text(encoding='utf8')
if 'outputs/dynamic-human-enerf-background-fit-033/${name}' not in text:
    lines=text.splitlines(keepends=True)
    index=next(i for i,line in enumerate(lines) if 'outputs/dynamic-human-enerf-background-032/${name}' in line)+1
    lines.insert(index,"""  ...['final-summary.json', 'density-capped/independent-audit.json', 'density-capped/visual-review.json', 'density-capped/evaluation/camera06-wall-comparison.png', 'density-capped/evaluation/camera06-trained.png'].map(name => [`outputs/dynamic-human-enerf-background-fit-033/${name}`, '背景拟合后的真实纹理检查，尚未通过质量要求']),
  ...['final-summary.json', 'visual-review.json', 'static-registration.json', 'static-pose-diagnostic.json', 'pose-floor-check.json', 'camera08-frame000750-wall-matte-comparison.png'].map(name => [`outputs/dynamic-human-enerf-matte-background-034/${name}`, '人物前景透明度、真实阴影与静态投影诊断；未修改正式相机']),
""")
    sync.write_text(''.join(lines),encoding='utf8')
print('Theory and bounded evidence updated;11selected public assets. No models or raw archives.')
