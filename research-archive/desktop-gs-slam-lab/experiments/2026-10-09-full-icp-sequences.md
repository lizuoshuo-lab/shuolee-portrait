---
id: "gslab-exp-full-icp-sequences-20261009"
slug: "full-associated-rgbd-icp-sequence-controls"
title: "完整关联序列 ICP：零拒绝与长程漂移"
order: 17
status: "completed_full_associated_sequences_independently_reviewed"
updated: "2026-10-09"
sources:
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats"
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download"
---

# 完整关联序列 ICP：零拒绝与长程漂移

同一套固定参数的 CPU 深度 ICP 已跑完 TUM `fr1_desk` 的 549 个关联帧和 `fr1_xyz` 的 748 个关联帧。两轮都没有触发拒绝标记，但首相机坐标下的平移 RMSE 分别为 142.881mm 和 44.151mm。局部配准能连续接受相邻帧，不代表累计轨迹准确，也不代表每个优化层都收敛。

## “完整”是完整关联序列

| 序列 | 原始 RGB / 深度文件数 | 保留的关联帧 | 相邻配准对 | 关联 RGB 时间跨度 |
| --- | ---: | ---: | ---: | ---: |
| fr1_desk | 613 / 595 | 549，ID 0–548 | 548 | 19.807376s |
| fr1_xyz | 798 / 798 | 748，ID 0–747 | 747 | 26.572059s |

这里没有把全部原始 RGB 帧送进求解器。冻结 loader 先按 RGB 时间选最近 depth 和最近 GT，两项时间差都严格小于 20ms，再使用 32Hz 上限并禁止复用深度文件。表中的 ID 是这些筛选后的序号，不能当作原始 RGB 索引。

GT 时间戳参与关联；GT 位姿只用于完成后的轨迹评测，不进入初始化、对应搜索、残差、法方程或候选选择。两条序列分别从首相机 `c2w=I` 开始。数据仍采用 Freiburg1 内参、深度 `/5000` 的米制换算和已注册 RGB-D 假设，没有 GT 插值、RGB/深度运动补偿或去畸变。

## 冻结求解器，只扩展输入长度

[原始 ICP 源码](../scripts/track_rgbd_icp.py)的 SHA256 保持 `21b7e11bb6d6337e7c9bbfcacd8f324349bed22dd9f2fea9858caf44fe915f0a`，loader 保持 `471a0aebb80c7268b5388bc85a556bdf99c8fda0355dd3307ca7bd3639b71b11`。[desk 完整序列 wrapper](../scripts/track_rgbd_icp_full.py)与 [xyz 完整序列 wrapper](../scripts/track_rgbd_icp_xyz_full.py)在各自进程内添加固定 clip，调用已有 API，没有修改求解器或调参。

两轮都使用 40×30、80×60、160×120 三层，迭代上限分别为 15、12、10；距离门限为 0.10、0.07、0.05m，法向夹角门限 45°，Huber 尺度 0.01m。有效深度为包含端点的 0.2–5m。点到平面残差、边界拒绝与退化检查的含义见[雅可比与可观测性](../knowledge/10_rgbd_icp_observability.md)。

相邻配准返回的是把当前相机点变到上一相机坐标的 `T_previous_from_current`。累计相机到世界矩阵必须按以下方向递推：

```text
p_previous = T_previous_from_current @ p_current
G_current  = G_previous @ T_previous_from_current
G_first    = I
```

每一对只使用当前、上一帧深度，没有高斯地图、回环、重定位或全局 BA。失败策略是保留该帧并传播上一估计；本轮两条序列均未触发拒绝，但全部帧及诊断仍保留在结果中。

## 全部帧上的轨迹结果

平移单位为 mm，旋转单位为度。RPE 的间隔为相邻关联帧 `Δ=1`，不是固定毫秒间隔。

| 序列 | 首相机锚定平移 RMSE | SE3 对齐 ATE | 相邻平移 RPE | 相邻旋转 RPE | 末帧锚定平移误差 | 最大锚定平移误差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| desk | 142.881 | 83.043 | 8.913 | 0.867657° | 184.811 | 234.271 |
| xyz | 44.151 | 30.163 | 3.540 | 0.503527° | 71.448 | 85.491 |

原始锚定参考为 `H_i=inverse(GT_first) @ GT_i`。ATE 另外为每条序列拟合一次全局旋转和平移，使对应相机中心误差最小，尺度固定为 1；它没有修正求解器的输出，也没有消除局部漂移。相邻 RPE 使用：

```text
delta_GT       = inverse(H_i) @ H_(i+1)
delta_estimate = inverse(G_i) @ G_(i+1)
error          = inverse(delta_GT) @ delta_estimate
```

平移 RPE 取 error 的平移向量范数；旋转 RPE 取其旋转角，再分别对全部相邻对计算 RMS。SE3 对齐 ATE 较小不等于首相机锚定轨迹同样准确，因此两类指标一起保留。

![完整关联序列的轨迹、锚定中心误差与相邻平移误差](../outputs/full-icp-two-sequence-summary-001/comparison.png)

两条序列具有独立的首相机世界坐标，图中各序列也使用独立坐标轴范围。左列只是 X–Z 投影，数值指标使用完整三维位姿。不能按曲线看起来更贴合、图上轨迹大小或两序列误差大小，将场景与运动差异解释为方法优劣。两轮使用的是同一个求解器，这也不是跨场景泛化的证明。

## 零失败标记不等于每层收敛

| 序列 | 拒绝帧 / 总帧 | 达到迭代预算的层 | 小更新停止的层 | 全部迭代记录 |
| --- | ---: | ---: | ---: | ---: |
| desk | 0 / 549 | 868 / 1644 | 776 / 1644 | 16076 |
| xyz | 0 / 748 | 814 / 2241 | 1427 / 2241 | 18425 |

1644 和 2241 分别等于相邻配准对数乘以三层。`iteration_budget` 表示该层耗尽预设迭代次数，不等于触发整帧失败，也不能改写成“已收敛”。`small_update` 只表示当前更新量满足停止阈值，不能证明全局最优或正确对应。

所有最终配准的数值秩检查都得到 rank 6，但秩、列归一化 Hessian condition、残差阈值都是当前线性化与启发式诊断。它们不是标定过的位姿协方差，更不能用来替代轨迹误差。desk 已显示超过 200mm 的局部累计中心偏差，同时整轮仍为零拒绝。

## 实测耗时的边界

| 序列 | tracking 循环墙钟时间 | 冻结 main 调用墙钟时间 |
| --- | ---: | ---: |
| desk | 43.598s | 58.720s |
| xyz | 53.039s | 73.835s |

循环时间包含几何构造、配准、位姿累计，以及逐迭代 JSONL/CSV 写入和刷新、逐帧输出；不包含金字塔 PNG 预加载、最终 GT 评测和最终指标导出。main 调用还包含 loader、评测及产物导出，但不含 wrapper 的预检与收尾校验。

这些是本机该次运行的墙钟耗时，不是进程 CPU 时间、隔离性能测试或完整 SLAM FPS。不能用序列帧数直接除上述时间后，当作与其他系统公平对比的速度。

## 短片段相对候选相同，累计矩阵仍需重置坐标

xyz 的完整运行包含此前的关联帧 200–299。对照采用这一百帧内的 **99 对**相邻候选：完整运行的 200←201 到 298←299（箭头指向上一相机坐标），与短运行的相同传感器输入逐一对应；完整运行中 199←200 的那一步不属于短片段。

[一致性记录](../outputs/tum-xyz-icp-full748-001/short_segment_relative_parity.json)显示，99 对 `T_previous_from_current` 的最大绝对数值差为 **0**，接受/拒绝状态、对应计数、层停止原因相同，792 个最终统计字段也通过比较。这支持当前逐对几何计算在这两次运行中的一致性，不是另一个 ICP 算法独立证明配准准确。

完整运行以 frame 0 为世界原点，短运行以 frame 200 为原点，不能直接相减累计 c2w。先做：

```text
G_full_rebased_i = inverse(G_full_200) @ G_full_i,  i=200..299
```

使用解析刚体逆重置坐标后，累计矩阵最大绝对差为 **1.709743457922741×10⁻¹⁴**，低于 `1e-10` 比较容差，但并非逐字节相同。独审改用通用矩阵求解时差为 `2.109424×10⁻¹⁵`；不同浮点求逆/累计路径存在舍入差异。不能把“相对候选差为零”扩大成所有累计矩阵和产物文件 byte-identical。这一比较不使用 GT 位姿。

## 独立复算能排除哪些错误

[只读汇总源码](../scripts/summarize_full_icp_sequence_controls.py)从原始 TUM xyzw 和已保存关联索引重建 GT。[独立审阅](../outputs/full-icp-two-sequence-summary-001/independent_summary_review.json)再用另一条计算路径核对：不导入该脚本、求解器或 loader 的函数，直接读取 RGB/depth/GT 索引，以 `searchsorted` 重建全部时间关联；用 SciPy 四元数变换、通用矩阵求解和旋转向量角度复算指标，并核对每个实际估计矩阵及累计方向。

独审与汇总的最大指标差约为 `1.11×10⁻¹⁶m` 和 `8.93×10⁻¹¹°`；全部文件证据 hash、帧数、失败标记和层预算统计一致。该精度描述的是数值复算一致性，不是定位对物理真值达到这些精度。

负控也必须有辨别能力。此次核对保留了三点边界：

- 错把原始 xyzw 当成 wxyz，重建矩阵和 raw 锚定误差会明显改变；但只用相机中心的全局 SE3 对齐 ATE 可以消除首帧锚定引入的常量旋转，因而几乎不变。不能只靠这个 ATE 检查四元数约定。
- 若把刚体 RPE 残差换成它的逆，平移由 t 变为 `−Rᵀt`，范数不变，旋转角也不变。因此“残差取逆后标量相同”不是有效的检错成功。换乘法侧在实际和非对易合成数据上能改变平移范数，旋转角却仍可能因共轭不变而相同；独审没有把它冒充成旋转负控。
- 若累计时误用 `inverse(T_previous_from_current)`，逐矩阵比较会失败；实际单步矩阵最大差分别为约 0.256857 和 0.071245。方向检查比只看最终某个标量更直接。

[完整序列汇总](../outputs/full-icp-two-sequence-summary-001/summary.json)、[desk 原始摘要](../outputs/tum-icp-full549-001/summary.json)、[xyz 原始摘要](../outputs/tum-xyz-icp-full748-001/summary.json)和 [xyz 校验](../outputs/tum-xyz-icp-full748-001/validation.json)保留具体配置、真实数值、停止原因、时间边界与来源。以上审查证明产物及评测约定相互对应；它没有重新运行配准，不证明优化器总能找到正确对应，也没有补上地图、回环或全局一致性能力。
