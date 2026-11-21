---
id: "gslab-exp-affine-icp-fallback-controls-20261009"
slug: "affine-icp-fallback-controls"
title: "25 相邻 ICP 回退：解除停滞与定位误差的取舍"
order: 25
status: "two_full_CPU_paths_and_public_fields_independently_checked"
updated: "2026-10-09"
sources:
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats"
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools"
---

# 相邻 ICP 回退：解除停滞与定位误差的取舍

这次在固定八维光度—几何路线之外，加入一个明确的选择规则：primary 被拒绝时，可以使用同一当前帧到立即前一传感器帧的既有 ICP 相对候选。desk 的最终拒绝从旧 8D 的 420 帧变成 0，但新轨迹 raw RMSE 仍为 189.421 mm，高于旧完整 ICP 的 142.881 mm。xyz 同样没有最终拒绝，raw、ATE 和末帧位置误差却比旧 8D 更高。**保持状态持续更新，与轨迹持续准确，是两种不同的结果。**

## 固定两条完整关联序列，保留全部状态

实际按 desk、xyz 顺序执行两个 CPU 子进程，真实退出码均为 0。desk 包含全部 549 个关联帧，xyz 包含全部 748 个关联帧，总计 1,297 帧、1,295 个非首帧对。关联规则、输入、原始 8D 求解器、权重、尺度、门槛和每对 identity/a=0/b=0 初始化均预先冻结，没有根据 GT 调参、改片段、重试或挑选候选。

这里的全部关联帧是固定 loader 完成时间关联、深度去重和帧率规则后的序列，不是目录中的全部 raw RGB。首相机每个序列只设置一次 identity；每个时间戳都保留最终估计。比较对象是此前已完成的完整 last-accepted 8D 与完整 ICP，没有为本次对比重跑旧模型。

## 三条分支：primary、fallback、hold

设 r 是最后一次**最终接受**的传感器参考，k 是当前帧，`Ĉ` 是保存的 c2w 估计：

```text
primary 接受：Ĉ_k = Ĉ_r T_(r←k, primary)
否则，合格的相邻 ICP 候选可用：Ĉ_k = Ĉ_(k−1) T_((k−1)←k, ICP)
否则，两路拒绝：Ĉ_k = copy(Ĉ_(k−1))

任一接受分支都将当前 RGB-D、传感器 ID 和最终位姿一起提升为新参考。
只有双拒绝才 hold；primary 自身失败标记始终保留。
```

候选优先级固定，GT 不参与分支选择。回退读取的是已完成 ICP 的**相邻相对候选**，不是把 ICP 绝对轨迹拼进当前结果，也不是拿 ICP 绝对位姿初始化 primary。合格条件包括原候选接受、原 flags 为空、相邻 ID、完整传感器身份/时间戳、冻结核心源码及正确 current→previous 方向和刚体结构。

提取工具确实读过包含 GT/绝对位姿字段的旧完整 JSON，不能写成“整个数据获取链从未见过这些字段”；传入选择逻辑的是严格白名单化的相对候选证据。它不带 GT 矩阵或 ICP 绝对位姿。冻结 loader 也读取 GT 字段与时间进行关联，但 GT 矩阵不进入配准、初始化、优化或选择，全部最终估计导出后才用于评价。

合法 held-reference 账本中，若中间帧一直 hold，`Ĉ_r` 和 `Ĉ_(k−1)` 可以数值相等；传感器 ID 与候选实际方向仍然不同。若此前双拒绝遗漏了运动，后续接受一个相邻 ICP 变换也不会自动补回那段运动。此次没有双拒绝，因此真实结果并没有检验这种缺失历史的恢复能力。

## 分类必须按尝试层和最终层分别统计

| 序列 | 总帧 / 非首帧对 | primary 接受 | primary 拒绝 | 选择 ICP 回退 | 最终 hold / 拒绝 |
|---|---:|---:|---:|---:|---:|
| desk | 549 / 548 | 525 | 23 | 23 | 0 |
| xyz | 748 / 747 | 726 | 21 | 21 | 0 |

表中 primary 接受不包含首相机；原 `selection_counts` 另外保留一次 `fixed_first_camera_identity`。这次冻结的相邻 ICP 证据在 desk 的 548 对与 xyz 的 747 对中均合格，但只有 primary 拒绝后才被选择。不要把“候选合格数量”当作“实际回退数量”。

desk 的 23 次 primary 拒绝含 16 次 `finest_level_line_search_no_descent` 与 7 次强度工程门槛标记；xyz 的 21 次 primary 拒绝均为 line-search 标记。这些失败并未从账本消失，而是与该帧最终接受的 fallback 并存。最终 flags 为空，不能被简写成“所有 primary 均收敛”。

旧 desk 8D 的 420 次最终拒绝与新管线的 23 次 primary 拒绝，来自两条不同的递推历史。第一次 fallback 改变最终位姿和后续参考后，后续 primary 接收的状态已经不同，不能说把原 420 个相同候选逐一救回。xyz 两条路径的计数都为 21，也不是同一失败集合：旧 8D 中 381 被拒绝，新 primary 中没有这一拒绝，而新增了 573。接受路径变化本身就是比较的一部分。

## 六条完整轨迹：指标各自回答什么

平移列统一为 mm，旋转 RPE 为度。新两条轨迹使用原 metrics 保存的值；旧 8D 使用 `d6a5…` 完整摘要，旧 ICP 使用 `97b6…` 已独审的 baseline 字段，未重新拟合或重算。全部帧参与 raw/ATE，全部相邻对参与 RPE。

| 序列 | 方法 | raw RMSE / mm | SE3 ATE / mm | 相邻 RPE 平移 / mm | 相邻 RPE 旋转 / ° | 末帧位置 / mm | 最终拒绝 / 全帧 |
|---|---|---:|---:|---:|---:|---:|---|
| desk | 旧完整 ICP，只读 | 142.881 | 83.043 | 8.913 | 0.868 | 184.811 | 0 / 549 |
| desk | 旧完整 last-accepted 8D，只读 | 933.760 | 825.241 | 17.246 | 1.393 | 1035.527 | 420 / 549 |
| desk | 新 8D + 相邻 ICP 回退 | 189.421 | 93.344 | 10.010 | 0.725 | 275.284 | 0 / 549 |
| xyz | 旧完整 ICP，只读 | 44.151 | 30.163 | 3.540 | 0.504 | 71.448 | 0 / 748 |
| xyz | 旧完整 last-accepted 8D，只读 | 106.978 | 58.622 | 6.134 | 0.408 | 173.651 | 21 / 748 |
| xyz | 新 8D + 相邻 ICP 回退 | 116.652 | 63.845 | 5.102 | 0.387 | 186.365 | 0 / 748 |

desk 新管线不再出现旧 8D 的 415 帧连续 hold，表中各项也比旧 8D 更低。但与完整 ICP 相比，新管线的 raw、ATE、平移 RPE 和末帧误差都更高，只有相邻旋转 RPE 更低。因此“解除长期停留”有直接行为证据，“整体优于 ICP”没有。

xyz 的零最终拒绝没有带来更低的位置偏差：相对旧 8D，raw 从 106.978 增到 116.652 mm，ATE 从 58.622 增到 63.845 mm，末帧从 173.651 增到 186.365 mm；平移和旋转 RPE 则都降低。相对完整 ICP，新管线的位置类指标仍更高，旋转 RPE 更低。这些方向不同的变化不能压成一个“提升百分比”。

raw 使用首相机定义的共同坐标，不做拟合；SE3 ATE 对完整位置轨迹做后验刚体对齐，尺度固定为 1；相邻 RPE 测的是局部变化差异。局部增量更贴近评价参考，不保证累计位置更准确；后验 ATE 较低也不表示在线输出已经完成该对齐。

全程 raw 旋转 RMSE 是另一项：新 desk 为 2.754°，新 xyz 为 4.804°；旧完整 8D 分别为 45.402°、4.396°。它衡量与首相机评价坐标下参考方向的整体偏差，不是表中的相邻旋转 RPE。不能混用两个旋转指标来挑选更有利的表述。

## 完整误差曲线与参考间隔

下面直接绘制每个关联帧保存的 raw 位置误差，保留全部 hold。纵轴是 m；两个序列使用各自量程。右图只使用原八维和新回退管线保存的参考 ID 间隔，没有给 ICP 补造该字段。叉号标记实际选择的 fallback，desk 阴影表示旧八维 134–548 的 415 帧连续拒绝。

![desk 全帧位置误差与参考间隔](../outputs/rgbd-affine-icp-fallback-figures-001/desk-full-raw-error-and-reference-gap.png)

![xyz 全帧位置误差与参考间隔](../outputs/rgbd-affine-icp-fallback-figures-001/xyz-full-raw-error-and-reference-gap.png)

图中 xyz 的参考间隔保持为 1，但新轨迹累计位置误差并没有因此低于旧路线。图片没有重对齐、重锚定、删除帧或重新配准；它们把状态连续与位置准确的差别直接展示出来。

[绘图源码](../scripts/plot_rgbd_affine_icp_fallback_comparison_v2.py)、[固定输入预检](../outputs/rgbd-affine-icp-fallback-figure-preparation-002/preflight.json)、[图像清单](../outputs/rgbd-affine-icp-fallback-figures-001/figures.json)及[源码与目视核对](../outputs/rgbd-affine-icp-fallback-figures-001/root_visual_review.json)保留全部来源。整份旧 JSON 被载入，包含保存的 GT 派生后验对齐及关联字段；绘图只复制原误差数组，没有使用其中矩阵计算新误差。首次过强 endpoint 浮点相等断言的失败保留在本地，当前沿用原审计 1e−11 m 的比较界，曲线值仍来自原保存数组。

## 这是什么控制，又不证明什么

比较对象是固定八维 primary 加一条固定选择与回退路径的完整管线。primary 数学模型并没有因为 fallback 增加亮度自由度；变化包括选中位姿、参考何时前移以及随后所有观测的线性化和接受路径。结果不是“照明校正导致改善”的单变量因果证据，更不能把这些工程失败标记解释成已证明的曝光原因。

合格 fallback 在本实验中是提前计算并冻结的 ICP 相邻候选。新管线保存的 CPU 计时包括 primary、几何准备与 I/O，desk 为 105.481 s、xyz 为 143.768 s，却没有重新包含生成全部这些 ICP 候选的成本。它们只能作为这次执行的范围说明，不可除以帧数宣称在线 FPS，也不能与旧 ICP 做相同计算预算的速度排名。

两个序列同属 TUM fr1，内参和采集域相关；本页不跨序列平均、不称通用跨场景优势、不称完整 SLAM。系统仍没有凭这些结果实现全局重定位、回环或联合 BA。拒绝率、持续运行与位姿误差应分别报告。

## 真实独审与公开子集的范围

[新独立结果审计](../outputs/rgbd-affine-icp-fallback-validation-001/independent_affine_icp_fallback_result_audit.json)已经真实 exit 0，SHA256 为 `69b3856c…87715`。它绑定 f953… 源码、0865… 审计协议与 9727… 主代理源码审阅，覆盖全部 1,297 帧、1,295 对、33,900 条 primary 步骤、35,746 条试探及 2,594 个原始传感器 PNG，5,280 份消费文件前后 hash 不变。

审计显式复用既有 b549 的保存八维步进核查与 e807 的原 GT/SciPy 指标路径，并独立核对 primary/fallback/hold 方向、传感器身份、选择优先级、完整 flags 与原 immutable 文件。它没有重新跑配准，也没有独立重建每个真实对应集的秩和目标函数。

平移指标及数组的最大绝对差不超过 `9.385e−16 m`，旋转指标及数组不超过 `2.987e−14°`。保存 SE3 步进矩阵的最大元素差不超过 `7.000e−16`，原始 GT 构造的 4×4 anchor 矩阵最大元素差不超过 `1.333e−15`；后两项混合旋转、平移和齐次元素，不能统一标成米。很小的审核差异说明声明的数值路径一致，不证明真实轨迹准确。

旧完整 8D 的原审计为 a948…，对应摘要 d6a5…；后加的 1d75… 只补充旧六维接受步骤的本行 proposal 守卫，不属于本次 hybrid 的新轨迹重算。旧完整 ICP 仍绑定 70f65… 独审与 97b6… 摘要。三种证据不能拼成“多套完全独立算法都证明当前定位准确”。

新 [collector 源码](../scripts/summarize_rgbd_affine_icp_fallback_results.py)仅复制已审数据，SHA256 为 `b5f7dba3…7c08fd`，已在源码审核后真实 exit 0。它按实际字节数动态分片：desk 为 5 页、xyz 为 7 页，共 12 页，每页最多 200 帧、严格小于 1 MiB，完整覆盖 1,297 帧。每行保存原 pair 全部顶层字段、完整 primary/fallback **最终候选**、亮度、统计、配置、引用 ID、gap、最终 c2w、完整当前 sensor metadata 和原四条逐帧/逐间隔误差数组。anchor 没有的尝试不补造，原显式 null 不换成零。

仅 `primary.iteration_history` 大型试探历史不复制到公开页，而用声明过编码方式的 canonical hash、步骤/试探数及原 immutable pair、candidate、JSONL 文件 hash 绑定。所有试探仍保留在本地原证据中；“每帧最终候选完整公开”不等于“全部中间试探都在网站”。本次导出核对 2,672 份消费文件前后 hash 不变，6 页输入 hash 清单完整保留；完成清单列出的 22 份文件（含源码快照）也已重新核对大小与 SHA。这是导出完整性检查，独立公开字段复核已另行执行，通过全部 1,297 行及其原始绑定。

## 可核对的来源

- [固定两轮协议](../outputs/rgbd-affine-icp-fallback-cpu-preparation-001/protocol.json)、[执行前主代理审核](../outputs/rgbd-affine-icp-fallback-cpu-preparation-001/root_protocol_review.json)、[实际终态批次](../outputs/rgbd-affine-icp-fallback-validation-001/metrics.json)。
- [新审核源码](../scripts/audit_rgbd_affine_icp_fallback_results.py)、[审核协议](../outputs/rgbd-affine-icp-fallback-result-audit-cpu-preparation-001/protocol.json)、[审核源码审阅](../outputs/rgbd-affine-icp-fallback-result-audit-cpu-preparation-001/root_source_review.json)、[实际独审结果](../outputs/rgbd-affine-icp-fallback-validation-001/independent_affine_icp_fallback_result_audit.json)。
- [旧完整 8D 摘要](../outputs/rgbd-affine-full-comparison-001/summary.json)、[原完整审计](../outputs/rgbd-affine-full-validation-001/independent_full_affine_result_audit.json)、[旧六维字段补充审计](../outputs/rgbd-affine-full-6d-proposal-addendum-001/metrics.json)。
- [旧完整 ICP 所在摘要](../outputs/tum-online-v4-full-sequence-recovery-summary-001/summary.json)、[旧完整轨迹独审](../outputs/tum-online-v4-full-sequence-recovery-summary-001/independent_summary_audit.json)。
- [新完整汇总](../outputs/rgbd-affine-icp-fallback-comparison-001/summary.json)、[desk 摘要与 5 页索引](../outputs/rgbd-affine-icp-fallback-comparison-001/desk-summary.json)、[xyz 摘要与 7 页索引](../outputs/rgbd-affine-icp-fallback-comparison-001/xyz-summary.json)、[完成清单](../outputs/rgbd-affine-icp-fallback-comparison-001/completion.json)。原帧页依上述索引顺序覆盖 desk 0–548 与 xyz 0–747；独立字段保留检查已实际通过，覆盖所有帧、历史绑定和原始输入哈希。
- [原完整 last-accepted 实验 24](2026-10-09-full-affine-reference-controls.md)。

新结果表明：desk 的这条固定回退路径确实解除原实现的长期 hold，但两条完整序列都没有在所有指标上优于旧 ICP；xyz 还展示了最终拒绝清零、相邻 RPE 降低而全程位置误差增大的反例。分支状态、全部失败与完整轨迹共同构成结论。

## 完整分页与字段保留

[独立字段核对](../outputs/rgbd-affine-icp-fallback-public-field-check-001/metrics.json)以标准库逐行比较原终态数据、完整传感器记录、最终候选、四条原误差数组和历史哈希，未运行配准或重新评价 GT。全部 1,297 行通过，2,698 份消费文件保持；其中 2,672 份是原 collector 输入。 [主代理核对](../outputs/rgbd-affine-icp-fallback-public-field-check-001/root_public_summary_review.json)记录其范围。

desk 帧页：[0–116](../outputs/rgbd-affine-icp-fallback-comparison-001/desk-frames-00.json) · [117–232](../outputs/rgbd-affine-icp-fallback-comparison-001/desk-frames-01.json) · [233–348](../outputs/rgbd-affine-icp-fallback-comparison-001/desk-frames-02.json) · [349–464](../outputs/rgbd-affine-icp-fallback-comparison-001/desk-frames-03.json) · [465–548](../outputs/rgbd-affine-icp-fallback-comparison-001/desk-frames-04.json)

xyz 帧页：[0–116](../outputs/rgbd-affine-icp-fallback-comparison-001/xyz-frames-00.json) · [117–231](../outputs/rgbd-affine-icp-fallback-comparison-001/xyz-frames-01.json) · [232–347](../outputs/rgbd-affine-icp-fallback-comparison-001/xyz-frames-02.json) · [348–462](../outputs/rgbd-affine-icp-fallback-comparison-001/xyz-frames-03.json) · [463–577](../outputs/rgbd-affine-icp-fallback-comparison-001/xyz-frames-04.json) · [578–692](../outputs/rgbd-affine-icp-fallback-comparison-001/xyz-frames-05.json) · [693–747](../outputs/rgbd-affine-icp-fallback-comparison-001/xyz-frames-06.json)

输入哈希：[分页 1](../outputs/rgbd-affine-icp-fallback-comparison-001/input-hashes-00.json) · [分页 2](../outputs/rgbd-affine-icp-fallback-comparison-001/input-hashes-01.json) · [分页 3](../outputs/rgbd-affine-icp-fallback-comparison-001/input-hashes-02.json) · [分页 4](../outputs/rgbd-affine-icp-fallback-comparison-001/input-hashes-03.json) · [分页 5](../outputs/rgbd-affine-icp-fallback-comparison-001/input-hashes-04.json) · [分页 6](../outputs/rgbd-affine-icp-fallback-comparison-001/input-hashes-05.json)

网站对这 12 份帧页仍执行本地路径清洗，随后保留紧凑 JSON，并再次检查 1 MiB 上限。原科学证据的 source SHA 与网站 served SHA 分别记录；网页排版变化不被声称为原文件字节完全相同。

字段保留的[核查源码](../scripts/check_rgbd_affine_icp_fallback_public_fields.py)、[固定协议](../outputs/rgbd-affine-icp-fallback-public-field-check-preparation-001/protocol.json)和[执行前源码审阅](../outputs/rgbd-affine-icp-fallback-public-field-check-preparation-001/root_source_review.json)也可直接阅读。

## 实测：第一次回退后 参考历史怎样分叉

只读对齐旧完整 8D 与新 hybrid 的保存结果：desk 第 1–10 帧、xyz 第 1–13 帧，固定求解器的全部 **19 个常规返回字段**（含完整层级、步骤与试探诊断）对应值完全相同；这也包含第一次 primary 拒绝。首次 ICP 回退发生在 desk 第 10 帧、xyz 第 13 帧。旧路径保持参考 9/12，新路径接受回退后把当前传感器提升为参考 10/13。

紧接着的 desk 第 11 帧、xyz 第 14 帧使用不同参考；19 个求解器字段中的 12 个已经出现真实差异，包括候选位姿、亮度估计和迭代诊断。这给“递归路径比较”提供了具体边界：回退改变之后的输入历史，不能把完整序列拒绝次数之差称作同一批候选逐个获救。比较只覆盖前缀与边界后的一帧，不宣称全未来等价。

旧记录把求解器字段平铺，新记录放在 primary 下并另存 fallback；传感器记录中旧有的 GT 关联元数据也另列为 schema 差异。这里的对应返回对象相同不意味着完整行或原 JSON 文件字节相同。[比较源码](../scripts/compare_affine_hybrid_primary_prefix.py)、[实际逐帧摘要](../outputs/affine-hybrid-primary-prefix-comparison-001/metrics.json)、[归档清单](../outputs/affine-hybrid-primary-prefix-comparison-001/completion.json)及[主代理复核](../outputs/affine-hybrid-primary-prefix-comparison-001/root_source_and_result_review.json)保存输入哈希和完整差异文件的绑定。原完整 JSON 含 GT 评估字段，被解析但没有重算 GT 指标、注册或优化；5,430 条差异记录留在本地。
