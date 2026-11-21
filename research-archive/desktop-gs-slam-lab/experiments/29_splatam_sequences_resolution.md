---
id: "gslab-exp-splatam-sequences-resolution-20261010"
slug: "splatam-sequences-resolution"
title: "29 SplaTAM：运动序列与分辨率对照"
order: 29
status: "two_sequences_resolution_and_common_gt_view_independently_verified"
updated: "2026-10-10"
sources:
  - "https://github.com/spla-tam/SplaTAM/tree/da6bbcd24c248dc884ac7f49d62e91b841b26ccc"
  - "https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/configs/tum/splatam.py"
  - "https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/datasets/gradslam_datasets/basedataset.py"
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset"
---

# SplaTAM：运动序列与分辨率对照

本轮保留原始源码、renderer 和优化规则，已完成 **xyz 全 759 帧、320×240** 与 **desk 全 592 帧、640×480** 两次真实运行，并通过独立验收。xyz 相机中心 ATE RMSE 为 0.009429 m；desk 升至 640×480 后 ATE RMSE 为 0.019527 m，比原 320×240 降低约 3.04%，地图高斯数量却为原来的 4.35 倍、完整处理墙钟为 1.93 倍。两份 desk 地图在共同像素网格和共同 GT 视角下的实际重渲染也已完成，图像质量没有全面改善。这是单 seed 的实测配置比较，基线模型与旧 20 帧 smoke 完整保留。

这两组比较回答不同问题。desk 与 xyz 使用同一 Freiburg1 公共相机参数，属于另一条运动序列的适用性检查；它们不构成跨传感器泛化。desk 320×240 与 desk 640×480 使用同样的输入时间段和逐帧关联，才形成同序列的分辨率配置对照。[首轮完整项目](28_splatam_official_local_project.md)保存基线，机制解释见[RGB-D 高斯 SLAM](../knowledge/24_splatam_official_rgbd_system.md)。

## 比较对象与实际状态

| 配置 | 官方关联帧 | 观测跨度 | 状态 | 独立相机中心 ATE RMSE |
|---|---|---|---|---|
| desk，320×240，seed 0 | 592 / 592 | 19.843 s | 完整运行及独立验收通过 | 0.020141 m |
| xyz，320×240，seed 0 | 759 / 759 | 26.572 s | 完整运行及独立验收通过 | 0.009429 m |
| desk，640×480，seed 0 | 592 / 592 | 19.843 s | 完整运行及独立验收通过 | 0.019527 m |

基线还给出相邻帧 RPE 0.009633 m／0.568176°、223,097 个高斯、120 个关键帧，实际官方子进程用时 1,620.381 s。数值来源为[基线项目汇总](../outputs/splatam-project-001/metrics.json)及[完整独立轨迹评测](../outputs/splatam-independent-eval-full-001/metrics.json)，不能把这些旧运行的成绩移到新配置上。

## xyz 完整序列的真实结果

| 项目 | xyz，759 帧，320×240，seed 0 |
|---|---|
| 相机中心 SE(3) ATE RMSE／均值／最大值 | 0.009429 / 0.008066 / 0.032731 m |
| 首相机参考系、无整段额外对齐的平移 RMSE | 0.017195 m |
| 相邻帧平移／旋转 RPE RMSE | 0.005120 m / 0.376391°，758 对 |
| 保存高斯数量 | 156,264 |
| 实际官方子进程墙钟 | 1,808.525 s，约 30.14 min |
| 独立轨迹验收／实际运行读回 | 22 / 22；28 / 28 |
| 已观测采样视角平均 PSNR／MS-SSIM／LPIPS | 25.517306 dB / 0.961528 / 0.071223，38 个样本 |
| 有效 GT 像素上的平均深度 MAE | 0.022784 m |

[xyz 独立指标](../outputs/splatam-independent-eval-xyz-full-001/metrics.json)、[轨迹评测报告](../outputs/splatam-independent-eval-xyz-full-001/report.md)和[实际运行补充审核](../outputs/splatam-independent-eval-xyz-full-001/runtime-review.md)绑定 `freiburg1_xyz_full_320x240_s0_20261010T072916Z_8b1944d0`。真实子进程退出 0，日志确认首帧初始化后 758 个 tracking 帧逐帧完成 200 / 200 次迭代，759 个 mapping 帧完成 30 / 30。验收通过表示运行和评测证据完整，不隐含额外精度阈值。

![xyz 全 759 帧真实轨迹与误差](../outputs/splatam-independent-eval-xyz-full-001/trajectory-errors.png)

38 个采样帧的编号和原生／Windows 输出哈希均通过[逐样本核对](../outputs/splatam-independent-eval-xyz-full-001/sample-verification.json)。最低 PSNR 来自第 79 帧，17.794897 dB；该帧深度 MAE 为 0.079121 m，显示器边缘、人物轮廓与深度细节存在明显差异。平均值不能覆盖这类局部较差样本，单张图也不能把原因归于某一模块或推断人物运动。

![xyz 第 79 帧：38 个固定采样中最低 PSNR 的实际输出](../outputs/splatam-followup-001/xyz-full-lowest_psnr.png)

![xyz 第 739 帧：固定采样的最后一个实际输出](../outputs/splatam-followup-001/xyz-full-last_sample.png)

两张图分别按最低 PSNR 和最后采样帧选取，均与上游原图字节一致。完整输入最后一个观测为第 758 帧，图像评测最后采样为第 739 帧；采样结束位置不代表轨迹或建图提前结束。选图规则与哈希见[本轮报告](../outputs/splatam-followup-001/report.md)。

xyz 在这一次运行中的轨迹误差低于 desk 基线，但纹理、几何、观测运动与时长同时变化；这不证明模型泛化能力获得改进，也不能把高斯更少或总用时差异单独归因于序列帧数。26.572 s 的观测用约 30.14 min 处理，仍是离线复现。

## desk 分辨率对照：成本增加与轨迹变化

| 同一 desk 全 592 帧，seed 0 | 320×240 | 640×480 |
|---|---|---|
| 相机中心 SE(3) ATE RMSE | 0.020141 m | 0.019527 m |
| 相邻帧平移 RPE RMSE，591 对 | 0.009633 m | 0.009260 m |
| 相邻帧旋转 RPE RMSE，591 对 | 0.568176° | 0.557781° |
| 首相机参考系平移 RMSE，无整段额外对齐 | 0.043678 m | 0.047260 m |
| 高斯／关键帧数量 | 223,097 / 120 | 970,345 / 120 |
| 实际官方子进程墙钟 | 1,620.381 s，27.01 min | 3,122.868 s，52.05 min |
| 独立轨迹验收／实际运行读回 | 22 / 22；28 / 28 | 22 / 22；28 / 28 |

[640×480 独立指标](../outputs/splatam-independent-eval-desk-640-full-001/metrics.json)、[完整评测报告](../outputs/splatam-independent-eval-desk-640-full-001/report.md)和[真实运行审核](../outputs/splatam-independent-eval-desk-640-full-001/runtime-review.md)绑定 `freiburg1_desk_full_640x480_s0_20261010T080134Z_823ff9d6`。日志实际确认 591 个 tracking 帧逐帧 200 / 200、592 个 mapping 帧逐帧 30 / 30，真实子进程退出 0；第 0 帧初始化。[本轮汇总](../outputs/splatam-followup-001/metrics.json)核对两种 desk 配置的原始输入、选中时间戳、其它协议、官方算法源码和 renderer 二进制，6 项对照检查全部通过，旧基线汇总的 SHA256 未改变。

![desk 640×480 全 592 帧真实轨迹与误差](../outputs/splatam-independent-eval-desk-640-full-001/trajectory-errors.png)

整段 SE(3) 对齐后的 ATE 略低，而不额外对齐的首相机参考系误差反而从 4.368 cm 变为 4.726 cm。两者允许的坐标调整不同，说明不能只挑一个指标宣称“分辨率越高定位越好”。本次没有重复 seed、置信区间或显著性检验；约 3.04% 的 ATE 变化只能描述这两次运行。地图 4.349 倍、墙钟 1.927 倍则是实际成本变化，墙钟包含最终图像评测和模型保存。

| 各自估计相机视角、各自像素网格的 30 个已观测样本 | desk 320×240 | desk 640×480 |
|---|---|---|
| 平均 PSNR | 22.898804 dB | 22.872328 dB |
| 平均 MS-SSIM | 0.940193 | 0.907941 |
| 平均 LPIPS | 0.126026 | 0.155412 |
| 有效 GT 像素上的平均深度 MAE | 0.032347 m | 0.031496 m |

上表来自真实原生评测，但两列的像素网格、掩码和估计视角不同，PSNR 相差 −0.026476 dB 不能直接解释为相同像素集合上的地图质量变化。MS-SSIM 与 LPIPS 也受到尺度和输入细节影响。为统一比较条件，已实际完成下面的固定地图重渲染。

![desk 640×480 第 39 帧：30 个固定采样中的最低 PSNR](../outputs/splatam-followup-001/desk-640-full-lowest_psnr.png)

![desk 640×480 第 579 帧：固定采样最后一帧](../outputs/splatam-followup-001/desk-640-full-last_sample.png)

第 39 帧 PSNR 为 18.211582 dB，显示器边缘和桌面细节存在差异；第 579 帧按最后采样位置选取，恰好也是该运行中最高 PSNR 样本，30.187307 dB，不能代表平均效果。两张展示图是上游原图的字节一致副本，来源和哈希见[项目汇总](../outputs/splatam-followup-001/metrics.json)。原图深度 L1 标题四舍五入到两位小数，`0.00` 不表示误差为零；silhouette 自动灰度显示不能提供精确覆盖率。

## 同网格、同 GT 视角：固定地图的另一种验收

两份 desk 最终地图都使用同一个 **320×240 网格、公开 K、30 个采样帧、GT 相机视角、参考 RGB／depth 和有效深度掩码**。地图参数冻结，保持第一相机参考系，没有对任一地图单独应用 SE(3) 配准。GT 位姿只在评测内存中转换为 `w2c/wxyz` 后替换相机数组；没有修改保存的 NPZ，也没有继续训练或调参。统一评测相机不修复建图时已经写入的地图误差。

| 共同 GT 视角，30 个已观测帧均值 | desk320 地图 | desk640 地图 |
|---|---|---|
| PSNR，越高越好 | 14.700114 dB | 14.663791 dB |
| MS-SSIM，越高越好 | 0.599564 | 0.594832 |
| LPIPS，越低越好 | 0.265900 | 0.260763 |
| 有效 GT 像素上的深度 MAE，越低越好 | 0.0859456415 m | 0.0850276509 m |

640 地图的 LPIPS 和深度 MAE 略低，PSNR 与 MS-SSIM 略低；不存在全面改善。图像比较现在使用同样像素集合和视角，但仍只覆盖已观测视角、每种配置一次 seed 0 运行。实际 GPU worker 均退出 0，分别用时 28.949 s 和 26.407 s，控制器退出 0；[完整指标与哈希](../outputs/splatam-common-grid-001/metrics.json)和[共同视角报告](../outputs/splatam-common-grid-001/report.md)保存全部逐帧 raw 指标与模型／输入／源码不变检查。

原估计视角下 desk320 平均 PSNR 为 22.898804 dB，换成共同 GT 相机条件后是 14.700114 dB。为检查下降是否来自保存模型加载或评测封装错误，另做了必要正控制：完全保留原估计相机数组，使用同一冻结地图、320×240 loader 和官方 eval 重渲染 30 帧。真实 worker 与控制器均退出 0，用时 26.934 s；`psnr.txt、ssim.txt、lpips.txt、l1.txt、rmse.txt` 每份 30 个数值与冻结基线逐值比较，**五份最大绝对差均为 0.0**，精确复现原 PSNR 22.898804 dB。

这项控制证明该加载与重渲染入口能够复现原结果。相机条件改变后的落差是真实观察，提示需要检查地图与估计位姿之间的拟合关系；它尚未分离地图形变、累计位姿误差、遮挡和局部渲染残差，不能仅凭这些数字确定全部原因。[正控制数值与收据](../outputs/splatam-common-grid-001/positive-control.json)、[正控制入口](../scripts/splatam_local/evaluate_common_grid_positive_control.py)和[共同视角入口](../scripts/splatam_local/evaluate_common_grid.py)可逐项核对。官方 eval 在替换 GT 相机后打印的 ATE 约为 0，属于已替换相机的诊断，**不作为定位成绩**；真实定位指标仍来自原估计轨迹的独立 ATE／RPE。

![desk320 冻结地图：共同 GT 视角，第 579 帧](../outputs/splatam-common-grid-001/desk320-common-gt-view-0579.png)

![desk640 冻结地图：共同 GT 视角，第 579 帧](../outputs/splatam-common-grid-001/desk640-common-gt-view-0579.png)

两张图按最后一个固定采样帧 579 选择，没有按质量筛选，副本与实际原图哈希相同。RGB 为官方量化 PNG，depth 为着色可视化；它们适合检查空间错位和局部差异，数值结论使用未舍入的 raw 指标。

## 保持哪些条件才能解释差异

三组使用固定 SplaTAM 提交 `da6bbcd24c248dc884ac7f49d62e91b841b26ccc`、原始 depth renderer `cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110`，同一个本机 WSL／RTX 5080 环境。后续帧 tracking 200 次、每帧 mapping 30 次、每 5 帧关键帧、窗口最多 20 帧，损失权重、优化器、isotropic 初始化、新点插入和 pruning 沿用官方配置。逐帧 GT tracking 关闭，不开启 BA、回环或梯度 densification。

输入使用相同 loader：RGB 分别匹配最近 depth 和 GT，两种时间差都小于 0.08 s，再保留距上一已保留 RGB 大于 1/32 s 的观测。完整运行均从第 0 个关联帧开始，`stride=1`，不另行删除复用 depth 的帧。desk 592 帧使用 549 张独立 depth；xyz 759 帧使用 750 张独立 depth。输入列表与图片哈希和关联结果在运行前后核对，[公开数据审核](../outputs/splatam-dataset-audit-001/metrics.json)记录来源、许可和完整口径。

完整序列统一 `eval_every=20`、checkpoint 间隔 100。最终图像评测采样首帧及第 19、39、59……帧，因此 desk 为 30 个已观测采样视角，xyz 为 38 个。所有配置帧都参与轨迹评测；图像指标不覆盖所有帧。最终模型、有效配置、原始日志和真实退出收据保存在唯一运行目录，独立评测按各自序列重建 GT。

seed 0 固定初始随机数设置，方便追溯；它不能保证不同分辨率具有相同随机抽样过程或 CUDA 的逐位确定性。图像大小、高斯数量和运算路径改变后，随机数消费和浮点计算也可能改变。本轮仍是每个配置一次运行，暂时没有多 seed 方差。

## 分辨率怎样改变优化问题

从 320×240 到 640×480，像素数变为四倍。公开 Freiburg1 的原始内参为 `fx=517.3、fy=516.5、cx=318.6、cy=255.3`；320×240 时四项分别乘 0.5，640×480 使用原值。loader 对颜色采用线性缩放，对深度采用最近邻，深度始终除以 5000 得到米。固定代码没有应用 `crop_edge` 裁剪，不能额外减去主点偏移。

较高分辨率保留更多纹理和深度边缘，也会增加首帧有效反投影点和投影计算。projective 初始化尺度与焦距有关：同一深度下，像素对应的空间范围变小，高斯数量与尺寸会一并改变。因此这项对照测量的是官方流程在不同输入分辨率下的整体行为，不能只归因于“纹理更多”或“像素算得更多”。

Tracking RGB／depth 项采用有效掩码中的残差求和，Mapping 项主要采用均值。同样的 tracking 权重和迭代数，在分辨率增加后并不意味着优化器看到相同数量、相同总量的残差。有效深度、silhouette 和深度差掩码还会随缩图改变。若高分辨率结果改善或恶化，需要同时检查轨迹、最终地图、每帧异常与原始日志，不能从损失总值直接判断精度。

四倍像素不保证四倍耗时或四倍显存。高斯数量、投影覆盖、地图插入与裁剪、窗口抽样以及最终图像评测都影响成本；显存还包含缓存和其他进程的占用。处理时间采用实际官方子进程从启动到退出的墙钟，包含初始化、tracking／mapping、最终采样评测和保存，排除环境安装、Windows 镜像与独立评测。单次 `nvidia-smi` 采样只能称为该时刻占用；间隔采样的最大值应称为“采样最大值”，不能称作精确峰值。未测量的显存数据不填入表格。

## 序列比较怎样避免过度结论

xyz 是 TUM 官方提供的平移运动序列，含 759 个关联后观测，跨度 26.572 s；desk 为 592 帧、19.843 s。两条序列的视角、纹理、运动和深度条件一起变化；同为 Freiburg1 不等于只有运动不同。它们可以揭示同一配置在另一条公开序列上是否稳定运行、误差如何变化，却不是隔离某个模块或某种运动影响的消融实验。

不同序列的相邻帧 RPE 还受真实帧间运动和采样间隔影响。这里报告每个有效相邻观测对的平移 RMSE 与旋转 RMSE，不把“每帧误差更小”直接转写成“每米漂移更小”。ATE 使用全部配置帧的 `c2w` 相机中心，SE(3) 对齐且尺度固定为 1；另保留首相机参考系下未经整段额外对齐的误差。原始官方 ATE 标签为 RMSE、实际为另一种误差均值，不能混入独立表格。

20 帧 smoke 和 592 帧 full 不是分辨率对照，它们覆盖的时间区间和评测样本不同。两个完整序列加一组分辨率对照也没有验证多 seed 鲁棒性、户外无人机场景、其他相机、物理标定或硬件同步。

## 图像质量与轨迹精度分别看

最终地图的 PSNR、MS-SSIM 与 LPIPS 来自已经进入训练／建图的采样视角，属于已观测视角拟合；它们不代表留出测试、新视角或新场景泛化。深度指标按源码实际计算标为 **MAE，单位 m**，不照抄上游的 RMSE 标签。颜色在无效 GT depth 处置零后仍按整幅图计算，深度仅使用有效 GT 像素，必须保留该口径。

desk 的两种分辨率可以采用相同帧索引，但原分辨率 PSNR 的像素网格和有效深度掩码不同。未经统一尺度重新评测，数值只能描述各自配置的输出质量，不能视为相同像素集合上的严格逐像素增益。轨迹采用同样关联帧与物理单位，分辨率对照中的 ATE／RPE 可以按同一口径阅读。

平均质量和较差样本应一起查看。展示图来自真实最终输出，并记录帧号、选择规则和哈希；不能只选最高 PSNR 来代表整条序列。上游 silhouette 图使用自动灰度归一化，纯黑可能来自常数掩码，不能仅根据显示颜色断言没有地图覆盖。

## 本机复现入口

```powershell
cd (Join-Path ([Environment]::GetFolderPath('Desktop')) 'GS-SLAM-Lab')
.\.venv\Scripts\python.exe scripts/splatam_local/run_wsl.py --sequence freiburg1_xyz --frames -1 --width 320 --seed 0
.\.venv\Scripts\python.exe scripts/splatam_local/run_wsl.py --sequence freiburg1_desk --frames -1 --width 640 --seed 0
```

两个 GPU 实验依次运行，避免并行占用造成时间和显存口径混杂。独立评测的 `--sequence` 必须指向该模型实际使用的数据目录；运行完成和模型保存并不等于独立验收已经通过。[入口与输出说明](../configs/splatam/LOCAL_PROJECT.md)解释有效配置、唯一目录和 Windows 镜像，[本轮证据报告](../outputs/splatam-followup-001/report.md)记录实际完成状态与结果来源。

当前已建立两条完整公开序列、同序列两种分辨率的真实结果。[项目入口清单](../manifests/splatam-followup.json)与[汇总脚本](../scripts/summarize_splatam_followup.py)绑定实际模型、配置、源码、独立指标及图像副本。该证据链支持项目答辩中的分辨率、优化成本和累计误差解释；单 seed、两条同传感器室内序列仍不覆盖完整产品级 SLAM 的回环、重定位或实时部署。
