---
id: "gslab-exp-renderer-semantics-20261009"
slug: "gaussian-renderer-feature-depth-gradient-controls"
title: "gsplat 渲染控制：累积深度、覆盖、二阶矩与六维相机梯度"
order: 9
status: "completed_synthetic_gsplat_controls_original_backend_not_run"
updated: "2026-10-09"
sources:
  - "https://docs.gsplat.studio/main/apis/rasterization.html"
  - "https://github.com/nerfstudio-project/gsplat/blob/937e29912570c372bed6747a5c9bf85fed877bae/gsplat/rendering.py"
  - "https://github.com/nerfstudio-project/gsplat/blob/937e29912570c372bed6747a5c9bf85fed877bae/gsplat/cuda/_wrapper.py"
  - "https://github.com/nerfstudio-project/gsplat/blob/937e29912570c372bed6747a5c9bf85fed877bae/gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu"
  - "https://github.com/nerfstudio-project/gsplat/blob/937e29912570c372bed6747a5c9bf85fed877bae/gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu"
  - "https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/utils/slam_helpers.py"
---

# gsplat 特征、深度和相机梯度的合成控制

gsplat 1.5.3 的 GPU 实测支持三项结论：零背景下渲染 `[z,1,z²]` 得到累积深度、覆盖和加权二阶矩；内置 ED 是按 alpha 归一化的深度；直接优化 viewmat 与把高斯变换到相机系后渲染，在本实验中具有一致的图像与六维梯度。原始 SplaTAM renderer 没有执行，结果只比较同一个 gsplat 后端内的语义与参数化，不能证明跨后端数值等价。

成功运行保存为 `renderer-feature-semantics-002`。首轮 `001` 的负控制设计不足而失败，证据保留，没有覆盖历史输出。没有训练地图、读取真实数据集或评估 SLAM 轨迹。

## 固定版本、运行与可审阅证据

| 项目 | 实际设置 |
| --- | --- |
| GPU / PyTorch | RTX 5080 / 2.7.1+cu128 |
| gsplat | 1.5.3，commit `937e29912570c372bed6747a5c9bf85fed877bae` |
| 图像 / 内参 | 32×32，fx=fy=24，cx=cy=15.5 |
| 中心采样 | 数组像素 y=x=15，采样位置 15.5，恰好命中投影中心 |
| 背景 / 特征 | 零背景；原值 `[camera_z,1,camera_z²]`，不做 sigmoid 或裁剪 |
| 渲染配置 | `packed=False`、`rasterize_mode="classic"`；分别 `RGB`、`RGB+D`、`RGB+ED` |
| 公式比较容差 | 最大绝对误差 `5e-6` |
| 六维中心有限差分 | 每个 twist 分量正负 `1e-4`，旋转分量用弧度 |
| 总耗时 | 1.1671 秒，含首轮渲染与所有控制检查；不是 SLAM FPS |

运行入口为 `scripts/run_local.py scripts/check_renderer_feature_semantics.py --run renderer-feature-semantics-002`，配置与源码哈希记录在文件中，不安装新依赖。合成空间未标定物理米制。

核心证据是可直接读取的小 JSON：[特征与手算比较](../outputs/renderer-feature-semantics-002/feature_semantics.json)、[完整指标](../outputs/renderer-feature-semantics-002/metrics.json)、[配置](../outputs/renderer-feature-semantics-002/config.json)、[源码版本](../outputs/renderer-feature-semantics-002/source.json)。[完整逐像素实际输出 JSON](../outputs/renderer-feature-semantics-002/actual_outputs.json)约 777 KB；`raw_outputs.npz` 只提供本地数组读取便利。

[002 运行源码快照](../outputs/renderer-feature-semantics-002/source_script.py)的 SHA256 为 `5217a9e9c54784dab770d88e0a101e05984c9b4e0a4722777f378d02a72394c6`。两个运行各自保存源码，避免之后修改实验脚本使历史结果失去对应实现。此实验调用已有 `se3_exp` 与协方差 helper，helper 的哈希也记录在 `source.json`；有限差分由本实验脚本重新渲染计算。没有修改旧 helper 或在线脚本。

## 控制一：单高斯的覆盖恰好为 0.5

单个高斯中心为 `[0,0,2]`，各轴 scale=0.18，opacity=0.5。投影中心像素的高斯指数因子为 1，故 alpha 权重 w=0.5。在零背景下，手算为：

```text
S  = w        = 0.5
D  = w·z      = 1
M2 = w·z²     = 2
ED = D/S      = 2
M2−D²         = 1
M2/S−(D/S)²   = 0
```

实际中心输出 `[D,S,M2]=[1,0.5,2]`，内置 D=1、ED=2，与手算完全一致。严格 mask `S>0.5` 为 false；`S>=0.5` 为 true。这个边界不是描述性近似，实际 float32 中心 alpha 保存为恰好 0.5。

M2−D² 仍为 1，虽然单个中心深度的条件方差为 0。缺少覆盖归一化时，不能把这个量解释为“只对覆盖表面计算的方差”。M2 是高斯中心 z 的合成二阶矩，也没有表示单个高斯沿射线的体内方差。

## 控制二：两个同中心投影高斯的遮挡权重

两个高斯中心分别为 `[0,0,2]`、`[0,0,4]`，各轴 scale 分别 0.18、0.36，opacity 分别 0.5、0.7。尺度随深度同比例增加，使投影足迹匹配；中心像素先合成近处，再合成远处。

```text
w1 = 0.5
w2 = (1−0.5)·0.7 = 0.35

S  = 0.5 + 0.35         = 0.85
D  = 0.5·2 + 0.35·4     = 2.4
M2 = 0.5·4 + 0.35·16    = 7.6
ED = 2.4 / 0.85         = 2.8235294118
M2−D²                   = 1.84
M2/S−(D/S)²             = 0.9688581315
```

| 中心像素量 | 实测通道 / 由实测通道计算 | 手算 |
| --- | ---: | ---: |
| S / alpha | 0.8500000238 | 0.85 |
| D / 内置 D | 2.4000000954 | 2.4 |
| M2 | 7.5999999046 | 7.6 |
| 内置 ED | 2.8235294819 | 2.8235294118 |
| M2−D² | 1.8399994469 | 1.84 |
| 条件方差 | 0.9688575821 | 0.9688581315 |

中心 S、D、M2 与手算最大误差为 `9.54e-8`。M2−D² 与条件方差是把保存的 float32 通道转为 Python float 后计算的派生量，并非内核直接输出的方差通道。D 不是平均两个深度，也不是 opacity 直接加权之和；远处贡献必须乘近处留下的透射率。此处 `S>0.5` 为 true。

两个控制分别比较全图而非只比较中心：

| 全图比较最大绝对误差 | 单高斯 | 双高斯 |
| --- | ---: | ---: |
| 常数 1 特征通道 vs alpha | 2.98e-8 | 5.96e-8 |
| z 特征通道 vs 内置 D | 0 | 0 |
| 内置 ED vs `D/max(alpha,1e-10)` | 0 | 0 |
| RGB 特征 vs RGB+D 的前三通道 | 0 | 0 |
| RGB 特征 vs RGB+ED 的前三通道 | 0 | 0 |

实际最大原值特征分别为 2、7.6，均超过 1。若包装器把这些值当颜色裁剪，以上控制会失败。分母 clamp 避免空像素除零，但空像素的 ED 没有因此成为有效几何深度。

## 控制三：两条相机路径使用同一个标量 loss

为避免对称场景掩盖旋转错误，使用三个非对称中心、各向异性尺度与非单位高斯旋转。完整几何与 wxyz 四元数记录在 `metrics.json`。目标由合成 twist `[0.03,−0.02,0.04,1°,−2°,1.5°]` 的渲染产生，只有此合成控制使用该目标；它不是数据集真值评测。

相机 twist 顺序为 `[ρ_x,ρ_y,ρ_z,ω_x,ω_y,ω_z]`，实际刚体矩阵由 `exp(ξ^)` 生成；ρ 是李代数平移分量，在零扰动处与平移的一阶导数一致，有限旋转时实际 t 还经过 SE3 指数映射。直接路径渲染世界中心与协方差，viewmat=W；另一条路径使用 `μ_c=Rμ_w+t`、`Σ_c=RΣ_wRᵀ`，viewmat=I。两者都用相机系中心的 z 构造 `[z,1,z²]`，保留 z 的计算图。只变换中心而忽略各向异性协方差不构成等价路径。

两条路径在零 twist 处比较相同标量：

```text
L = mean(((rendered_features−target_features) / [4,1,16])²)
```

| 比较 | 实际值 |
| --- | ---: |
| 直接 viewmat 的 loss | 0.0005994620733 |
| 相机系中心/协方差的 loss | 0.0005994620151 |
| 图像最大绝对差 | 7.1526e-7 |
| loss 绝对差 | 5.8208e-11 |
| 解析六维梯度相对 L2 差 | 8.3860e-8 |

直接路径的解析六维梯度约为 `[0.008530882,0.008341553,−0.000054395,−0.022928068,0.023789529,−0.000662428]`。两个路径的六个分量均有限且非零，各自重新渲染正负扰动，获得中心有限差分：

| 路径 | 解析 vs 有限差分方向余弦 | 相对 L2 误差 |
| --- | ---: | ---: |
| 直接 viewmat | 0.999996662 | 0.002631126 |
| 相机系中心/协方差 | 0.999996424 | 0.002635364 |

所以证据包含逐像素比较、相同标量残差和六维方向检查，不是按 shape 或 `.grad is not None` 判定成功。被测位置是零 twist 及中心有限差分 `1e-4` 的邻域；非零 target pose 仅构造残差，不能据此宣称任意非零被测 pose 两路径等价。有限差分结果受 float32 与步长影响；此小场景没有刻意跨越排序、裁剪或可见性不连续边界。

## 首轮失败：任意 1% 阈值不能隔离特征梯度

`renderer-feature-semantics-001` 先用上述混合图像 loss 检查 detach z、z² 后的总梯度变化，设定“相对变化至少 1%”的门槛。001 留存证据证明该门槛处断言失败，没有保存具体变化值。002 复测相同混合 loss，记录相对变化约 0.16864%；这个数值归属于 002，不追记到 001。几何投影、协方差和覆盖都能产生位姿梯度，混合 loss 中这部分占据主要贡献；特征路径断开，整体梯度仍可能看似正常。任意比例门槛不能稳定识别目标错误。

保留的[失败记录](../outputs/renderer-feature-semantics-001/failure.json)、[当时源码快照](../outputs/renderer-feature-semantics-001/source_script.py)、[源码哈希](../outputs/renderer-feature-semantics-001/source.json)与本地归档日志 `logs/renderer-feature-semantics-001.log` 对应这次失败。早先的公式和两路径检查已运行到该断言，但初版尚未保存全部实际数值；不能把首轮列成完整成功，也没有用第二轮结果补写首轮历史。

第二轮保存自己的混合 loss 小幅变化，同时设计可解析、能隔离目标路径的负控制，未通过降低原阈值来强行完成。

## 控制四：隔离中心 ED 的 z 特征梯度

使用单高斯中心像素，`ED=D/S=z_camera`。标量 loss 取 `(ED−2.2)²`，在零位姿、z=2 处：

```text
∂L/∂t_z = 2·(2−2.2)·1 = −0.4
```

同一高斯的覆盖权重在 D/S 中相消，几何权重路径不能替代 z 特征的导数。该控制有明确预期，不依赖另一个同构实现给答案。

| 检查 | 实际输出 |
| --- | --- |
| 保留 z 图的解析六维梯度 | `[0,0,−0.4000000954,0,0,0]` |
| t_z 中心有限差分 | −0.3998167813 |
| detach z 图的解析六维梯度 | `[0,0,0,0,0,0]` |
| 两者前向 loss | 均为 0.04000001773 |

detach 时正负扰动的前向数值仍会变化，所以有限差分仍描述原前向函数；autograd 的零梯度暴露的是计算图与前向函数不一致。控制说明包装器不能因为图像数值相同就忽略 z、z² 的梯度连接。

## 结果边界

这些输出是合成控制，坐标为未标定场景单位，z 表示相机轴向深度。加权中心二阶矩不是高斯内部体积深度方差，也不是经标定的不确定度。`S=alpha` 的测试使用零背景；非零背景附加项需要单独处理。

只检查固定 gsplat 版本的 classic、非 packed 调用及连续区域的有限差分。没有评测真实数据、排序变化、dummy means2D 的 NDC 梯度桥接、原后端 median-depth、radii 与增删点行为。原始 SplaTAM renderer 没有运行，不能宣称它与 gsplat 数值等价，更不能把这个 1.17 秒控制实验当作完整 SLAM 性能。

对应的知识推导与固定源码边界见[09 渲染后端语义](../knowledge/09_renderer_backend_semantics.md)。
