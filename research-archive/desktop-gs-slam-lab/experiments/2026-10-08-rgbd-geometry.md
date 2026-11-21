---
id: tum-geometry-001
slug: rgbd-geometry-first-pair
title: "实验 001：TUM RGB-D 单帧几何与坐标约定检查"
order: 1
status: completed
updated: "2026-10-08"
sources:
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats"
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset#license"
  - "https://github.com/spla-tam/SplaTAM/blob/main/configs/data/TUM/freiburg1_desk.yaml"
  - "https://github.com/spla-tam/SplaTAM/blob/main/datasets/gradslam_datasets/tum.py"
---

# TUM RGB-D 单帧几何与坐标约定检查

2026-10-08 已在本机 `.venv` 的 Python 中真实运行，退出码 0。使用 NumPy 和 Pillow，CPU 完成。此实验完成的是**数据读取、反投影和坐标变换的数学一致性检查**；没有训练模型、估计相机运动或运行 SLAM。

## 为什么先做这个检查

RGB-D GS-SLAM 将深度像素初始化成三维点或 Gaussian。深度单位、相机内参、四元数顺序或变换方向错误，会在优化前就破坏几何。这里先用一个有记录的输入，建立可检查的坐标约定。

## 输入与时间戳配对

使用官方 `rgbd_dataset_freiburg1_desk` 原始 TGZ 解压数据，许可为 **CC BY 4.0**。署名：TUM RGB-D Benchmark；J. Sturm、N. Engelhard、F. Endres、W. Burgard、D. Cremers，IROS 2012。

配对规则与本次检查过的 SplaTAM TUM loader 一致：遍历 RGB，找到时间最近的 depth 和 ground truth，两项时间差都严格小于 0.08 秒，取首个满足条件的 RGB。并未直接把三个文件列表的第一行拼成一帧。

| 项目 | 实际输入 |
|---|---|
| RGB | `rgb/1305031453.323682.png`，RGB 索引 16 |
| Depth | `depth/1305031453.374112.png`，depth 索引 0 |
| Ground truth | 时间戳 `1305031453.3195`，GT 索引 351 |
| RGB 与 depth 时间差 | **50.430 ms** |
| RGB 与 GT 时间差 | **4.182 ms** |
| Depth 与 GT 时间差 | **54.612 ms** |

这些输入并非严格同步。首对数据尤其具有约 50 ms 的 RGB-depth 差异；彩色点云不能作为已准确对齐的证据。GT 取最近 RGB 的样本，没有按 depth 时间重新插值或补偿运动。

## 几何约定与实际计算

使用上游 `freiburg1_desk.yaml` 的内参：`fx=517.3, fy=516.5, cx=318.6, cy=255.3`；原图为 `640×480`。深度整数除以 `5000.0` 转成米，零深度无效。本实验不裁边，不进行畸变校正。

相机坐标约定为 X 向右、Y 向下、Z 向前。对像素 `(u,v)` 及深度 `z`：

```text
Xc = (u - cx) * z / fx
Yc = (v - cy) * z / fy
Zc = z
Pw = T_camera_to_world * [Xc, Yc, Zc, 1]
Pc_restored = inverse(T_camera_to_world) * Pw
u_restored = fx * Xc_restored / Zc_restored + cx
v_restored = fy * Yc_restored / Zc_restored + cy
```

GT 按 `tx ty tz qx qy qz qw` 读取；四元数归一化后构建 camera-to-world 变换。计算与误差检查使用 float64；PLY 为节省空间使用 float32 XYZ 和 uint8 RGB。

## 实测结果

| 指标 | 结果 |
|---|---:|
| 原图像素数 | 307,200 |
| 有效深度像素数 | 255,190 |
| 有效深度比例 | **83.0697%** |
| 有效深度最小值 / 最大值 | 0.7378 m / 2.0256 m |
| 深度 p01 / p10 | 0.7770 m / 0.8694 m |
| 深度中位数 | 1.0766 m |
| 深度 p90 / p99 | 1.6194 m / 1.8426 m |
| 点云像素采样步长 | 4 |
| 导出有效点数 | **16,008** |
| 相机→世界→相机最大欧氏误差 | **9.5505e-16 m** |
| 重投影最大欧氏误差 | **3.5951e-13 pixel** |

脚本的通过阈值为回环误差小于 `1e-10 m`、重投影误差小于 `1e-8 pixel`，本次均通过。另行读取导出的 PLY 验证了 16,008 个顶点、240,120 字节顶点数据及全部 XYZ 数值有限。预览图已打开检查：RGB 与深度均成功显示，无效深度显示为黑色，深度显示范围明确标注为 p01 至 p99。

## 这组误差能证明什么

它证明本次代码中反投影、齐次变换和投影运算在设定约定下数值自洽。**变换与自身的逆相消，即便采用错误的物理标定，也可能得到近零误差。** 因此这里不能证明相机内参、深度尺度、传感器时空标定正确，更不能证明跨帧标定或 SLAM 质量。

当前简化还包括：没有补偿 RGB-depth 时间差，没有 GT 插值，没有畸变校正，直接以同像素位置的 RGB 为 depth 点着色；仅处理一帧，没有 ATE、RPE、跨帧对齐或建图评估。

## 产物与复现

- 脚本：`scripts/check_rgbd_geometry.py`
- 完整指标和 pose 矩阵：`outputs/tum-geometry-001/metrics.json`
- 世界坐标彩色点云：`outputs/tum-geometry-001/first_pair_world.ply`
- RGB-depth 预览：`outputs/tum-geometry-001/preview_rgb_depth.png`
- 数据来源与校验清单：`manifests/datasets/tum_freiburg1_desk.json`

在 `GS-SLAM-Lab` 根目录运行：

```powershell
.\.venv\Scripts\python.exe .\scripts\check_rgbd_geometry.py --stride 4
```

为保留实验结果，脚本会拒绝覆盖已存在的 `outputs/tum-geometry-001/`。本页记录的是已经完成的第一次运行，不需要为了检查结果再次运行。

下一步可将这个数据与几何检查接入 SplaTAM 的小帧数测试，再分别检查 tracking、mapping 与最终评测输出；本次结果不应填写为“已经训练或复现 GS-SLAM”。
