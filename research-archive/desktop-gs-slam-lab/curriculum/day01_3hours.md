---
id: "gslab-day01"
slug: "day01-local-first-three-hours"
title: "Day 01｜3 小时：本地高斯拟合与已知位姿建图的第一步"
order: 1
status: "baseline_executed_user_practice_pending"
updated: "2026-10-08"
sources:
  - "https://github.com/nerfstudio-project/gsplat/tree/v1.5.3"
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats"
  - "https://github.com/spla-tam/SplaTAM"
---

# Day 01：今天只推进一个能解释清楚的本地实验

**每天先打开本页，再按时间表读对应笔记。** 主目录为 `C:\Users\Administrator\Desktop\GS-SLAM-Lab`。学习材料放在 knowledge，实验入口与实测记录以主目录 README、scripts、experiments、logs 和 outputs 中当前已验证内容为准。

已选路线：本地单图拟合 → 已知位姿多视图建图 → 后续 tracking/SLAM。当前 RTX 5080、PyTorch 2.7.1+cu128、gsplat v1.5.3 的 CUDA 前后向和两个训练实验均已实际完成。单图 PSNR 20.17 dB；多视图 6 个留出视角平均 PSNR 27.50 dB。先读 [本次实测记录](../experiments/2026-10-08-local-gsplat.md)，再按下表理解与复做。已有结果不覆盖，复做时使用新 run 名。用户阅读与掌握情况尚待自测。

## 180 分钟导航

| 时间 | 要做的事 | 阅读/入口 | 结束时应留下什么 |
|---|---|---|---|
| 00–15 分钟 | 查看当天已验证启动入口；确认环境、设备、代码版本、输入和输出位置 | 主目录 README 与 logs；[知识索引](knowledge_index.md) | 一条能复查的启动命令；明确此刻哪些环节已测、哪些未测 |
| 15–50 分钟 | 画出高斯到像素的链路，复算两层 alpha 合成；确认单图实验优化哪些参数 | [02 高斯投影与合成](../knowledge/02_gaussian_projection_compositing.md) | 一张手绘流程图或 5 行说明；合成结果 `(0.5,0,0.25)` |
| 50–105 分钟 | 运行本地最小图像拟合；保存初始与最终结果、loss 和日志；只做一次小改动 | 本地已验证 gsplat 示例/适配脚本；不猜网络上的 main 分支命令 | 初始/最终图像、实际迭代数、loss、耗时；失败则留下最小可复现日志 |
| 105–150 分钟 | 手算 w2c/c2w 与深度；准备少量已知位姿多视图，检查 K、位姿、正深度与数据大小 | [01 相机几何](../knowledge/01_camera_geometry.md)；当前数据准备脚本 | 数据小样本与约定说明；若训练已经可跑，保存一个小规模结果 |
| 150–170 分钟 | 区分单图拟合、已知位姿建图、tracking；核对固定量和优化量 | [03 优化变量与边界](../knowledge/03_tracking_mapping_scope.md) | 一张参数表；相机是否参与 optimizer 的明确结论 |
| 170–180 分钟 | 写下今天完成/失败的证据，口述 3 分钟；给明天只留一个问题 | [04 评测协议](../knowledge/04_evaluation_protocol.md) 的开头与计时部分 | 日志/图片路径、一个已解决问题、一个待解决问题 |

## 数据已就绪，按实验选用

- Tiny NeRF：官方教程 NPZ（约 12.7 MB）已下载校验；106 个 100×100 视角，0–99 用于训练、100–105 留出评测。先用它理解共享高斯表示与多视图约束；没有深度监督或相机估计。
- TUM fr1_desk：完整官方 RGB-D 包（约 344 MB）已下载校验，613 张 RGB、595 张深度。单帧反投影与坐标变换检查已完成，见 [几何记录](../experiments/2026-10-08-rgbd-geometry.md)。下一阶段才扩展为短序列建图。
- 两份数据的来源、SHA256、许可与目录记录在 `manifests/datasets/`。原始数据保留本地，网站内容包只导出文档及结果引用。

今天不要求读完 SplaTAM 全仓库、不要求跑完整 TUM 序列、不追论文指标。SplaTAM 是后续把地图优化扩展到 tracking 的参考；当前实验名称按实际能力填写。

## 一次实验至少记这些

```text
实验日期与名称：
代码仓库、tag/commit、启动命令：
设备、PyTorch/CUDA、rasterizer 版本：
输入文件与来源；分辨率；帧数：
固定参数 / 优化参数：
迭代数、随机种子：
初始 loss / 最终 loss：
耗时及计时范围：
输出图片 / checkpoint / 日志路径：
观察到的变化或错误：
本实验能证明什么、尚不能证明什么：
```

这些字段是待填写模板，没有填写的数值保持空白或标“未测”，不要用论文数字替代本机结果。

## 今天的通过条件

- [ ] 有一个可复查的本地拟合运行记录，或明确定位到编译/数据/接口哪一层失败。
- [ ] 能指出高斯参数、相机参数、loss 和 optimizer 分别在哪里。
- [ ] 手算 w2c/c2w、米制深度和 alpha 合成，并能解释计算结果。
- [ ] 能说清“我完成单图拟合/已知位姿建图，尚未验证 tracking”。
- [ ] 保存真实输出和下一步问题；没有把渲染 FPS 写成 SLAM FPS。

## 明天的进入条件

基线与相机代数检查已经通过。理解当前脚本的固定量、优化量与评测划分后，进入 [Day 02｜高斯数量的单变量实验](day02_gaussian_count.md)。若相机坐标或深度概念仍不确定，先完成手算再修改实验。

## 当天状态

环境、数据、基线训练及输出：已由助手实际执行并记录。用户阅读、手算、独立复做与面试自述：待完成。以上勾选项用于用户自测，不自动替用户标记已掌握。
