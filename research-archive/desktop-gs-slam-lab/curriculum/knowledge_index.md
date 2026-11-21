---
id: "gslab-knowledge-index"
slug: "knowledge-index"
title: "学习导航｜每天 3 小时与四篇基础笔记"
order: 0
status: "ready_for_study_pending_practice"
updated: "2026-10-08"
sources:
  - "https://github.com/nerfstudio-project/gsplat/tree/v1.5.3"
  - "https://github.com/spla-tam/SplaTAM"
---

# 从每天 3 小时开始

**第一入口：[Day 01｜本地拟合与已知位姿建图](day01_3hours.md)。** 按当天实验需要查笔记，不先把所有理论学完再开始。

当前主线：

2026-10-08 实测入口：[两项 GPU 训练](../experiments/2026-10-08-local-gsplat.md)、[TUM 几何检查](../experiments/2026-10-08-rgbd-geometry.md)。这些实验已经完成；下表状态记录用户学习进度。完成第一天自测后进入 [Day 02 单变量实验](day02_gaussian_count.md)。

```text
本地环境 + rasterizer 可用
        ↓
单图拟合：参数 → 渲染 → loss → 梯度
        ↓
相机/位姿/单位正确 + 已知位姿多视图建图
        ↓
保留视角评测 + 单变量实验
        ↓
固定地图估计单帧位姿
        ↓
Tracking / Mapping 交替，形成 SLAM 项目
```

## 按问题查笔记

| 顺序 | 笔记 | 遇到什么问题时读 | 当前状态 |
|---|---|---|---|
| 1 | [相机内外参、深度、缩放](../knowledge/01_camera_geometry.md) | 地图翻转、尺度异常、点在相机背后、缩图后无法对齐 | 来源已读；用户学习与实测待完成 |
| 2 | [高斯参数、投影、合成](../knowledge/02_gaussian_projection_compositing.md) | loss 怎么来的、为什么颜色相加不对、高斯变大但图更糊 | 来源已读；用户学习与实测待完成 |
| 3 | [Tracking / Mapping 变量与开源边界](../knowledge/03_tracking_mapping_scope.md) | 相机到底有没有被优化、代码“跑通”能否写成 SLAM 项目 | 来源已读；用户学习与实测待完成 |
| 4 | [ATE、RPE、MS-SSIM、FPS](../knowledge/04_evaluation_protocol.md) | 指标对不上论文、`ssim.txt` 含义、如何诚实报告速度 | 来源已读；用户学习与实测待完成 |

每篇都有原理、官方代码关联、手算/小实验、面试自测和状态。相机与高斯两篇是互补基础；第一天为了尽快看见优化结果，可以先读 02，再按数据需要回到 01。

## 状态规则

- `sources_read_pending_practice`：笔记中的指定官方材料已阅读核对；不等于用户已经掌握或项目已经运行。
- `planned_pending_run`：当天计划已写好，运行记录尚待填写。
- 用户阅读状态用正文勾选项/文字维护；实际运行状态只由 logs、输出文件和实验记录支持。
- 编译成功、单图拟合成功、已知位姿建图成功、tracking 成功、完整 SLAM 通过是不同里程碑，分别记录。
- 每次实测补上代码 tag/commit、命令、数据来源、配置和输出路径；不靠记忆填写质量或速度数字。

## 之后如何扩展

主项目先完成小规模 baseline、保留视角评测和一次单变量实验。之后根据岗位只选一个分支：无人机多传感器定位、大场景分块/LOD，或动态街景仿真。VGGT 等前沿知识暂用“问题—方法—限制—与本项目连接点”一页笔记，不挤占每天的实验时间。

## 文档与后续网站同步约定

文档以 UTF-8 Markdown 保存，YAML frontmatter 使用稳定的 `id` 和 `slug`，不因标题润色而改变它们。`order` 控制导航排序；`updated` 是内容更新时间；`sources` 只放实际参考的公共源。正文相对链接用于本地知识导航，未来网站同步应按稳定 id/slug 转换链接。

当前仅创建本地材料，未创建网站或云端 Page。没有使用私有 EPIGS 资料。
