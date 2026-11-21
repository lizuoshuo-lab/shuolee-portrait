---
id: "gslab-day02"
slug: "day02-gaussian-count"
title: "Day 02｜3 小时：4000 与 8000 高斯的单变量实验"
order: 2
status: "planned_pending_run"
updated: "2026-10-08"
sources:
  - "https://github.com/nerfstudio-project/gsplat/tree/v1.5.3"
  - "https://github.com/bmild/nerf/blob/master/tiny_nerf.ipynb"
---

# 用一组可解释的对照实验积累项目经验

问题：在相同相机、训练划分、1500 步和 seed 0 下，把高斯数从 8000 降至 4000，会如何影响留出视角质量、细节与训练资源？先写预测，再运行。首轮 8000 高斯的实测基线已保存在 [训练记录](../experiments/2026-10-08-local-gsplat.md)，4000 高斯尚未运行，不预填结果。

## 180 分钟安排

| 时间 | 动作 | 产物 |
| --- | --- | --- |
| 0–40 分钟 | 阅读相机与投影笔记，解释脚本中的 `poses_gl @ convention`、求逆、`scales.exp()` 与 `opacity.sigmoid()` | 一页自己的说明；标明哪些参数有梯度 |
| 40–90 分钟 | 执行 4000 高斯实验，比较原基线的六个相同留出视角 | 新运行目录及指标表 |
| 90–130 分钟 | 对照细杆、车轮与底座，找出渲染误差及可能原因 | 三条带图或视角编号的观察 |
| 130–160 分钟 | 区分模型容量、优化未收敛与初始化影响；提出下一项控制变量 | 一个假设及可否证条件 |
| 160–180 分钟 | 写实验记录并口述三分钟 | “问题—控制量—结果—限制—下一步”的说明 |

## 运行入口

在实验室根目录打开 PowerShell，使用未使用过的 run 名：

```powershell
.\.venv\Scripts\python.exe scripts\run_local.py scripts\train_tiny_gs.py --points 4000 --steps 1500 --seed 0 --run tiny-nerf-g4000-s0
```

重复实验须修改 run 名，脚本不会覆盖已有目录。基线是 `outputs/tiny-nerf-gsplat-001/metrics.json`，对照输出是 `outputs/tiny-nerf-g4000-s0/metrics.json`。训练 100 视角，留出 6 视角；保持数据与脚本相同，不调整分辨率、步数和学习率。

| 比较项 | 8000 高斯实测 | 4000 高斯 |
| --- | --- | --- |
| 留出视角平均 PSNR | 27.4966 dB | 待测 |
| 优化循环耗时（排除编译/最终评测） | 3.4979 s | 待测 |
| PyTorch allocated 显存峰值 | 23.3013 MiB | 待测 |
| 图像细节 | 车辆轮廓恢复，细节仍模糊 | 待观察 |

耗时来自极小教学数据，单次几秒的差异只作探索；显存数字不是进程总显存。高斯数改变会改变随机初始化张量与后续优化过程，固定 seed 不意味着两个实验具有逐粒对应的初始化。若要提出稳定的优劣结论，应对每个点数运行相同的一组额外种子，再报告均值与波动，当前先学习一次完整对照。

## 验收问题

- 训练图像与留出图像分别在代码哪里被使用？六张留出图不能用于挑选超参数后又声称是独立最终测试；若持续据此调参，应改称验证集并预留最终测试。
- 相机有没有进入 optimizer？当前为什么不具有定位能力？
- 如果训练质量提升但留出质量下降，下一步如何排查？
- 为什么这个实验不能直接证明无人机大场景实时 SLAM？

完成后在 `experiments/` 新建实验记录，包含真实配置、来源、指标、图像引用与解释，更新 `manifests/progress.json`，运行 `scripts/export_site.py`。格式参考现有实测记录，保留稳定 id/slug；网站此时仍只得到本地内容包。
