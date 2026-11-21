# 本机官方 SplaTAM 项目入口

项目第一优先为 SplaTAM。固定提交 `da6bbcd24c248dc884ac7f49d62e91b841b26ccc`，原始 depth rasterizer 为 `cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110`。已有 Windows gsplat 教学实验和此项目分别记录，不能相互替代。

隔离环境在 WSL Ubuntu 20.04 的 `/opt/gs-slam-lab/splatam`，位于 D 盘虚拟磁盘。`env.sh` 只设置本项目子进程的 CUDA、库路径和模型权重缓存；不依赖或修改 FAST-LIVO2 的 ROS 环境。不安装 Linux 显卡驱动。

## 运行

在桌面实验目录，用 `.venv/Scripts/python.exe` 调用：

```text
scripts/splatam_local/run_wsl.py --frames 20 --width 320
scripts/splatam_local/run_wsl.py --frames -1 --width 320
```

第一条是 desk 前 20 个官方关联帧，第二条是完整 desk 592 帧。每次生成唯一输出目录，不覆盖旧结果。`--width 640` 可用原始分辨率，但资源消耗和输出不同，需单独评测。`--sequence freiburg1_xyz` 使用 xyz 的 759 个官方关联帧，显式复用同一 Freiburg1 公开相机参数。

Tracking 200 / mapping 30 迭代、损失、优化器、关键帧、插入与 pruning 沿用固定官方配置；GT tracking 关闭。调整的配置包括本机路径、分辨率、保存 checkpoint、图像评测采样频率、禁用 WandB 和 GUI。没有云上传。

## 结果与验收

真实原始运行保存在 Linux `outputs/runs/<unique-name>`。Windows 入口把 final `params.npz`、有效配置、真实退出收据、sensor YAML、环境验收和日志复制到桌面 `outputs/splatam-official-runs-001/<unique-name>`，便于独立评测。Checkpoint 留在 Linux；它不是带全部优化器状态的恢复点。

```text
scripts/splatam_local/evaluate_run.py --run-dir <桌面结果目录> --sequence data/tum/rgbd_dataset_freiburg1_desk --output outputs/<新评测目录>
```

评测强制比对原生训练前后输入与 Windows 原始数据 SHA256、有效配置、source before/after、模型 hash、真实 child exit=0、禁用 GT tracking、完整配置帧数、公开 K、独立原始 GT。轨迹使用 camera-to-world，尺度固定为 1，报告 rigid SE(3) ATE RMSE 和相邻帧平移／旋转 RPE。原始官方打印的 ATE “RMSE” 是另一个 mean 指标，保留原日志但不混写。

## 网站

学习内容以网站 `http://localhost:8799/gs-lab` 为入口，SplaTAM 主线在知识列表首位。桌面 `knowledge/24_splatam_official_rgbd_system.md` 是维护源；网站只同步知识和明确引用的小型证据、图像与脚本，不复制原始数据、模型或环境。
