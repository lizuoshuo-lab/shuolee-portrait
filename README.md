# SMPL-X 视频重建

一个 Git 工程、一个视频入口。复用官方 SMPLify-X 拟合流程，仅适配二维关键点、完整相机输入和 SMPL-X 输出到自由衣物高斯训练的接口，最终导出标准 PLY 及同坐标相机。原始视频、模型权重和运行结果保存在 Git 目录之外。

**当前 `smplx/reconstruction` 是本地集成开发分支，尚未发布或完成新视频端到端验证。** 默认入口已连接官方逐帧拟合、按帧选择姿态的高斯训练和指定参考姿态导出。历史结果的 CPU 装配与接口检查通过；画质、服务器执行及总耗时仍未通过验收。

## 部署和运行

服务器需要 Linux x86_64、NVIDIA GPU、CUDA 12.8 开发工具、Python 3.10、Git、FFmpeg 和 C++ 构建工具。模型许可与文件来源见 [模型说明](docs/models.md)。官方 SMPLify-X 固定版本源码的自动配置，以及旧部署依赖的清理仍待完成。以下是开发接口；本地分支尚未推送，不能据此认为服务器已经获得此版本。

```bash
git clone --branch smplx/reconstruction --single-branch https://github.com/lizuoshuo-lab/shuolee-portrait.git ~/smpl-x/app && bash ~/smpl-x/app/deploy.sh
```

部署仅准备环境、模型及服务器配置。视频放在服务器指定目录，使用同一入口运行：

```bash
bash ~/smpl-x/app/scripts/train.sh ~/smpl-x/videos/person.mp4 --camera-config ~/smpl-x/camera.json
```

默认参考帧是 `frame_000001`。需要指定时追加 `--reference-frame frame_000019`。每次创建独立运行目录；`--output /path/to/new-run` 可指定目录，已有结果不会覆盖。

运行命令打印结果目录。状态在该目录的 `status.json`，首个错误的完整堆栈在 `logs/error.log`：

```bash
python run_smplx.py status --run /path/to/run
```

## 相机配置

相机配置描述**上传视频的实际像素域**。传感器原始标定、去畸变后的标定和裁剪后视频的标定不能互换。横竖屏按实际宽高处理，不通过交换宽高猜测 K。

`camera.json` 使用 `schema: "smplx-camera-v1"`，包含：

- `intrinsics`：`mode: "fixed_pinhole"`、`input_domain: "rectified_pinhole"`、实际视频的 `image_wh`、3×3 `K`、`pixel_convention`（`integer_centers` 或 `pixel_corners`）、显式零 `distortion`、处理来源 `provenance` 及 `fixed_processing_across_frames: true`。
- `camera_length_contract`：所有相机共同的 `translation_unit`、`meters_per_camera_unit`、`basis`（`measured` 或 `model_conditioned`）和 `evidence`。当前代码只换算已给定尺度，不能从单目视频自动证明物理米制。刚恢复的 SfM 平移不能默认当作米。
- 可选 `sampling`：`frame_stride`、`max_image_size`、`min_frames`、`max_frames`，默认分别为 5、1600、12、300。

未知去畸变、电子防抖或裁剪映射会返回具体输入错误。仓库中的历史拍摄配置不会自动套用到新视频；输入配置也不能预先认可重建质量。

## 当前执行范围

相机配置还需包含 `official_smplifyx.source_root` 和显式 `official_smplifyx.shape_policy`。`source_root` 指向固定版本官方源码目录，启动时逐文件核对哈希。`shape_policy` 为 `{"mode":"fit_reference"}` 时，先拟合指定参考帧的体型，再对全段冻结这一组 beta；也可使用 `{"mode":"supplied","path":"/absolute/shape.npz","sha256":"实际文件哈希","key":"shared_betas"}`。每次体型更新必须创建新的完整拟合运行，不能让各帧体型漂移。

当前主线为视频与 mask、二维关键点和相机 → 官方 SMPLify-X → 现有 SMPL-X 高斯训练适配 → 通用 PLY 和相机。GVHMR 不再作为主流程初始化；相关历史代码与实验结果保留。官方单帧模块已通过一次完整 CUDA 等价回放。逐帧装配保留完整 K、图像与 mask 身份、命名帧顺序、共享体型及一次相机平移，并核对真实完成的官方拟合记录。默认执行无需人工审核文件。

高斯阶段复用现有训练方法，按帧身份选择对应 SMPL-X 网格，导出指定参考帧的静态 PLY。默认参数保持交接基线：200000 个初始点、18000 步、高度 899、随机种子 6060、缓存 8 帧。优化循环、损失、正则项及增密策略保持一致。导出包含标准 3DGS PLY、3DGS 相机 JSON 和 COLMAP PINHOLE 文本，不部署查看器。静态 PLY 一致性检查使用同一个参考姿态，逐帧重建图使用各自匹配的姿态。

自动面部关键点是拟合观测，不能同时当作独立真值。历史 94 帧装配一致不等于新视频训练或画质通过。先完成一次真实全流程并记录视频输入至模型导出的总耗时，再验证 15 分钟目标。

内部 CPU 检查使用 `python -m unittest discover -s tests -p 'test_*.py' -v`。检查通过与真实视频效果、服务器 GPU 验证分别记录。
