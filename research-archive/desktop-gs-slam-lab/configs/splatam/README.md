# SplaTAM：固定代码与待运行配置

2026-10-08 状态：已浅层、稀疏获取官方代码；配置和轨迹导出工具已准备。**未安装 SplaTAM CUDA 扩展，未执行其跟踪/建图，未获得其 ATE 或重建指标。** 当前优先完成本机 gsplat 小实验。这一目录不能作为 SplaTAM 已复现的证明。

- 官方来源：<https://github.com/spla-tam/SplaTAM>
- 固定提交：`da6bbcd24c248dc884ac7f49d62e91b841b26ccc`（2024-06-19）
- 数据：`data/tum/rgbd_dataset_freiburg1_desk`，可用 `SPLATAM_TUM_ROOT` 覆盖父目录。
- `GS_SLAM_LAB` 可覆盖实验根目录；默认由本文件所在目录推导，Windows / Linux 均可加载配置。
- `tum_desk_20.py` / `tum_desk_200.py`：匹配筛选后的前 20 / 200 帧，320×240，seed 0；沿用官方 TUM 的 tracking 200 次、mapping 30 次优化。短序列/半分辨率仅用于调试，不对应论文全量结果。
- `use_wandb=False`、`visualize_tracking_loss=False`，启动器设置 Matplotlib Agg，不调用 Open3D viewer。虽然不发送日志，原版导入仍需要安装 wandb。
- `tracking.use_gt_poses=False`：跟踪估计位姿。数据加载器仍需要 GT 做帧关联、首帧坐标系转换和评测；不能称其为完全不依赖 GT 文件的部署系统。

## 本地可先做的无 CUDA 检查

在 lab 根目录用已有 Python 执行：

```text
python configs/splatam/run_headless.py 20 --check-only
python configs/splatam/run_headless.py 200 --check-only
```

检查成功仅意味着配置、版本、数据路径正确。实际运行需要单独适配的 SplaTAM 环境；当前 Windows RTX 5080 上已验证的 gsplat 环境并不等于 SplaTAM 环境。

## 代码阅读顺序

1. `datasets/gradslam_datasets/tum.py`：RGB/depth/GT 最近时间匹配，阈值 `<0.08s`；之后按 RGB 时间间隔 `>1/32s` 保留帧；最后由 base dataset 应用 start/end/stride。
2. `datasets/gradslam_datasets/basedataset.py`：RGB 线性缩放、depth 最近邻缩放；depth 除以 5000，内参按尺寸缩放，位姿变成相对首帧的 c2w。YAML 的 crop_edge 虽被读入，此版实际预处理没有裁剪图像。
3. `scripts/splatam.py`：首帧反投影 → 位姿外推 → 固定地图优化相机 → 按遮挡/覆盖新增 Gaussian → 选重叠关键帧优化地图 → pruning → 保存。
4. `utils/slam_helpers.py` / `utils/recon_helpers.py`：相机和 Gaussian 坐标变换、RGB/depth/silhouette 渲染参数。
5. `utils/eval_helpers.py`：官方质量评测和轨迹日志；`utils/common_utils.py`：保存 params.npz。

该版本无显式回环检测 / 全局位姿图优化模块。局部关键帧优化不能替代回环。

## 轨迹评测

`export_evo.py` 从最终 params.npz 读取 `cam_unnorm_rots`（wxyz）和 `cam_trans`，构造 w2c 后求逆得到 c2w；GT 同样从已保存的 gt_w2c_all_frames 求逆。时间戳复用上游 TUM 的匹配/筛帧规则，并严格检查帧数。

```text
python configs/splatam/export_evo.py configs/splatam/tum_desk_20.py
```

在生成的 `outputs/splatam/<run_name>/evo` 目录中，用单独安装的 evo 执行：

```text
evo_ape tum reference.tum estimated.tum -a --save_results ape_se3.zip
evo_rpe tum reference.tum estimated.tum -r trans_part --delta 1 --delta_unit f --save_results rpe_1frame.zip
```

RGB-D 用 SE(3) 刚体对齐，不加 `--correct_scale`。RPE 的一帧指 loader 保留帧。上游 `evaluate_ate` 返回对齐后平移误差均值，却标作 RMSE，且传入 w2c 平移；独立导出用于规避该口径混淆，原日志另存。图像评测用 MS-SSIM 与有效 depth mask，不能直接标成任意 SSIM / 全图评测。

## 可选 Linux 资料（未执行）

`scripts/cloud/install_splatam_optional.sh` / `verify_splatam_cuda.py` / `run_splatam_optional.sh` 仅保留为以后已有 Linux RTX 3090/4090 资源的候选方案，不涉及云服务开通或购买。用户当前选择是本地实验。候选 recipe 采用 README 提到测试过的 Torch 2.3.0 + CUDA 12.1，并固定自定义 depth rasterizer `cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110`。

辅助依赖的固定版本是我们选择的候选组合，**不是已通过复现的 lockfile**；只覆盖离线 TUM/headless/PLY 路径，省略 Open3D GUI 和 CycloneDDS 手机实时传输。将来安装后必须先通过 pip check、CUDA forward/backward，再执行 20 帧，保存实际 pip freeze。官方 LPIPS 初始化可能下载 AlexNet 权重，需允许网络并记录缓存。

## 许可

主仓 LICENSE 为 BSD-3-Clause。`utils/slam_external.py`、`diff-gaussian-rasterization-w-depth` 等继承组件另有 Inria 研究/非商业条件；不能仅根据主仓许可证宣称整个依赖链可商用。学习与求职展示保留出处和适用声明。
