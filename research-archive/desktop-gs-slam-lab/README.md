# GS SLAM 学习与实验室

固定工作目录：`C:\Users\Administrator\Desktop\GS-SLAM-Lab`。

每天 3 小时，岗位方向为无人机/机器人定位建图、大场景重建、世界模型。当前新增工作按用户要求长期推进大场景三维高斯重建：统一 SfM 坐标、带重叠的分块训练与合并、层次细节（LOD）。固定项目状态见 [大场景项目档案](manifests/large-scene-gs-work.json)，知识继续集中在现有网站。SplaTAM 和 FAST-LIVO2 的已完成成果与原始失败记录保留。

**首选入口：[GS / SLAM 知识库](http://localhost:8799/gs-lab)。** 桌面双击 `GS知识库.cmd` 可启动本地网站并直接打开该栏目。网站按知识组织：相机几何、高斯表示、Tracking/Mapping、评测；包含全文搜索、公式与代码、实验图像和来源链接。网站不展示每日任务或日程。下面的文件用于维护与复现实验，阅读无需逐个找文件。实际指标以 `outputs/*/metrics.json` 与 `experiments/` 对应记录为准。

**第一优先官方项目：** SplaTAM 在本机 RTX 5080／WSL2 以 320×240 完成 TUM desk 全部 592 个官方关联帧；保存 223,097 个高斯，真实子进程用时约 27.01 分钟。独立 C2W、SE(3) 固定尺度 1 的 ATE RMSE 为 2.014 cm，相邻帧 RPE 为 0.963 cm／0.568°。原始 renderer 无算法补丁。完整结果与复现命令见 [官方本地项目](experiments/28_splatam_official_local_project.md)，知识网站以该项目为主线。

**后续完整对照已完成：** xyz 320×240 全 759 帧的独立 ATE RMSE 为 0.943 cm；desk 640×480 全 592 帧为 1.953 cm。desk 升分辨率后单次 ATE 仅下降约 3.04%，高斯数变为 4.35 倍、耗时为 1.93 倍。协议、轨迹图、实际原图与指标口径集中在 [运动序列与分辨率对照](experiments/29_splatam_sequences_resolution.md)，保留全部初始基线。

**历史基础实验：** 官方单图拟合、多视图地图优化、4000/8000 高斯三种子对照、TUM 已知位姿 RGB-D 建图与深度损失消融、固定地图 SE(3) 位姿恢复、20/100 帧在线 tracking/mapping、两条序列完整 549/748 帧深度 ICP，以及固定 V4 两序列三个种子的十二轮轨迹观察。十二张最终地图还完成了相同参考相机下的 120 次渲染及覆盖误差分解。当前地图有明显缺失、模糊和错位，覆盖率或局部 RMSE 单独改善不能说明整体重建改善；完整误差、失败和适用范围均在站内实验记录中。

CPU 光度与几何里程计已完成两序列、光度项关闭/开启的四轮固定对照，共 400 帧、396 对配准，四个拒绝帧原样保留并完成保存试探诊断。加入光度项后片段位置 RMSE / ATE 改善，但平移 RPE 和末帧误差变差；独立指标复核与完整拒绝记录见网站“光度与几何里程计”主题及实验依据。完整序列 V4 的四轮 GPU 对照继续按冻结协议串行推进。

| 目录 | 用途 |
| --- | --- |
| `knowledge/` | 带稳定 slug 和来源的基础知识笔记 |
| `curriculum/` | 早期学习安排存档，不进入网站 |
| `experiments/` | 已执行实验、结果、失败与解释 |
| `data/` | 公开数据，仅保留本地；下载来源/校验在 manifests/datasets |
| `repos/` | 固定版本的 gsplat 和 SplaTAM；修改补丁另存 |
| `scripts/` | 可重复执行的数据检查、训练和导出脚本 |
| `outputs/` | 图像、模型、指标，不默认进入网站内容包 |
| `logs/` | 编译和运行日志 |
| `manifests/` | 版本、环境、进度、来源、哈希记录 |
| `site-export/` | 后续网站使用的 Markdown 与 JSON 内容包 |

**本地运行**

在本目录打开 PowerShell：

```powershell
.\.venv\Scripts\python.exe scripts\run_local.py scripts\check_gsplat.py
.\.venv\Scripts\python.exe scripts\run_local.py scripts\train_tiny_gs.py --run tiny-nerf-gsplat-002
.\.venv\Scripts\python.exe scripts\check_rgbd_geometry.py
.\.venv\Scripts\python.exe scripts\export_site.py
```

多视图训练的 run 名应每次不同，以保护既有结果；代码中的默认首跑编号可能已被使用。`train_official_image.py` 保留首跑输出，不重复覆盖。运行前查看本文对应实验记录。

`.venv` 只在本实验目录中安装小型附加依赖，通过 `lab_gpu_runtime.pth` 只读复用本机已有的 PyTorch 2.7.1+cu128 包。原环境没有升级或改写。此链接环境不能直接复制到另一台机器，独立复建见 [环境记录](manifests/environment.json)。MSVC/CUDA 环境变量仅在 `run_local.py` 的子进程中设置。

**当前学习范围**

- 官方 `image_fitting.py`：单张图像拟合，帮助看清梯度与优化，不证明三维几何。
- `train_tiny_gs.py`：使用 gsplat 的教学训练器，固定已知相机、固定高斯数量、RGB 颜色；没有 SH、密度控制或相机跟踪，不是原始 3DGS 完整论文复现。
- `train_tum_gs.py`：已知真值相机的 RGB-D 地图优化，包含深度损失消融。
- `track_fixed_map.py`：给定地图和局部初值的 SE(3) 位姿恢复，不是在线 SLAM。
- `train_tum_online_gs.py`：首相机 identity、后续 GT 位姿仅评测的教学 tracking/mapping；V2 的 20/100 帧与 V3 的 ICP 初始化对照已审计，数值结构正确仍可能漂移，不是完整 SLAM。
- `track_rgbd_icp.py` 与两个 full wrapper：相邻深度 projective ICP，含退化校验、相同输入的短片段与完整 desk549/xyz748 帧基线；无回环或全局 BA。独立完整序列指标重算见 `summarize_full_icp_sequence_controls.py`。
- `diagnose_online_pose_gradients.py`：仅采样首次梯度检查，保存固定地图与输入；硬覆盖控制未进入训练，历史有限差分失败未复现。
- `geometric_pose_prior.py` 与 `train_tum_online_gs_v4.py`：相机中心与旋转的工程先验，两序列三个种子、共十二轮 100 帧产物已审计；先验流程接近 ICP，各指标收益并不一致，实际 warmup 地图差异保留。
- `evaluate_tum_v4_final_maps.py` 与 `decompose_tum_v4_coverage_sse.py`：固定参考相机评测最终地图，并在同一传感器有效像素集合上分解覆盖变化与 SSE；GT 提供评测相机，不据此评判跟踪精度。
- `replay_pose_prior_context.py`：复用已保存的同一张地图与观测进行双组单帧优化，验证局部先验机制；GT 在完成后另行评测，单帧结论不推广为轨迹收益。
- `prepare_tum_fr1_xyz.py`：第二条官方序列的完整图像、索引、刚体姿态和时间关联检查；同一 Freiburg1 传感器域，不代表跨场景泛化。
- `audit_tum_trajectory.py`：NumPy 独立轨迹统计，含时间戳、gauge、无 scale ATE 与 RPE 校验。
- SplaTAM：固定官方 renderer 已在 RTX 5080 / WSL2 原样编译，20 帧和完整 592 帧真实 tracking/mapping、独立轨迹评测和实际源码／动态库审核均通过。结果见网站项目记录与 [本地复现入口](configs/splatam/LOCAL_PROJECT.md)。

- `track_rgbd_photogeometric_last_accepted.py`：固定光度与几何目标的最后接受传感器参考，两轮新 CPU 轨迹与两轮旧参考完整对照；参考传感器与位姿前缀逐帧核对，desk 累计位置误差改善但相邻误差与拒绝增多，不代表完整恢复或 SLAM。

**网站维护**

笔记采用稳定 id/slug 的 UTF-8 Markdown。运行导出器会重新生成 `site-export/content.json` 和 Markdown 副本，供已有网站 `engine-home` 的 `scripts/sync-gs-lab.mjs` 读取；不导入 curriculum 日程。网站同步已审阅的笔记、实验图片、指标和源码，不携带原始数据、模型、环境、日志或凭证。当前为本地网站，未发布新云端版本。最新数量与核验记录见 `manifests/website.json`。

本地 Python 源码按 SHA256 归档于 `manifests/source-snapshots/`，索引为 `manifests/source-snapshots.json`。实验改版后保留旧源码与结果；不能只凭当前脚本猜测旧实验实现。

TUM 数据为 CC BY 4.0；Tiny NeRF 的 Lego 源模型具有非商业署名条款，具体来源见数据清单。公开展示其渲染时保留署名，不把教学数据重新打包进网站。
