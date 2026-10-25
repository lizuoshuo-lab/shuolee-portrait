# Shuolee 新视频静态人像重建

目标：在 Linux NVIDIA 服务器上输入一段新视频，自动完成抽帧、整图相机估计、人物检测与分割、身体拟合、高斯训练和标准 PLY 导出。每次使用新视频自己的观测，不套用历史人物的相机、拟合或高斯。

**当前是 0.1.0-rc1 开发候选，未完成端到端质量验收，不能作为已验收稳定版。** 2026-10-26 已在本机的独立 Linux Python 环境，从同一参考原视频重新跑通 23 帧抽帧、全图 SfM、人物分割、全身特征、GVHMR 先验和 1,200 步全新静态拟合，停在 fit-gate。记录见 provenance/server-interface-check-008.json。这是既有参考视频的接口验证，不是新人物泛化测试。未开始高斯训练，尚未完成干净服务器安装及全帧质量验收。历史重影问题仍未解决。

独立 Python 环境已完成实际 CUDA 前后向、投影和 PLY 数值核验，21 个外置模型资产校验通过，未借用旧实验的 Python 包。此检查复用了现有 WSL 系统、Python 解释器和 CUDA 工具链，001安装过程经过修复；之后从私有Git重新克隆，在002全新虚拟环境中一次性安装成功，未补改源码或恢复安装。四个CUDA扩展重新编译，实际算子、完整推理配置和投影/PLY数值检查通过。仍未验证全新操作系统或异地服务器。准确范围见 provenance/installation-check-001.json 与 provenance/installation-check-002.json。007 曾因遗漏上游 wis3d 依赖而停止；补齐依赖与配置预检查后，008 从原视频重新执行，没有恢复失败实验。

## 文件怎么传

- **GitHub 私有仓库**：代码、配置、安装脚本、来源和运行说明。仓库：https://github.com/lizuoshuo-lab/shuolee-portrait
- **WPS 的 D:\00**：报告、按天的工作日志、选定图片与点云；不放预训练权重、虚拟环境或大型缓存。
- **服务器模型目录**：公开模型按上游说明下载，已授权的人体模板单独配置。模型不提交 Git，不再经 WPS 整目录同步。清单及 SHA-256 见 configs/server-assets.json。
- 本机现成资源：D:\GS-SLAM-Data\shuolee-portrait-runtime-assets。此前误放 WPS 的权重与分卷已移至同步范围外的 D:\GS-SLAM-Data\shuolee-portrait-transfer-cache，原文件保留。
- 历史 data/ 只用于 114 对照复核，不是新视频运行的依赖，不在 Git 中。

## 服务器取得代码

在服务器完成自己的 GitHub 授权，再执行：

~~~bash
gh auth login --hostname github.com --git-protocol https --web
git clone https://github.com/lizuoshuo-lab/shuolee-portrait.git
cd shuolee-portrait
~~~

需要 Linux、Python 3.10、NVIDIA 驱动、CUDA 12.8 开发工具（nvcc）、C++ 编译工具与 FFmpeg。安装脚本不安装或修改显卡驱动。当前本机环境是 RTX 5080、PyTorch 2.7.1+cu128；它不等于异地服务器部署已验证。

~~~bash
sudo apt-get update
sudo apt-get install ffmpeg build-essential cmake python3.10-venv libgl1 libglib2.0-0
export PYTHON_BIN=/usr/bin/python3.10
export CUDA_HOME=/usr/local/cuda-12.8
export SHUOLEE_ENV_DIR="$HOME/.local/share/shuolee-portrait/env-cu128"
bash scripts/setup_server.sh --preflight-only
bash scripts/setup_server.sh
source "$SHUOLEE_ENV_DIR/bin/activate"
~~~

请将 Python 和 CUDA 路径改成服务器的实际位置。CUDA 扩展在目标机重新编译，不复制本机的 .so。若系统没有 Python 3.10，先安装兼容解释器。环境、缓存和编译目录默认放在用户主目录的 .local/share 中；禁止安装到 Git 工程内或 D:/00 的同步范围。安装脚本不借用历史实验的 site-packages，不下载模型权重。代码环境与模型资产分开；模型取得方式见 docs/模型配置.md。

安装日志在 `$SHUOLEE_ENV_DIR/install/setup.log`，实际安装版本在 `installed.txt`。直接和间接依赖由 constraints-linux-cu128.txt 固定，避免安装日期变化导致版本漂移。安装末尾会检查包的来源，并运行 kNN、网格和高斯 CUDA 前向/反向小样例；这只验证环境，不能证明人像质量。首次安装拒绝覆盖现有环境；只有依赖安装中断且查明日志原因后，使用 `bash scripts/setup_server.sh --resume-install`。该选项不会恢复任何人像训练。

已有官方 CUDA 运行库 wheel 缓存时，可设置 `CUDA_WHEELHOUSE=/data/official-cuda128-wheels`，脚本按 scripts/cuda128-requirements.txt 离线安装固定版本，减少重复下载；缓存不完整会停止。`MAX_JOBS` 默认 2，控制编译占用。没有配置 `TORCH_CUDA_ARCH_LIST` 时，仅针对当前 GPU 编译；换 GPU 架构需在目标机重新安装。

~~~bash
python run.py doctor --assets /data/shuolee-models
~~~

doctor 检查模型完整性、FFmpeg、CUDA、关键依赖和完整 GVHMR 推理配置。SAM2 与 GVHMR 的 Hydra 配置上下文会分别处理。任何一步失败先处理日志，不继续训练。

需要单独排查服务器环境或导出坐标时，可运行不依赖人体模型的实现检查：

~~~bash
PYTHONNOUSERSITE=1 python scripts/check_environment.py --output /data/install-check/environment.json
python scripts/check_projection_export.py --output /data/install-check/projection-export.json
python scripts/check_prior_crop.py --output /data/install-check/prior-crop.json
~~~

`check_environment.py` 实际执行 CUDA 算子并检查包是否来自本环境；`check_projection_export.py` 使用合成相机和点，检查投影、PLY 位置、协方差及 SH 旋转。通过这些检查不代表输入视频的物理内外参正确，也不代表人像重影已经解决。

裁剪检查使用编码源图像坐标的测试图片，验证 HMR2／ViTPose 输入裁剪与原图的正反映射。新工程已改为调用固定版本 GVHMR 的 `crop_and_resize`：256 像素裁剪覆盖的像素中心是 0–255，不能用 256／框大小替代上游的 255／框大小。每次新运行会保存 `inference/crop-mapping.json`，包含各帧的原图到裁剪及其逆变换。整图相机和高斯训练图片不随这项修正变形。008 的拟合及对照图来自修正前代码，尚未重跑；本次只完成裁剪坐标验证，未证明历史重影的根因。

## 新视频一条命令

~~~bash
python run.py reconstruct \
  --video /data/new_person.mp4 \
  --assets /data/shuolee-models \
  --output /data/runs/person_001
~~~

默认配置 configs/server.json：单个共享静态人物、20 万初始高斯、18,000 步。当前实现是 SMPL-X 网格绑定的本地静态适配，不是缺失公开入口的完整官方复现，也不是双路线自动比较。身体模板用于初始化，高斯有可学习的位置偏移；是否落在真实衣物表面必须验收。

输出目录必须不存在。每次从头拟合、从头采样、空优化器；不覆盖已有实验、不跨实验续训。一个项目同时只运行一个任务。意外中断遗留锁时，先核对锁中记录的机器和 PID，再处理失效锁。

输入应是同一人物尽量静止，相机有足够视差和绕行覆盖，焦距/曝光尽量稳定的视频。当前不学习动态网络，不承诺凭水平视角准确恢复未见区域。默认保留 12–600 帧；长视频需要在配置里明确 frame_stride，不自动丢弃注册失败的相机。

## 执行顺序和保护

1. extract：保留原视频真实帧号、时间戳、图像哈希；按比例缩放，以 RGB24 解码。HDR 需要先明确处理，不直接当作 sRGB。
2. sfm：整张图重新提取特征并建图，估计共享内参及每帧各自外参，再去畸变。共享内参是采集稳定性假设，不是设备原始标定。帧缺失、支持不足或残差过大立即停止，不插值补齐。
3. observations：逐帧关联同一人物，生成 SAM2 mask 与 RTMW 全身、手、脸特征点；对身份歧义、检测失败进行拦截。
4. prior：新视频和当前相机进入 GVHMR，得到初始化先验；不读旧人物输出。
5. fit / fit-gate：全部保留帧共同拟合一个静态 SMPL-X，使用 VPoser 约束。粗拟合门槛未通过则不启动高斯。
6. train：全新 20 万表面采样与优化器。RGB 只学习 mask 内部，alpha 轮廓单独监督；固定本轮相机，检查实际 CUDA 投影矩阵与图像坐标是否一致。视角按打乱循环参与。
7. export：输出标准世界 PLY 与 +Y 版本，旋转中心、完整协方差和球谐系数；原生回读与同相机渲染通过才发布。+Y 来源是身体根方向，不等于实测重力。

自动门槛用于拦截明显失败。SfM 自身残差和用于拟合的特征点不是独立真值，不能据此宣布物理标定准确或重影消失。还需检查面部、眼镜、肩袖、衣摆、腿脚，以及连续未知视角。

当前自动门槛仍只有 SfM 拟合残差和粗身体误差，独立跨视角材料对应、相机精度及人物静止性准入尚未实现，不能把 `fit-gate` 的通过当作正式训练条件已满足。现有参考视频的正式重训仍受 117 诊断结论限制；可先按下方的有界命令完成接口检查，等待新的采集／标定证据。

## 看进度和结果

~~~bash
cat /data/runs/person_001/status.json
tail -f /data/runs/person_001/logs/train.log
~~~

- logs/：每个阶段独立日志，失败返回非零退出码并保留证据。
- data/training-inputs.json：当前视频逐帧相机、原帧号、图像与 mask 哈希。
- fit/geometry-gate.json、reviews/fit/：粗几何诊断与无文字叠加的原图/覆盖图。
- gaussian-run/step-*/：阶段世界 PLY、同相机原图与渲染，保持比例。
- export/point_cloud.ply：最终标准 +Y 高斯；在查看器点“打开文件”。
- export/cameras.json：与导出坐标一致的相机。
- export/review.html：原图/原生渲染/导出渲染及连续未知视角。
- result.json：完成训练仍保持 quality_accepted=false，等待实际质量验收。

point_cloud.checking.ply 表示导出核验未通过，不应作为最终结果。

## 有界诊断

~~~bash
python run.py reconstruct --video /data/new_person.mp4 --assets /data/shuolee-models \
  --output /data/runs/camera_check_001 --stop-after sfm

python run.py reconstruct --video /data/new_person.mp4 --assets /data/shuolee-models \
  --output /data/runs/smoke_001 --config configs/smoke.json --stop-after fit-gate
~~~

短配置降低分辨率、按明确步长选择真实帧，只测试接口与停止条件，不替代全帧正式训练。008 的粗关节投影误差中位数为人物框高度的 0.838%，P90 为 2.207%；这是拟合集上的初始化检查，身体模板也不是穿衣表面。六组实际原图/覆盖图已经检查，头部、手和鞋部轮廓仍有偏差，没有通过物理几何或重影质量验收。步长采样还改变了时序先验的输入间隔，不能将这次先验当作正式全帧结果。不能把短流程的拟合或高斯接续到正式实验。

## 历史与许可

114 双路线原实验说明保存在 docs/114实验快照.md，新入口不调用其历史人物数据。上游版本见 upstream.lock.json；各原许可证随代码保留。公开 GaussianAvatars_SMPLX 缺失入口的本地实现有明确标注。代码、权重、人体模板的许可分别适用；本项目不重新授予商业授权。

