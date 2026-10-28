# Shuolee 新视频静态人像重建

目标：在 Linux NVIDIA 服务器上输入一段新视频，自动完成抽帧、整图相机估计、人物检测与分割、身体拟合、高斯训练和标准 PLY 导出。每次使用新视频自己的观测，不套用历史人物的相机、拟合或高斯。

**当前是 0.1.0-rc1 开发候选，未完成端到端质量验收，不能作为已验收稳定版。** 2026-10-26 裁剪映射修正后，009在本机的独立 Linux Python 环境，从同一参考原视频重新跑通23帧抽帧、全图SfM、人物分割、全身特征、GVHMR先验和1,200步全新静态拟合，停在fit-gate。每帧裁剪正逆映射、图像哈希和身份记录均通过核验，六组实拍/模板覆盖图已实际审阅。记录见 provenance/server-interface-check-009.json。这是既有参考视频的工程回归，不是新人物泛化测试。侧面头部、胸前和手脚仍有轮廓差异，未开始高斯训练，异地服务器及全帧质量尚未验收。历史重影问题仍未解决。

**当前版本会在 `alignment-gate` 停止，不能完成正式高斯训练；缺少纹理或可测留出点时会更早停止。** 原入口仅检查粗拟合误差就继续训练，与“先验证相机、对应和人物静止性”的要求不一致，现已修正。三个独立验证器仍未完成，退出码3和对应的缺项报告会说明原因。这是开发阶段的阻断状态，一次性自动重建的完整目标仍未交付；不能靠修改JSON里的布尔值放行。

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

请将 Python 和 CUDA 路径改成服务器的实际位置。CUDA 扩展在目标机重新编译，不复制本机的 .so。若系统没有 Python 3.10，先安装兼容解释器。环境、缓存和编译目录默认放在用户主目录的 .local/share 中；禁止安装到 Git 工程内或 D:/00 的同步范围。安装脚本不借用历史实验的 site-packages，不下载模型权重。代码环境与模型资产分开；模型取得方式见 docs/models.md。

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

裁剪检查使用编码源图像坐标的测试图片，验证 HMR2／ViTPose 输入裁剪与原图的正反映射。新工程已改为调用固定版本 GVHMR 的 `crop_and_resize`：256像素裁剪覆盖的像素中心是0–255，不能用256／框大小替代上游的255／框大小。每次新运行会保存 `inference/crop-mapping.json`，包含各帧的原图到裁剪及其逆变换。整图相机和高斯训练图片不随这项修正变形。008保留修正前的拟合及对照图，009已从原视频重做至拟合检查；009的23帧裁剪端点误差最大约0.000021裁剪像素。两轮重新求解SfM，不能把拟合数字的差异归因于裁剪修正。该修正尚未证明历史重影的根因。

## 新视频入口（当前停在对齐诊断）

~~~bash
python run.py reconstruct \
  --video /data/new_person.mp4 \
  --assets /data/shuolee-models \
  --output /data/runs/person_001
~~~

目标配置 configs/server.json：单个共享静态人物、20万初始高斯、18,000步。当前命令完成前面的处理后，会在缺少独立验证的 `alignment-gate` 返回退出码3，尚不会创建高斯或优化器。代码中的训练实现是SMPL-X网格绑定的本地静态适配，不是缺失公开入口的完整官方复现，也不是双路线自动比较。身体模板用于初始化，高斯的位置偏移和真实衣物表面分布仍需后续验收。

输出目录必须不存在。每次从头拟合、从头采样、空优化器；不覆盖已有实验、不跨实验续训。一个项目同时只运行一个任务。意外中断遗留锁时，先核对锁中记录的机器和 PID，再处理失效锁。

输入应是同一人物尽量静止，相机有足够视差和绕行覆盖，焦距/曝光尽量稳定的视频。当前不学习动态网络，不承诺凭水平视角准确恢复未见区域。默认保留 12–600 帧；长视频需要在配置里明确 frame_stride，不自动丢弃注册失败的相机。

## 执行顺序和保护

1. extract：保留原视频真实帧号、时间戳、图像哈希；按比例缩放，以 RGB24 解码。HDR 需要先明确处理，不直接当作 sRGB。
2. sfm：整张图重新提取特征，在几何匹配前预留一组纹理轨迹，从用于相机求解的数据库中移除其原始特征行及附近别名。再匹配、建图，估计共享内参及每帧各自外参并去畸变。保留前后行号映射及实际匹配／三维点关联检查；预留不足、训练支持不足或帧缺失就停止，不回退为全部特征。共享内参仍是待验证的采集假设，不是设备原始标定。
3. observations：逐帧关联同一人物，生成SAM2 mask与RTMW全身、手、脸特征点；记录非目标人物检测框及面部区域。面部框只用于区域划分，不定义材料对应。整图像素域约定保留到训练清单中。
4. material-tracks：在当前去畸变图上提取SIFT纹理候选，做双向最近邻比例检查、三视角闭环和全局身份冲突检查；排除图像／mask边界及已检测的其他人。保留每对匹配的拒绝原因。此CPU基线只产出候选，尚未证明材料身份、静止性或相机验证独立性；没有闭环轨迹就停止。
5. alignment-diagnostics：将求相机前冻结的观测按面部ROI、非脸部前景、背景候选和未知分区，记录每帧覆盖、三角化视差及留出投影误差。区域不一致、mask边界、其他人及不可测点均保留。原观测、相机与轨迹划分不变；没有可测留出点时返回3，停止在身体拟合之前。计算出误差不等于独立物理验收。
6. prior：新视频和当前相机进入GVHMR，得到初始化先验；不读旧人物输出。
7. fit / fit-gate：全部保留帧共同拟合一个静态SMPL-X，使用VPoser约束。粗拟合门槛未通过则不启动高斯。
8. alignment-gate：记录当前视频、相机清单、拟合及裁剪文件的哈希和独立验证器状态。三个独立验证器尚未完成，因此拒绝进入训练，保存 `alignment/training-readiness.json`。哈希只标识这些文件，不证明物理几何正确。
9. train（当前未开放）：目标为全新20万表面采样与优化器。RGB只学习mask内部，alpha轮廓单独监督；固定本轮相机，检查实际CUDA投影矩阵与图像坐标是否一致。视角按打乱循环参与。
10. export（正式训练开放后）：输出标准世界PLY与+Y版本，旋转中心、完整协方差和球谐系数；原生回读与同相机渲染通过才发布。+Y来源是身体根方向，不等于实测重力。

自动门槛用于拦截明显失败。SfM 自身残差和用于拟合的特征点不是独立真值，不能据此宣布物理标定准确或重影消失。还需检查面部、眼镜、肩袖、衣摆、腿脚，以及连续未知视角。

现有数值检查包括SfM拟合残差、求解前留出的观测及粗身体误差；独立物理材料对应、相机精度及人物静止性验证尚未实现。原始观测行未用于求相机，不等于同一物理点的所有别名都已排除。`alignment-gate` 阻止把粗拟合通过当作正式训练条件满足，直接调用训练函数也会在加载CUDA模型前停止。手写或过期的“通过”报告不作为输入。现有参考视频的正式重训仍受117诊断结论限制；等待新的采集／标定证据，不为反复比较指标而重跑已关闭的诊断。

只读查看已有运行的缺项，不修改其文件：

~~~bash
python run.py alignment-status --output /data/runs/person_001
~~~

当前返回退出码3表示尚未准入，不能将其当作可忽略的安装告警。检查训练入口是否正确阻断，可运行 `python scripts/check_training_admission.py --output /data/install-check/training-admission.json`；该测试只验证缺少证据时停止，不验证人像已经对齐。

固定相机的三维点诊断模块已实现，可以将预先指定视角用于三角化，其余视角只用于重投影检查。它保留每条轨迹的错配残差、视差不足和负深度，不按误差大小删掉失败点。纹理候选已可由 `material-tracks` 阶段自动生成，但独立的相机／材料身份／微动验证尚未完成，因此不能放行训练。参与过相机求解的观测会标为非独立；未知使用关系保留未知，不能因为换了特征编号就声称独立。

已有合格的轨迹文件时，工程接口为：

~~~bash
python scripts/evaluate_fixed_tracks.py \
  --run /data/runs/person_001 \
  --tracks /data/runs/person_001/data/material-tracks/tracks.json \
  --report /data/diagnostics/person_001_points.json
~~~

轨迹格式见 [固定点诊断接口](../固定点诊断接口.md)。结果文件必须位于原运行目录之外且尚不存在；原视频、相机和拟合均不修改。当前已有的 `observations.npz` 只是检测器语义关键点，不能直接改名为这份材料轨迹文件。`python scripts/check_fixed_tracks.py --output /data/install-check/fixed-tracks.json` 用OpenCV独立投影的已知几何测试数值与退化情况，未测试实拍人物质量。

`python scripts/check_material_tracks.py --output /data/install-check/material-tracks.json` 会实际执行SIFT提取、双向／闭环匹配和新阶段入口。测试图像有已知像素变换和人为设定的前景移动，不是实拍人像。这个后置OpenCV候选库仍可能与COLMAP观测重合；它和下面预先留出的原生特征库是不同的证据，不能混称独立标定验收。

新流程的求解前留出库为 `data/sfm-holdout-tracks.json`。固定点诊断命令将 `--tracks` 改为该文件即可；它引用单独保存的 `data/sfm-inputs.json`，避免人物分割阶段追加字段后相机清单哈希改变。原始畸变观测及去畸变后的整数像素中心坐标都保留，转换使用本轮原生相机模型，不能直接把原图坐标放入去畸变K。

`python scripts/check_sfm_holdout.py --report /data/install-check/sfm-holdout.json` 包含实际COLMAP特征提取，以及已知三维观测注入后的实际匹配、几何验证、建图、去畸变和留出重投影。5视角工程测试预留71条轨迹、355个观测，实际求解使用其中0个；9组检查通过。这些都是模拟输入，不是新人物、历史视频修复或服务器质量验收。记录见 `provenance/sfm-holdout-validation.json`。

只读生成一个新流程的分区诊断及同相机图片，可运行：

~~~bash
python scripts/diagnose_alignment.py --run /data/runs/person_001 \
  --output /data/diagnostics/person_001_regions
~~~

输出目录必须不存在且在源运行目录外。该入口要求当前流程完整的求解前留出、使用记录和转换哈希；不能修改旧运行的清单来冒充具备这些证据。`reviews/index.html` 提供完整原图与面部裁剪的成对查看：原图不变形，另图只画观测／留出投影标记，文字放网页及JSON中。面部框不是精细人脸分割，非脸部前景也不能全称作衣服。只有数值可计算时阶段才完成，物理标定、材料身份与微动条件仍未验收。`python scripts/check_alignment_diagnostics.py --report /data/install-check/alignment-regions.json` 使用模拟数据实际检查分区、未知覆盖、4像素前景运动、原生SfM到诊断CLI与提前停止路径。

同一阶段的 `material-review/index.html` 还提供逐轨迹原图审阅：保留全部冻结轨迹及全部观测，可以按区域／轨迹查看，在1×、2×、3×相同倍率下检查纹理，再用“在完整原图中定位”核对上下文。原图副本字节一致，页面不依赖互联网资源，可通过本机查看器的静态服务打开；显示标记可以关闭，文件本身没有文字或标记。局部窗口随观测位置移动，不代表相机／三维点已经对齐。每条轨迹仍标为未审阅，不自动放行训练。将整个 `material-review` 目录保留在服务器运行目录中；D:/00只同步精选截图与结论，代码目录不收集这批运行原图。

异地查看时，在服务器终端启动仅监听回环地址的静态服务，目录替换为实际运行输出：

~~~bash
python -m http.server 18991 --bind 127.0.0.1 \
  --directory /data/runs/person_001/data/alignment-diagnostics/material-review
~~~

在自己的电脑另开终端建立SSH转发：`ssh -L 18991:127.0.0.1:18991 用户名@服务器`，保持两端终端开启，再访问 `http://127.0.0.1:18991/`。已有静态服务也可以挂接该目录。此操作不需要向公网发布人像，当前尚未在你的异地服务器实测。

## 看进度和结果

~~~bash
cat /data/runs/person_001/status.json
tail -f /data/runs/person_001/logs/train.log
~~~

- logs/：每个阶段独立日志，失败返回非零退出码并保留证据。
- data/training-inputs.json：当前视频逐帧相机、原帧号、图像与 mask 哈希。
- data/material-tracks/：预先保存的匹配方案、特征、双向匹配决策、冲突轨迹、候选共享点和覆盖统计；不包含相机优化结果。
- sfm/raw-features.db、sfm/holdout/：不经匹配的原生提取库、预留方案、训练行号到原始行号的映射及实际使用记录。sfm/database.db只含允许参加求解的特征。运行数据放服务器输出目录，不放进Git或WPS同步工程。
- data/sfm-inputs.json、data/sfm-holdout-tracks.json：冻结的相机清单与求解前留出观测；可独立计算固定点重投影，尚不具备物理身份或人体静止性保证。
- data/alignment-diagnostics/：预先保存的源证据哈希、全部冻结观测的区域标签、所有点的残差／不可测原因、逐帧覆盖及原图审阅页。`scene-support.json` 分开记录固定外参的内参信息、允许外参和三维点变化后的内参信息，以及缺帧和规范自由度；只分析局部敏感度，不优化相机。缺失区域误差为null，不记为0；高残差不被筛掉。数值满秩不能证明物理标定正确。
- fit/geometry-gate.json、reviews/fit/：粗几何诊断与无文字叠加的原图/覆盖图。
- alignment/training-readiness.json：当前对齐验证缺项及所依据文件的哈希。当前版本在此停止，以下训练产物尚不能由默认命令生成。
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

短配置降低分辨率、按明确步长选择真实帧，只测试接口与停止条件，不替代全帧正式训练。009的粗关节投影误差中位数为人物框高度的0.840%，P90为2.115%；这是拟合集上的初始化检查，身体模板也不是穿衣表面。六组实际原图/覆盖图已经检查，头部、胸前、手和鞋部轮廓仍有偏差，没有通过物理几何或重影质量验收。步长采样还改变了时序先验的输入间隔，不能将这次先验当作正式全帧结果。不能把短流程的拟合或高斯接续到正式实验。

## 历史与许可

114 双路线原实验说明保存在 docs/114实验快照.md，新入口不调用其历史人物数据。上游版本见 upstream.lock.json；各原许可证随代码保留。公开 GaussianAvatars_SMPLX 缺失入口的本地实现有明确标注。代码、权重、人体模板的许可分别适用；本项目不重新授予商业授权。
