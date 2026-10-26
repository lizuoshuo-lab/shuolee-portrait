# 113：WinSCP 传输与服务器训练操作

本次目标：**复现现有 HOVER_VERSA_0102 视频的 113 实验**。使用 `shuolee/113` 分支和固定标签 `v113.0.0-reproduction`，不使用 `main` 的后续新视频诊断入口。

**操作位置：公司电脑 00 → WinSCP 连接训练服务器 → 打开 SSH 终端 → 从 Git 拉取 113 → 在服务器训练 → 用 WinSCP 下载点云。工程代码由 Git 同步，不需要用 WinSCP 上传整份源码。**

113 是基于 PGSR 的本地静态人像适配：175 帧真实整图相机固定，重新拟合人体和中性头部，用衣物轮廓及观测重新生成 20 万初始高斯，从零优化 18,000 步。输入中保留当时的检测结果与 COLMAP 估计相机，**不读取以前优化的人体几何、高斯或优化器**。它不是官方完整复现，也尚未通过重影质量验收。本版的目的首先是让服务器运行同一实验；换一台机器本身不会修好原来的重影。

## 1. 先确认连接的是训练服务器

WinSCP 用来传文件；训练在 Linux 服务器的 SSH 终端中执行。公司 Windows 电脑 00 可以运行 WinSCP，但不等于训练服务器。

优先打开现有 `Company-Training` 保存的连接。如果需要新建连接，使用之前提供的记录并核对当前是否仍有效：

| WinSCP 字段 | 填写内容 |
|---|---|
| 文件协议 | SFTP |
| 主机名 | `192.168.2.248` |
| 端口 | `60108` |
| 用户名 | `zhouhaoyi` |
| 密码 | 在 WinSCP 登录窗口输入，不写进文档、Git 或脚本 |

本次没有远程核验该地址能否连接。异地若无法访问 `192.168.2.248`，需先连接公司 VPN 或使用管理员提供的可达地址；它是内网地址。

首次连接出现主机指纹时，与服务器管理员提供的指纹核对。登录后，左边是当前 Windows 电脑，右边是服务器。

打开 SSH 终端：WinSCP 的“命令 → 在 PuTTY 中打开”（部分版本快捷键 `Ctrl+P`）。如果没有 PuTTY，直接在 Windows Terminal / PowerShell 输入：

```powershell
ssh -p 60108 zhouhaoyi@192.168.2.248
```

登录后执行下面三条，确认主机、用户和 GPU：

```bash
hostname
whoami
nvidia-smi
```

## 2. 准备服务器目录和 Git

下文统一使用 `$HOME/shuolee`，不需要在系统根目录创建文件。所有后续命令均在 Linux SSH 终端运行。

```bash
mkdir -p "$HOME/shuolee/datasets" "$HOME/shuolee/models" "$HOME/shuolee/runs"
cd "$HOME/shuolee"
```

私有仓库需要服务器上的 GitHub 身份有访问权限。本机登录成功，不会自动替服务器登录。若服务器已配置 GitHub 凭据，直接克隆；否则可以先在服务器安装好的 GitHub CLI 中执行：

```bash
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
```

按终端提示在浏览器输入它临时生成的验证码并授权 `lizuoshuo-lab` 账号。不把密码或令牌粘进命令。没有 `gh` 时，请按 [GitHub CLI 官方安装说明](https://github.com/cli/cli/blob/trunk/docs/install_linux.md) 安装，或使用服务器已经配置的 GitHub SSH 密钥。

**首次克隆**，目标文件夹需尚不存在：

```bash
git clone --branch shuolee/113 https://github.com/lizuoshuo-lab/shuolee-portrait.git portrait113
cd portrait113
git checkout --detach v113.0.0-reproduction
git describe --tags --exact-match
git status --short
```

预期标签为 `v113.0.0-reproduction`，最后一条没有输出。使用固定标签可以避免以后分支变化影响复现。不要执行 `git pull origin main`。已有其他工程时，保留它，使用独立的 `portrait113` 目录。

代码、配置、说明由 Git 传输；图像、模板、权重和运行输出不进入 Git。

## 3. 用 WinSCP 上传两份输入

在 WinSCP 右边进入服务器家目录的 `shuolee`。下面的 `$HOME` 是 SSH 中 `echo "$HOME"` 显示的目录；WinSCP 路径栏中填写实际路径，不输入字面量 `$HOME`。

| 左侧本机目录 | 右侧服务器目标目录 | 内容 |
|---|---|---|
| `D:\00\工作\人体重建\data\hover175` | `$HOME/shuolee/datasets/hover175` | 约 833 MB，175 帧图片、mask、相机、原始检测与对应观测 |
| `D:\GS-SLAM-Data\shuolee-113-assets` | `$HOME/shuolee/models/113` | 约 252 MB，8 个必要模板/先验/映射文件 |

具体做法：

1. 将左边 `hover175` 文件夹拖到右边 `datasets` 文件夹内。
2. 右边 `models` 内新建 `113` 文件夹；将左边 `shuolee-113-assets` **里面的四个目录** `gvhmr`、`models`、`smplx`、`vposer` 上传到右边 `113` 内。
3. 传输模式选“二进制”，保持 JSON、NPZ 等输入的原始字节和哈希，等待传输队列完全结束。
4. 检查不是多套一层目录：服务器应存在 `datasets/hover175/training-inputs.json` 和 `models/113/smplx/SMPLX_NEUTRAL.npz`。

这两份材料已在当前控制电脑准备好。公司电脑 00 可从 WPS 获取 `D:\00` 下的输入；若同步目录的盘符不同，以 00 的实际下载位置为准。`shuolee-113-assets` 刻意放在 WPS 外，需要从当前控制电脑直接通过 WinSCP 传到服务器，或另行携带。服务器已有同一批模板时可复用，后面的 audit 会核对哈希。**不要把整个旧项目、几 GB 的 HMR2/ViTPose 权重、旧 checkpoint 或 Python 环境再放回 WPS。** 113 使用已固定的检测观测，本次无需重新运行那些大检测模型。

上传后在 SSH 里检查：

```bash
test -f "$HOME/shuolee/datasets/hover175/route113/shared-observations.npz"
test -f "$HOME/shuolee/models/113/models/flame2023/flame2023.pkl"
test -f "$HOME/shuolee/models/113/vposer/V02_05/snapshots/V02_05_epoch=08_val_loss=0.03.ckpt"
```

三条均返回成功且无报错，再继续。后面的 `audit` 会逐个检查 SHA-256，不仅检查文件名。

## 4. 安装服务器环境

锁定环境是 **Linux x86_64、Python 3.10、PyTorch 2.7.1 + CUDA 12.8、CUDA Toolkit 12.8**。需要可用 NVIDIA 驱动及足够显存；本机核验不代表任意服务器显卡均能运行。`nvidia-smi` 显示的 CUDA 版本是驱动能力，不代表已安装编译工具 `nvcc`。

先查看：

```bash
python3.10 --version
nvcc --version
ffmpeg -version
g++ --version
cmake --version
```

缺少 Python 3.10、FFmpeg、C++ 编译工具、CMake、tmux 或 CUDA Toolkit 时，请先在服务器环境中安装对应依赖。已有 CUDA 12.8 在别处时，将下面 `CUDA_HOME` 改为它的根目录，目录内应有 `bin/nvcc`；不替换服务器驱动。

```bash
cd "$HOME/shuolee/portrait113"
export CUDA_HOME=/usr/local/cuda-12.8
export SHUOLEE_ENV_DIR="$HOME/.local/share/shuolee-portrait/env-cu128"
export MAX_JOBS=2
bash scripts/setup_server.sh --preflight-only
bash scripts/setup_server.sh
bash scripts/setup_113.sh
source "$SHUOLEE_ENV_DIR/bin/activate"
```

第一段安装四个基础 CUDA 扩展；第二段额外编译 113 的 `diff_plane_rasterization`。模型文件不会由这些安装命令下载。环境和编译缓存放在项目外，不进入 Git/WPS。

已有同一份合格 `env-cu128` 时，不重复运行第一段安装；先激活并运行后面的 `doctor`。基础安装中断后，检查日志，再使用 `bash scripts/setup_server.sh --resume-install` 只恢复依赖安装，**不是续训**。不要复制另一台机器的 `.so` 或旧虚拟环境过来。

## 5. 先核验输入与 CUDA

```bash
cd "$HOME/shuolee/portrait113"
source "$HOME/.local/share/shuolee-portrait/env-cu128/bin/activate"
unset PYTHONPATH SHUOLEE_EXTRA_PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
python -s scripts/route113.py audit \
  --dataset "$HOME/shuolee/datasets/hover175" \
  --assets "$HOME/shuolee/models/113"
python -s scripts/route113.py doctor \
  --dataset "$HOME/shuolee/datasets/hover175" \
  --assets "$HOME/shuolee/models/113"
```

`audit` 应报告 175 帧、输入和模型哈希通过；`doctor` 应报告 `CUDA_controls_passed`。这表示数值/环境检查通过，**不等于相机物理精度或重影质量通过**。任何一项失败都先解决，不编辑检查结果为 true。

## 6. 一条命令从头运行 113

先 `nvidia-smi` 确认目标 GPU 没有其他正在训练的任务。建议使用 `tmux`，避免关闭 WinSCP 或网络断开导致进程终止。

```bash
tmux new -s shuolee113
cd "$HOME/shuolee/portrait113"
source "$HOME/.local/share/shuolee-portrait/env-cu128/bin/activate"
unset PYTHONPATH SHUOLEE_EXTRA_PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
set -o pipefail
python -s scripts/route113.py all \
  --dataset "$HOME/shuolee/datasets/hover175" \
  --assets "$HOME/shuolee/models/113" \
  --run "$HOME/shuolee/runs/113_001" \
  2>&1 | tee "$HOME/shuolee/runs/113_001-launch.log"
```

`113_001` 必须不存在。第二轮若确实要重跑，改为 `113_002`，不会读取第一轮 checkpoint。不要同时打开两个窗口重复执行同一命令。

顺序为：输入核验 → 中性头部重新拟合 → 175 帧人体粗拟合 1,200 步 → 衣物观测重新三角化 → 175 视角衣物轮廓初始化 → 有观测支持的面部点替换 → 初始方向/共享三维目标 → 175 相机数值与支持度预检 → **20 万初始点、18,000 步静态高斯训练** → PLY 导出与回读核验。身体粗拟合的逐帧参数只辅助初始化；最终高斯是共享静态世界表示，没有动态网络。

初始点为 20 万；训练会按原方法增密/裁剪，最终点数不要求恰好 20 万。全流程使用固定的 113 损失、采样和随机种子，不开放任意调参开关。相机内外参不在训练中优化。

离开 tmux：按 `Ctrl+B`，松开后按 `D`。重新进入：

```bash
tmux attach -t shuolee113
```

## 7. 查看进度与失败位置

另开 SSH 终端：

```bash
ls "$HOME/shuolee/runs/113_001/stages"
cat "$HOME/shuolee/runs/113_001/stages/fit.json"
tail -n 30 "$HOME/shuolee/runs/113_001/logs/fit.log"
```

到了高斯训练阶段，再看：

```bash
tail -n 30 "$HOME/shuolee/runs/113_001/logs/train.log"
cat "$HOME/shuolee/runs/113_001/gaussian-run/progress.json"
nvidia-smi
```

尚未进入的阶段没有日志是正常的。总入口日志是 `runs/113_001-launch.log`；各阶段的 `stages/*.json` 会标记 completed 或 failed。查看 `native-volume-preflight/result.json` 可核查训练前的支持率和控制实验。

失败时保留失败日志与输入配置，先分析原因。程序不会自动换模型、降低门槛、覆盖原实验或跨实验续训。不要删掉 marker 或把 `passed` 手工改成 true 来继续。环境、OOM、文件遗漏等失败都应先定位；修复后的完整重跑使用新目录。

## 8. 下载查看器可直接打开的结果

先确认：

```bash
cat "$HOME/shuolee/runs/113_001/gaussian-run/result.json"
cat "$HOME/shuolee/runs/113_001/gaussian-run/delivery.json"
cat "$HOME/shuolee/runs/113_001/stages/export.json"
```

必须有 `steps: 18000`、`from_scratch: true`、`compatibility_passed: true`，并且 export 阶段 completed。只有训练结束而导出失败时，不将临时 PLY 当成验收完成。

用 WinSCP 从服务器下载这些文件：

| 服务器相对于 `runs/113_001/` 的路径 | 用途 |
|---|---|
| `gaussian-run/viewer-export/point_cloud.ply` | 首选：+Y 向上，完整协方差/旋转及 SH 一起变换的标准高斯 PLY |
| `gaussian-run/viewer-export/` 内 JSON/相机文件 | 坐标和导出核验记录，可一起保留 |
| `gaussian-run/review-current.jpg` | 训练视角最终对照，仅用于评审，不能单独证明未知视角质量 |
| `gaussian-run/result.json`、`delivery.json` | 步数、数量、格式/回读记录 |
| `gaussian-run/progress.json`、`logs/train.log` | 训练记录 |

下载到例如 `D:\00\工作日志\YYYY-MM-DD\113_001\`。公司电脑 00 的 **shuolee 查看器 → 打开文件**，选择下载的 `viewer-export/point_cloud.ply`。先选 Gaussian 模式、缩放 1×；再切 points 或调小高斯球检查分布。不要选 `initialization.ply`、`hull-*.ply` 或 `point_cloud.checking.ply`。

还需要检查相同实拍相机与连续未知视角：鼻嘴眼镜、肩袖衣摆、腿脚是否有多层重影。导出坐标正确、文件能打开、训练满 18,000 步，都不代表这些质量问题已经解决。本说明不代表本轮已在 00 上部署或在远程服务器训练通过。

## 9. 当前版本边界与更新方式

- Git 113 是可审查的历史实验移植入口；本次输入固定为这段现有视频。**不把新 MP4 直接丢进 `hover175`**，也不复用这套相机、mask 或检测观测到新视频。
- 新视频自动重建的研究代码保留在仓库其他路径，但本次不要调用 `run.py reconstruct` 或 `scripts/experiment.py`，后者是 114 对照路线。
- 本地开发更新使用 `shuolee/113`；服务器复现使用不可变标签。以后更新应给出新标签，不覆盖 `v113.0.0-reproduction`。
- 当前执行核验范围以 `provenance/route113-validation.json` 为准。重影质量仍未通过，不将它命名为“稳定成品版”。

官方基础代码与许可：PGSR https://github.com/zju3dv/PGSR；本项目增加的静态人体初始化、观测对齐、训练和导出适配均在源码及 provenance 中标注。模板由用户授权提供，仅上传自己的获授权服务器，不进 Git。
