"""Maintain one deployment guide in the user's SMPL-X work folder."""
from pathlib import Path
import hashlib
import html
import json
import re
import markdown

project = Path('D:/00/工作/人体重建')
out = Path('D:/00/工作/SMPL-X')
source = project / 'docs/113_WinSCP与服务器训练.md'
text = source.read_text(encoding='utf-8-sig')
text = text.replace('# 113：WinSCP 传输与服务器训练操作', '# SMPL-X 工作目录 · 113 详细部署与训练指南', 1)
intro = r'''

更新日期：2026-10-26。阅读位置：公司电脑 00。计算位置：公司 Linux NVIDIA 训练服务器。

**先在 00 双击本目录的 `00_部署与训练指南.html`。** 它是可离线阅读的说明，命令旁有“复制”按钮；复制后粘到对应终端执行。`.md` 是同一份说明的可编辑文本。

本文件夹是部署说明入口。现有代码工作副本仍在 `D:\00\工作\人体重建`，服务器代码从 GitHub 获取，不需要把整份 Windows 工程搬上服务器。目录名 SMPL-X 对应人体粗拟合部分；本次执行的完整路线是 **113 静态人像适配**，没有改成 114 的 GaussianAvatars_SMPLX 对照实验。

## 0. 先看部署清单与执行顺序

| 项目 | 本次固定值 |
|---|---|
| Git 仓库 | [lizuoshuo-lab/shuolee-portrait](https://github.com/lizuoshuo-lab/shuolee-portrait) |
| 开发分支 | `shuolee/113` |
| 服务器复现标签 | `v113.0.0-reproduction` |
| 标签对应提交 | `5fda749498cca899fb63011dc7b1c71511c14050` |
| 数据 | 现有视频的 175 帧、遮罩、相机与观测；约 833 MB |
| 必要模型资产 | SMPL-X、FLAME、VPoser 和映射文件，共 8 个，约 252 MB |
| 训练 | 从头拟合，20 万初始高斯，18,000 步，共享静态人物 |
| 当前验证范围 | 本地 Git 重新克隆、输入核验、模板前向、CUDA 控制检查已通过；公司服务器完整训练尚未验证 |

按顺序完成：**连接服务器 → 获取 Git 113 → 上传两份输入 → 安装依赖 → audit/doctor → tmux 中运行 all → 看日志 → 下载 viewer-export 点云。**

标为 `bash` 的命令在 **服务器 SSH 的 Bash 终端**执行；标为 `powershell` 的命令在公司电脑 00 的 PowerShell 执行。目录树只用于核对文件位置，不是命令。不要把 Bash 命令放进 Windows PowerShell，也不要放进 WinSCP 的远程路径栏。每个代码块执行完，核对成功标志，再进入下一步。
'''
first_end = text.index('\n', text.index('# '))
text = text[:first_end] + intro + text[first_end:]
text = text.replace('hostname\nwhoami\nnvidia-smi', 'hostname\nwhoami\ncat /etc/os-release\nuname -m\nprintf \'HOME=%s\\n\' "$HOME"\nnvidia-smi', 1)
text = text.replace('登录后执行下面三条，确认主机、用户和 GPU：', '登录后执行下面几条，确认主机、用户、操作系统、家目录和 GPU：', 1)
after_gpu = r'''

成功标志：系统是 Linux，架构显示 `x86_64`，用户名为你的服务器账号，`nvidia-smi` 能显示 NVIDIA GPU。若 SSH 不通、身份不对或没有可用 NVIDIA GPU，先处理连接/资源问题，不安装训练环境。此处的 `$HOME` 往往是 `/home/zhouhaoyi`，但以命令实际输出为准。

选择 GPU 时先看 `nvidia-smi` 的进程和显存。后续环境检查及训练应使用同一张空闲 GPU。例如使用编号 0：

```bash
export CUDA_VISIBLE_DEVICES=0
```

如果编号 0 正被别人使用，改成你获分配的编号；不停止其他人的任务。本机数值检查使用 RTX 5080，尚未测定所有服务器显卡的最低资源要求。
'''
needle='nvidia-smi\n```\n\n## 2.'
assert needle in text
text=text.replace(needle, 'nvidia-smi\n```'+after_gpu+'\n## 2.',1)

git_verify=r'''

再确认一次实际提交：

```bash
git rev-parse HEAD
```

必须显示 `5fda749498cca899fb63011dc7b1c71511c14050`。看到 `detached HEAD` 是固定标签复现的正常状态。已有 `portrait113` 且提交一致时，跳过克隆；如果目录已有其他代码或未提交修改，先保留原目录，不能用 `reset --hard` 清掉它。

私有仓库不存在网页匿名下载入口；出现 `Repository not found` 不一定是仓库不存在，也可能是服务器 GitHub 身份没有权限。WinSCP 的 Linux 登录账号与 GitHub 账号是两套认证。

服务器目录最终应为：

```text
$HOME/shuolee/
├── portrait113/                 Git 代码，含 scripts/route113.py
├── datasets/
│   └── hover175/                图片、遮罩、相机和观测
├── models/
│   └── 113/                     8 个必要资产
└── runs/
    ├── 113_001-launch.log       本轮总入口日志
    └── 113_001/                 本轮新生成的拟合、训练和导出
```

虚拟环境另外放在 `$HOME/.local/share/shuolee-portrait/env-cu128`。这样 Git 代码、输入和训练产物各有明确位置。
'''
text=text.replace('代码、配置、说明由 Git 传输；图像、模板、权重和运行输出不进入 Git。', '代码、配置、说明由 Git 传输；图像、模板、权重和运行输出不进入 Git。'+git_verify,1)

asset_tree=r'''

服务器 `models/113` 内的文件结构应是：

```text
113/
├── smplx/SMPLX_NEUTRAL.npz
├── gvhmr/hmr4d/utils/body_model/
│   ├── smpl_coco17_J_regressor.pt
│   └── smplx2smpl_sparse.pt
├── models/
│   ├── flame2023/flame2023.pkl
│   └── flame/
│       ├── landmark_embedding_with_eyes.npy
│       └── head_template_mesh.obj
└── vposer/V02_05/
    ├── V02_05.yaml
    └── snapshots/V02_05_epoch=08_val_loss=0.03.ckpt
```

这里出现两层名称 `models/113/models/flame...` 是入口约定，照此上传即可。不要把它误改为 `models/113/flame...`。清单以 Git 的 `configs/route113-inputs.json` 为准。

`hover175` 则应包含 `images`、`masks`、`inference`、`inspection`、`route113` 目录，以及 `inputs.json`、`training-inputs.json`、`observations.npz`、`hand-observations.npz`。本次复现的实际输入是这套已固定的观测，只有 MP4 文件并不足以运行这个标签。

在 WinSCP 上传时遇到同名文件，不要选择会删除远端额外文件的镜像同步模式。优先用新目录首次上传；已有输入时先使用 audit 核对。传输队列仍在运行或有失败项目时，不开始训练。
'''
text=text.replace('三条均返回成功且无报错，再继续。后面的 `audit` 会逐个检查 SHA-256，不仅检查文件名。','这三条 `test` 成功时没有输出；要明确显示检查结果，可在每条命令后单独输入 `echo $?`，返回 `0` 才成功。后面的 `audit` 会逐个检查 SHA-256，不仅检查文件名。'+asset_tree,1)

installation=r'''

### 4.1 系统基础依赖

先根据第 1 节的 `/etc/os-release` 判断服务器系统。**以下 apt 命令仅适用于 Ubuntu 22.04 且你有 sudo 权限的情况**；Ubuntu 24.04 默认 Python 版本不同，其他系统不要直接套用这条命令。已有合格依赖时跳过安装。

```bash
sudo apt-get update
sudo apt-get install -y git gh tmux ffmpeg build-essential cmake ninja-build pkg-config python3.10 python3.10-venv python3.10-dev libgl1 libglib2.0-0
```

没有 sudo 权限时，让服务器管理员提供 Python 3.10、FFmpeg、C++/CMake 与 CUDA 12.8 开发环境。不要为省略安装步骤而把 Python 3.12、另一个实验环境或旧 `.so` 当成这套锁定环境。

### 4.2 确定 CUDA Toolkit

在已经安装 CUDA 12.8 的服务器上，下面两条应成功：

```bash
test -x /usr/local/cuda-12.8/bin/nvcc
/usr/local/cuda-12.8/bin/nvcc --version
```

如果安装在其他位置，用该位置设置 `CUDA_HOME`。如果完全没有 Toolkit，由管理员按 [NVIDIA CUDA 12.8 官方归档](https://developer.nvidia.com/cuda-12-8-0-download-archive) 选择当前操作系统的安装方式。已配置对应 NVIDIA 软件源的 Ubuntu，可安装 `cuda-toolkit-12-8` 包；这与更换系统 NVIDIA 驱动是两件事。

```bash
# 仅当管理员已经配置好与本系统匹配的 NVIDIA 软件源时执行
sudo apt-get install cuda-toolkit-12-8
```

安装完成再检查 `nvcc --version`。在 `nvidia-smi` 中看到 `CUDA Version 12.8/13.x`，不能替代这项检查。

### 4.3 创建项目隔离环境并编译

第一次完整安装时逐条执行下面代码。依赖包下载耗时取决于服务器网络；安装日志保留在虚拟环境的 `install` 目录。
'''
text=text.replace('先查看：\n\n```bash\npython3.10 --version', installation+'\n先查看：\n\n```bash\npython3.10 --version',1)
text=text.replace('模型文件不会由这些安装命令下载。环境和编译缓存放在项目外，不进入 Git/WPS。', '模型文件不会由这些安装命令下载。环境和编译缓存放在项目外，不进入 Git/WPS。\n\n成功标志：基础安装最后输出 `Dependency installation passed`；113 扩展安装最后输出 `113 rasterizer imported`。通用安装脚本末尾可能提示 `run.py doctor`，本次请按下一节执行 `scripts/route113.py doctor`。',1)
text=text.replace('`audit` 应报告 175 帧、输入和模型哈希通过；`doctor` 应报告 `CUDA_controls_passed`。', '`audit` 应报告 `frames: 175`、`camera_and_pixel_identity_verified: true`、`assets_sha256_verified: true`；`doctor` 应报告 `state: CUDA_controls_passed`。',1)
text=text.replace('tmux new -s shuolee113\ncd', 'tmux new -s shuolee113\ncd',1)
text=text.replace('export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1\nset -o pipefail', 'export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1\nexport CUDA_VISIBLE_DEVICES=0  # 改为你获分配且空闲的 GPU 编号\nset -o pipefail',1)

progress=r'''

总入口通常只显示阶段开始和结束，具体迭代输出写入每个阶段日志。比如 `113 stage: fit` 持续一段时间时，查看 `logs/fit.log`，不要因此再启动一次 `all`。

| 阶段名 | 正在做什么 | 重点查看 |
|---|---|---|
| `prepare` | 核验输入并建立全新目录 | `input-audit.json`、`config.json` |
| `head` | 重新拟合中性头部 | `logs/head.log`、`head-fit/` |
| `fit` | 175 帧人体粗拟合，1,200 步 | `logs/fit.log`、`fit/progress.json` |
| `clothing`、`hull`、`patch` | 衣物观测、轮廓和面部支持点初始化 | 对应阶段日志、`clothing-init/` |
| `normals`、`targets` | 初始方向与共享三维约束 | 对应阶段日志、`shared-targets/` |
| `preflight` | 175 相机数值及支持率检查 | `native-volume-preflight/result.json` |
| `train` | 18,000 步静态高斯优化 | `gaussian-run/progress.json`、`logs/train.log` |
| `export` | 导出、回读和 +Y 坐标检查 | `gaussian-run/delivery.json`、`stages/export.json` |

需要终止正在进行的**高斯 train 阶段**时，可在另一 SSH 窗口给当前这一轮写停止标记：

```bash
touch "$HOME/shuolee/runs/113_001/gaussian-run/STOP"
```

等待训练代码处理标记并保存退出，再确认日志和进程状态。这会得到提前停止的实验，不算 18,000 步完整结果，`all` 不会将它作为完整训练继续交付。不要使用停止其他服务器用户进程的命令。其他拟合阶段的停止方式与这个标记不同。
'''
text=text.replace('## 8. 下载查看器可直接打开的结果', progress+'\n## 8. 下载查看器可直接打开的结果',1)

download=r'''

WinSCP 下载操作：右侧进入本轮 `gaussian-run/viewer-export` → 选择 `point_cloud.ply` 及旁边 JSON → 左侧切换到本地当天的工作日志目录 → 点击下载或按 `F5`。等队列结束后再在查看器打开。

为避免同名模型混淆，下载到本地后可将文件重命名为 `113_001_s18000_viewer.ply`；只改文件名，不修改内部坐标或数据。需要核对传输完整性时：

```bash
# 服务器 SSH 终端
sha256sum "$HOME/shuolee/runs/113_001/gaussian-run/viewer-export/point_cloud.ply"
```

```powershell
# 公司电脑 00 的 PowerShell；把路径替换成实际下载位置
Get-FileHash -Algorithm SHA256 -LiteralPath 'D:\00\工作日志\YYYY-MM-DD\113_001\113_001_s18000_viewer.ply'
```

两端 SHA-256 一致后，文件传输才算完整。日期目录和文件名可以按实际情况改，服务器导出源文件仍使用原名。
'''
text=text.replace('## 9. 当前版本边界与更新方式', download+'\n## 9. 当前版本边界与更新方式',1)

text += r'''

## 10. 常见问题定位

| 现象 | 先检查 | 处理方法 |
|---|---|---|
| WinSCP 超时 / SSH 连不上 | 公司内网/VPN、主机、端口 | 确认能访问服务器；不要把公司电脑 00 的 ToDesk 地址当 SSH 地址 |
| 认证失败 | Linux 用户名、服务器登录凭据 | 用正确服务器凭据；GitHub 登录不能替代 SSH 登录 |
| Git `Repository not found` / 401 | 服务器上的 GitHub 身份 | `gh auth status`；必要时重新 `gh auth login` 和 `gh auth setup-git` |
| `destination path ... already exists` | 是否已经克隆 | 在已有目录核对标签；不删除已有工程来强行克隆 |
| `python3.10: command not found` | Python 安装位置 | 提供 Python 3.10；非默认位置时设置 `PYTHON_BIN` 为其实际可执行文件 |
| `nvcc not found` / 版本不符 | `CUDA_HOME/bin/nvcc` | 改成 CUDA 12.8 Toolkit 的真实位置；不是驱动版本号 |
| `Environment already exists` | 环境是否之前已装好 | 已合格则跳过基础安装；仅对中断安装核对日志后使用 `--resume-install` |
| `ModuleNotFoundError: diff_plane_rasterization` | 当前 `which python`、是否执行第二个安装脚本 | 激活同一 env，再执行 `bash scripts/setup_113.sh`；不把旧二进制复制进来 |
| `Missing input` | 缺失文件的完整路径 | 检查是否传完、是否套了两层目录；按清单补齐 |
| `Input differs from frozen 113` | 对应文件大小与 SHA-256 | 用二进制模式重新传正确原件；不编辑清单让错误输入通过 |
| `Fresh run required` / `Stage already attempted` | 是否重复使用运行目录 | 先看旧日志；完整重跑使用新名称，不读旧 checkpoint |
| `Another ... GPU lock` | 是否已经运行一份同类任务 | 检查当前终端、tmux 和 GPU 进程，不重复启动 |
| CUDA out of memory | GPU 分配、其他进程、失败阶段 | 保存日志并确认可用资源；不要偷偷减点数/分辨率再称同一 113 对照 |
| preflight 未通过 | `native-volume-preflight/result.json` | 保留证据并定位数值/支持率问题，不改 passed 或降低阈值 |
| export 未完成 / PLY 校验失败 | `logs/export.log`、delivery | 暂不使用 checking 文件作交付；先排查完整协方差、SH 和回读检查 |
| 查看器没有更新 | 打开的文件是否本轮下载的 PLY | 用“打开文件”明确选 `113_001_s18000_viewer.ply`，先恢复 Gaussian 1× |

## 11. 遇到问题时需要提供的最少信息

在服务器 SSH 终端执行以下命令，将非敏感输出和报错末尾发回即可；不要发送账号密码、GitHub token 或私钥：

```bash
hostname
cat /etc/os-release
nvidia-smi
cd "$HOME/shuolee/portrait113"
git describe --tags --exact-match
git rev-parse HEAD
which python
python --version
```

再提供失败阶段的日志末尾，例如：

```bash
tail -n 80 "$HOME/shuolee/runs/113_001/logs/train.log"
```

若尚未进入 train，则换成实际失败阶段的日志：例如 `head.log`、`fit.log`、`preflight.log`、`export.log`；环境安装失败看 `$HOME/.local/share/shuolee-portrait/env-cu128/install/setup.log` 或 `setup113.log`。

## 12. 部署完成的判定

1. **Git 已取得**：标签和提交正确，工作区干净。
2. **材料已上传**：audit 对 175 帧和 8 项模型资产检查通过。
3. **环境可用**：doctor 的 CUDA 控制检查通过。
4. **训练完成**：本轮 result 记载从头训练、18,000 步，export 完成。
5. **文件可交付**：下载哈希一致，00 的查看器可分别打开 Gaussian 和 points。
6. **质量另行审阅**：同相机与未知视角的脸部和全身重影是否改善，需要实际查看，不能由前五项代替。

本指南只新增小体积操作说明，没有将模型、数据、环境或旧训练结果复制进 `D:\00\工作\SMPL-X`。原 113 Git 标签保持不变。
'''

text = text.replace(
    '```bash\ntmux new -s shuolee113\ncd "$HOME/shuolee/portrait113"',
    '先单独创建并进入 tmux 会话：\n\n```bash\ntmux new -s shuolee113\n```\n\n看到 tmux 中的新命令提示符后，再复制下面这段。若会话已经存在，先用 `tmux attach -t shuolee113` 查看是否已有任务，不能重复启动。\n\n```bash\ncd "$HOME/shuolee/portrait113"',
    1,
)

out.mkdir(parents=True, exist_ok=True)
md = out / '00_部署与训练指南.md'
md.write_text(text.strip()+'\n', encoding='utf-8', newline='\n')
renderer = markdown.Markdown(extensions=['fenced_code', 'tables', 'toc'], extension_configs={'toc': {'permalink': False, 'toc_depth': '2'}})
body=renderer.convert(text)
page='''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SMPL-X 工作目录 · 113 服务器部署</title><style>
:root{--ink:#182838;--muted:#516579;--blue:#175dd1;--line:#dce5ee;--bg:#f4f7fb}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.8 "Microsoft YaHei","Noto Sans SC",sans-serif}
.layout{max-width:1500px;margin:auto;display:grid;grid-template-columns:270px minmax(0,1fr);gap:28px;padding:32px}
nav{position:sticky;top:20px;align-self:start;max-height:94vh;overflow:auto;background:white;border:1px solid var(--line);border-radius:14px;padding:20px;font-size:14px}
nav strong{display:block;font-size:17px;margin-bottom:12px}nav ul{list-style:none;padding-left:0}nav li{margin:9px 0}nav a{text-decoration:none}
main{background:white;border:1px solid var(--line);border-radius:16px;padding:40px 44px;min-width:0}
h1{font-size:30px;line-height:1.45;margin-top:0}h2{font-size:24px;padding-top:22px;margin-top:36px;border-top:1px solid var(--line);scroll-margin-top:24px}h3{font-size:19px;margin-top:28px}
a{color:var(--blue);overflow-wrap:anywhere}p{margin:16px 0}li{margin:7px 0}code{font:14px/1.65 Consolas,"Microsoft YaHei",monospace;background:#eef3f8;padding:2px 5px;border-radius:4px;overflow-wrap:anywhere}
pre{position:relative;padding:44px 18px 18px;background:#142638;color:#f4f7fa;border-radius:10px;overflow:auto;font:14px/1.75 Consolas,monospace;white-space:pre}
pre code{padding:0;background:none;color:inherit;overflow-wrap:normal}.copy{position:absolute;top:9px;right:12px;border:1px solid #5c7a99;border-radius:6px;background:#284664;color:white;padding:4px 12px;cursor:pointer;font-size:13px}
.copy:hover{background:#375d83}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;margin:18px 0;font-size:14px}td,th{border:1px solid var(--line);padding:11px 13px;vertical-align:top;text-align:left}th{background:#edf3fb}tr:nth-child(even){background:#f9fbfd}
.top-note{color:var(--muted);font-size:13px;border-bottom:1px solid var(--line);padding-bottom:16px;margin-bottom:24px}
@media(max-width:980px){.layout{display:block;padding:14px}nav{position:static;max-height:none;margin-bottom:18px}main{padding:24px}h1{font-size:26px}}
@media print{nav,.copy{display:none}.layout{display:block;padding:0}main{border:0;padding:0}body{background:white;font-size:11pt}pre{white-space:pre-wrap;color:#111;background:#f3f3f3;break-inside:avoid}table{font-size:10pt}h2,h3{break-after:avoid}}
</style></head><body><div class="layout"><nav><strong>按顺序部署 113</strong>'''+renderer.toc+'''</nav><main><div class="top-note">离线说明 · 2026-10-26 · Git 固定标签 v113.0.0-reproduction · 代码由 Git 传输，训练在服务器执行</div>'''+body+'''</main></div>
<script>
document.querySelectorAll('table').forEach(t=>{const w=document.createElement('div');w.className='table-wrap';t.before(w);w.append(t)});
document.querySelectorAll('pre').forEach(pre=>{if(!pre.querySelector('code'))return;const b=document.createElement('button');b.className='copy';b.type='button';b.textContent='复制';b.addEventListener('click',async()=>{const s=pre.querySelector('code').textContent;try{if(!navigator.clipboard)throw Error('fallback');await navigator.clipboard.writeText(s)}catch(e){const a=document.createElement('textarea');a.value=s;document.body.append(a);a.select();document.execCommand('copy');a.remove()}b.textContent='已复制';setTimeout(()=>b.textContent='复制',1500)});pre.append(b)});
</script></body></html>'''
ht=out/'00_部署与训练指南.html'
ht.write_text(page,encoding='utf-8',newline='\n')
# Readable output only, no downloads, login, training or cloud-sync claims.
assert '5fda749498cca899fb63011dc7b1c71511c14050' in text
assert 'scripts/route113.py all' in text and 'viewer-export/point_cloud.ply' in text
assert len(re.findall(r'^```',text,re.M))%2==0
assert all('https://' in s or s.startswith('#') for s in re.findall(r'href="([^"]+)"',page))
print(json.dumps({'markdown':str(md),'html':str(ht),'bytes':md.stat().st_size+ht.stat().st_size,'sections':len(re.findall(r'^## ',text,re.M))},ensure_ascii=False))
