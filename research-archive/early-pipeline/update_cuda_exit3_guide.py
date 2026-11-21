"""Clarify the supported exit-3 path in the existing eight-step guide."""
from pathlib import Path
import hashlib,json,re

BASE=Path('C:/Users/Administrator/Documents/New project/shuolee-slide-update')
path=BASE/'simplify_smplx_deployment_guide.py'
source=path.read_text(encoding='utf-8')
before=source
start=source.index('**2026-10-26 更新：')
end=source.index('`Could not resolve host:', start)
intro=r'''**2026-10-26 更新：你反馈的是“返回码 3，找不到 CUDA”。在本页配套脚本中，`CUDA_12_8_NOT_FOUND` + 返回码 3 表示：这次检查没有找到可用的 CUDA Toolkit 12.8。检查命令默认不会安装，所以重复运行它仍会返回 3。**

**现在直接做第 4A：复制带 `--install` 的整段命令。** 看到 `CUDA_12_8_READY=实际路径` 和 `CUDA_ENV_READY` 后，再做 4B 安装工程依赖。没有这两个标志，不执行 `source cuda128.env`，也不开始训练。

已收到的旧日志只确认服务器存在 **CUDA 11.4.152**，并曾开始补装 12.8.1；尚没有收到安装完成记录。**不是让你重新传数据或重新克隆 Git，也不是改 113 版本。** 本次已根据你反馈的返回码更新操作；完整的新服务器日志仍未收到，不能据此声称服务器已修好。

'''
source=source[:start]+intro+source[end:]
start=source.index('## 4. 安装环境：只做一次')
end=source.index('## 5. 检查上传的材料和显卡环境',start)
section=r'''## 4. 安装 CUDA 和工程环境：返回码 3 从这里继续

以下命令全部在**已登录服务器的 SSH／PuTTY 交互终端**执行。不要在 WinSCP“执行命令”窗口里运行。WinSCP 只负责传文件。

### 4A. 找不到 CUDA：执行安装，不再重复只查找

**需要的文件：** 服务器 `~/shuolee/` 中已有 `server_cuda128.sh` 和 `server_cuda128.sh.sha256` 就直接复制下面命令。若尚未上传，用 WinSCP 从 `00 → 工作 → SMPL-X` 上传这两个文件，二进制模式。不要覆盖另一窗口里仍在执行的脚本。

**复制这一整段：**

```bash
(
set -euo pipefail
cd "$HOME/shuolee"
sha256sum -c server_cuda128.sh.sha256
bash server_cuda128.sh --install
test -s "$HOME/shuolee/cuda128.env"
echo CUDA_ENV_READY
)
```

**看什么：**

1. 先出现 **`server_cuda128.sh: OK`**，表示传输校验通过。
2. 没找到12.8时，出现 **`Installing official NVIDIA toolkit 12.8.1`**，然后下载、安装。`Collecting package metadata ... working...` 或 `Solving environment` 是安装过程，**此时等待，不再另开一份安装**。
3. 最后同时出现 **`CUDA_12_8_READY=实际路径`** 和 **`CUDA_ENV_READY`**，才进入4B。脚本会用实际12.8编译器编译一个小样例，通过后自动生成环境文件。

**安装到哪里：** `~/.local/share/shuolee-portrait/cuda-toolkit-12.8`。使用服务器已有 Conda 和 NVIDIA 官方软件源；保留现有11.4，不修改驱动，不改旧Python环境。已找到可用12.8时，`--install`会直接核验，不重复下载安装。

**不要做：** 不要反复执行不带 `--install` 的 `bash server_cuda128.sh` 来期待它安装；不要创建空的 `cuda128.env`；不要把11.4目录写成12.8；不要删除现有CUDA目录。

### 4B. CUDA 成功后：安装工程依赖

**只有4A的两个成功标志都出现后，复制这一整段：**

```bash
(
set -euo pipefail
cd "$HOME/shuolee/portrait113"
test -s "$HOME/shuolee/cuda128.env"
source "$HOME/shuolee/cuda128.env"
"$CUDA_HOME/bin/nvcc" --version
export CUDA_VISIBLE_DEVICES=0 MAX_JOBS=2
bash scripts/setup_server.sh
bash scripts/setup_113.sh
)
```

**看什么：** 开头是 **`release 12.8`**，最后是 **`113 rasterizer imported`**。看到最后这个标志，继续第5步。这里仍需下载Python/PyTorch依赖；Git离线包不含这些安装包。

如果提示 **`Environment already exists`**，不要删除环境：此前已看到 `113 rasterizer imported` 的直接去第5步；否则保存 `~/.local/share/shuolee-portrait/env-cu128/install/setup.log`，先确定上次中断位置。

### 4C. 没成功时：只做对应的一项

- **返回3且显示 `CUDA_12_8_NOT_FOUND`：** 先核对你复制的是4A中**带 `--install`**的命令。若确认带了仍返回3，按下面“保存诊断”操作；不反复安装。
- **返回8，`CUDA_INSTALL_IN_PROGRESS`：** 另一个安装或检查进程还在运行。等原窗口结束，不重复启动。想看进度，另开一个SSH终端复制下面两行。
- **返回5，`destination already exists but did not pass the check`：** 个人目录已经存在，但尚未通过检查，可能是不完整安装。**保留目录**，保存诊断；不要手动删掉重装。
- **返回4，`CONDA_NOT_FOUND`：** 自动安装所需Conda没有找到。请服务器管理员提供现成的CUDA12.8路径，或安装工具包。不能以其他版本替代。
- **返回7，`CUDA_12_8_COMPILE_FAILED`：** 找到了12.8，但实际编译失败，需看完整日志；不是单纯PATH问题。
- **校验FAILED、网络错误或其他返回码：** 停在当前步骤，保存终端最后的报错及下面的日志。不要继续4B。

**仅查看正在安装的进度：**

```bash
ps -u "$(id -u)" -o pid,etime,args | grep -E '[c]onda (create|install).*cuda-toolkit|[b]ash [^ ]*server_cuda128\.sh' || true
tail -n 40 "$HOME/shuolee/logs/cuda128-check.log"
```

**保存诊断（4A没有成功时才需要）：** 使用本目录新版脚本；下面只读检查，不再安装、不改环境。

```bash
(
set -euo pipefail
cd "$HOME/shuolee"
sha256sum -c server_cuda128.sh.sha256
bash server_cuda128.sh --diagnose
)
```

**需要保存什么：** 用WinSCP从服务器下载下面两个文件，放进 **`00 → 工作 → 报错`**：

- **`shuolee/logs/cuda128-check.log`**：安装及编译完整日志。
- **`shuolee/logs/cuda128-diagnostic.txt`**：检测到的编译器版本、头文件、安装目录和最近日志。

`DIAGNOSTIC_ONLY`只代表诊断完成，不能代替4A的成功标志。日志中引用的历史READY也不代表本轮成功。`nvidia-smi`顶部的CUDA版本只是驱动支持能力，不等于`nvcc`工具包版本。

若管理员提供了**确切的现成12.8目录**，可运行 `bash "$HOME/shuolee/server_cuda128.sh" --toolkit /管理员提供的实际路径`；编译成功并生成环境文件后，再做4B。官方工具包说明：[NVIDIA CUDA 12.8.1安装指南](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-installation-guide-linux/index.html)。

**本次改动与验证：** Git 113仍是 **`5fda749`**，数据和训练步骤不变。安装辅助脚本已在本机验证真实CUDA12.8编译、缺失／旧版编译器分类和重复安装保护；公司服务器的安装结果仍以4A、4B的实际输出为准。

'''
source=source[:start]+section+source[end:]
for n in [1,2,3,5,6,7,8]:
    old=re.search(rf'(?m)^## {n}\. .*?(?=^## \d+\.|\Z)',before,re.S).group()
    new=re.search(rf'(?m)^## {n}\. .*?(?=^## \d+\.|\Z)',source,re.S).group()
    assert old==new, f'Unrelated step changed: {n}'
path.write_text(source,encoding='utf-8',newline='\n')
(BASE/'cuda-exit3-guide-change.json').write_text(json.dumps(dict(
    evidence='User reports return code 3 and CUDA not found; no newer server log received.',
    old_generator_sha256=hashlib.sha256(before.encode()).hexdigest(),
    new_generator_sha256=hashlib.sha256(source.encode()).hexdigest(),
    unrelated_steps_unchanged=[1,2,3,5,6,7,8], frozen_git_113_changed=False,
    server_installation_verified=False),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Updated top instructions and step 4; seven other steps unchanged.')
