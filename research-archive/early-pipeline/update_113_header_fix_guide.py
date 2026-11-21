"""Update the existing deployment guide from the received compiler evidence."""
from pathlib import Path
import ast, hashlib, json, re, shutil
import markdown
HELP=Path(__file__).parent;OUT=Path('D:/00/工作/SMPL-X')
CACHE=Path('D:/GS-SLAM-Data/shuolee-portrait-transfer-cache/pip-bootstrap')
proof=json.loads((CACHE/'header-patch-validation.json').read_text())
assert proof['passed'] and proof['actual_cpp_after_passed']
log=Path('C:/Users/Administrator/Downloads/报错.md')
assert 'fatal error: crt/host_config.h: No such file or directory' in log.read_text(encoding='utf-8-sig')
saved=Path('D:/00/工作/报错/113_cuda头文件_20261027.md');shutil.copy2(log,saved)
text=(OUT/'00_部署与训练指南.md').read_text(encoding='utf-8-sig')
start=text.index('\n\n');end=text.index('你这份旧版CUDA脚本')
intro=r'''

**2026-10-27 更新：已读取小润10:31发来的《报错.md》。具体错误是 `cuda_runtime.h:82:10: fatal error: crt/host_config.h: No such file or directory`。CUDA12.8编译器已安装；这次是安装脚本遗漏了开发头文件搜索路径。下面按顺序操作，不重装CUDA、不删除Python环境。**

### 现在做1：上传更新包，重新启动依赖补装

在公司电脑00等WPS同步后，从 **`00 → 工作 → SMPL-X`** 取这两个文件，用WinSCP **二进制模式** 上传并覆盖到 **`/home/zhouhaoyi/shuolee/`**：

- **`server113_pip_repair.pyz`**（新版约3.05MB）
- **`server113_pip_repair.pyz.sha256`**

然后在WinSCP“执行命令”一次复制下面**整行**。只有校验OK才会开始补装；不用tmux，不用activate，不运行训练。

```bash
cd ~/shuolee && tr -d '\r' < server113_pip_repair.pyz.sha256 | sha256sum -c - && python3.10 server113_pip_repair.pyz --install-background
```

**看什么：** 先出现 `server113_pip_repair.pyz: OK`，再出现 `INSTALL_STARTED PID=...`。若是 `INSTALL_ALREADY_RUNNING`，只查看状态，不能再开安装。执行窗口可以关闭。

过一会复制这一行查看进度：

```bash
python3.10 ~/shuolee/server113_pip_repair.pyz --status
```

**看什么：** `revision` 应为 **`2026-10-27-offline-pip-3-cuda-headers`**。`state: running` 表示仍在安装；只有 **`state: completed`** 且末尾出现 **`DEPENDENCY_INSTALL_FINISHED_CONTINUE_GUIDE_STEP_5`** 才进入第5步。日志中的 `CUDA_HOST_HEADERS_OK` 只代表C++头文件探针通过，不代表全部依赖安装完成。

新版会备份并修补113的两个**安装脚本**，增加 `cuda_build_env.py`，把工具包 `include`／`targets/x86_64-linux/include` 放在pip运行时头文件前面，并验证 `crt/host_config.h`。Git113基线仍是 **`5fda749`**；训练算法、相机、数据、权重版本不变。原脚本备份保存在服务器 `~/shuolee/logs/installer-before-cuda-header-patch/`。如果发现未知的本地修改，会停止而不是覆盖。

**如果失败需要保存什么：** 用WinSCP下载下列文件到 **`00 → 工作 → 报错`**，发出最新末尾错误即可；不要因为旧日志有READY就直接训练。

| 服务器文件 | 用途 |
|---|---|
| `~/.local/share/shuolee-portrait/env-cu128/install/setup.log` | 依赖和编译完整日志 |
| `~/shuolee/logs/server113-install.log` | 后台安装日志 |
| `~/.local/share/shuolee-portrait/env-cu128/install/cuda-build-headers.json` | 头文件路径和C++探针结果，成功生成时保存 |

这里的 `~` 是 `/home/zhouhaoyi`。本机已在隔离测试中复现同一个缺头文件错误，并验证补丁后C++编译通过；**公司服务器的完整nvdiffrast编译和GPU运行仍以本次安装、后续doctor为准**。

### 现在做2：补齐113观测文件，再做第5步检查

公司电脑00的 **`00 → 工作 → SMPL-X`** 还有下面两个文件：

- **`shared-observations.npz`**：7,341,419字节，113原有二维观测与固定相机，不含旧高斯或拟合结果。
- **`shared-observations.npz.sha256`**：校验文件。

WinSCP右侧路径栏输入 **`/home/zhouhaoyi/shuolee/datasets/hover175/route113`**，把这两个文件用二进制模式传到这里。已经上传的只需校验，不重复创建。没有目录时先执行：

```bash
mkdir -p ~/shuolee/datasets/hover175/route113
```

上传后复制：

```bash
cd ~/shuolee/datasets/hover175/route113 && tr -d '\r' < shared-observations.npz.sha256 | sha256sum -c -
```

**看什么：** `shared-observations.npz: OK`。它必须在 `datasets/hover175/route113`，不能放到 `models/113`。不要创建空文件或替换成新视频的观测。

**下一步：** 两项都成功 → 按下方第5步执行audit/doctor → 全部通过后才按第6步开始训练。`audit`失败时，`&&`后面的doctor实际没有执行。补装只续接依赖安装；高斯训练仍从头开始。

'''
text=text[:start]+intro+text[end:]
text=text.replace('最新安装停在nvdiffrast编译，具体原因等待完整编译日志。','最新日志已定位到CUDA开发头文件路径遗漏；修复包加入路径和真实C++编译探针。')
template=(OUT/'00_部署与训练指南.html').read_text(encoding='utf-8-sig')
body=markdown.markdown(text,extensions=['fenced_code','tables'])
page=re.sub(r'(?s)(<main>).*?(</main>)',lambda m:m[1]+body+m[2],template,count=1)
for ext,value in [('md',text),('html',page)]:
    (OUT/f'00_部署与训练指南.{ext}').write_text(value,encoding='utf-8',newline='\n')
generator=HELP/'simplify_smplx_deployment_guide.py';code=generator.read_text(encoding='utf-8-sig')
tree=ast.parse(code);node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='content' for t in n.targets))
lines=code.splitlines(keepends=True);lines[node.lineno-1:node.end_lineno]=['content = '+repr(text)+'\n']
generator.write_text(''.join(lines),encoding='utf-8',newline='\n')
receipt=dict(received_log=str(saved),received_log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
    cause='setup_server.sh/ setup_113.sh overwrite CPATH using only pip nvidia include dirs; split-toolkit targets include omitted from host C++ compilation.',
    package_sha256=hashlib.sha256((OUT/'server113_pip_repair.pyz').read_bytes()).hexdigest(),
    local_validation=proof,company_server_verified=False,server_training_started=False,
    guide=str(OUT/'00_部署与训练指南.html'))
(HELP/'113-build-observation-update.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(guide=receipt['guide'],local_cpp_passed=True,server_verified=False),ensure_ascii=False))
