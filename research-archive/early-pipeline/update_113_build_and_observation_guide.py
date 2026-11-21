"""Publish the exact missing observation, and document the unresolved build error."""
from pathlib import Path
import ast, hashlib, json, re, shutil
import markdown
P=Path('D:/00/工作/人体重建')
OUT=Path('D:/00/工作/SMPL-X')
HELP=Path(__file__).parent
spec=json.loads((P/'configs/route113-inputs.json').read_text(encoding='utf-8-sig'))
row=next(x for x in spec['data_files'] if x['path']=='route113/shared-observations.npz')
src=P/'data/hover175'/row['path'];payload=src.read_bytes()
assert len(payload)==row['bytes'] and hashlib.sha256(payload).hexdigest()==row['sha256']
dst=OUT/src.name
if not dst.exists() or hashlib.sha256(dst.read_bytes()).hexdigest()!=row['sha256']:shutil.copy2(src,dst)
(OUT/(src.name+'.sha256')).write_text(row['sha256']+'  '+src.name+'\n',encoding='ascii',newline='\n')
content=(OUT/'00_部署与训练指南.md').read_text(encoding='utf-8-sig')
start=content.index('\n\n')
end=content.index('你这份旧版CUDA脚本')
intro=r'''

**2026-10-27 最新处理：依赖安装在 `nvdiffrast` 编译处失败；随后材料审计又发现 `shared-observations.npz` 未上传。现在停在安装/传输检查，没有开始训练。先做下面两项，不再按旧日志中的 PID 5090 等待，不直接跳到第6步。**

### 现在做1：补传一个7.34 MB的文件

公司电脑00等WPS同步后，在 **`00 → 工作 → SMPL-X`** 找到这两个文件：

- **`shared-observations.npz`**：7,341,419字节，113原有跨视角二维观测和固定相机，不含旧拟合或旧高斯。
- **`shared-observations.npz.sha256`**：校验文件。

WinSCP右侧路径栏输入 **`/home/zhouhaoyi/shuolee/datasets/hover175/route113`**，将这两个文件用**二进制模式**上传到这里。不要放进 `models/113`，也不要再套一层 `route113` 文件夹。右侧没有目录时先执行：

```bash
mkdir -p ~/shuolee/datasets/hover175/route113
```

上传后在WinSCP“执行命令”复制：

```bash
cd ~/shuolee/datasets/hover175/route113 && tr -d '\r' < shared-observations.npz.sha256 | sha256sum -c -
```

**看什么：** `shared-observations.npz: OK`。本机已核对文件与Git113冻结清单完全相同；这只补齐已明确缺失的文件，整个数据集是否完整仍由第5步audit验证。不要创建空NPZ或用其他视频观测代替。

### 现在做2：取出nvdiffrast真正的编译报错

`Failed building wheel for nvdiffrast`、`Install failed at line 99` 和外层 `CalledProcessError` 只是失败摘要。当前还不能判断是C++/CUDA编译器、头文件、内存或版本问题；不要据此重装CUDA或换依赖版本。

在WinSCP“执行命令”复制这一行，把输出文字保存到 `00 → 工作 → 报错`，或直接贴到聊天：

```bash
grep -n -B 8 -A 20 -m 3 -E 'fatal error:|nvcc fatal|error:|Killed|undefined reference' ~/.local/share/shuolee-portrait/env-cu128/install/setup.log
```

**看什么：** 编译器指出的第一个具体错误及前后行。如果只输出旧的缺pip信息，或者没有输出，请直接用WinSCP下载下面的完整文件，不重复安装：

| 服务器右侧文件 | 保存到00时的文件名 |
|---|---|
| `/home/zhouhaoyi/.local/share/shuolee-portrait/env-cu128/install/setup.log` | `113_nvdiffrast_setup.log` |
| `/home/zhouhaoyi/shuolee/logs/server113-install.log` | `server113-install.log` |

将这两个日志放入 **`00 → 工作 → 报错`**。我根据具体编译错误修复后，再使用原入口补装；保留现有环境和已安装的PyTorch/pip，不重建、不运行训练。

**此刻不要执行第5步doctor或第6步训练。** 缺输入文件不会导致nvdiffrast编译失败，两项必须分别解决。`audit`失败也意味着后面用`&&`连接的`doctor`尚未执行。

**安装修好后的顺序：** 原入口 `--install-background` 补装 → `--status` 显示 `completed` 且 `DEPENDENCY_INSTALL_FINISHED_CONTINUE_GUIDE_STEP_5` → 第5步audit/doctor → 第6步后台训练。它只接着安装依赖，不是接着旧高斯训练。

```bash
python3.10 ~/shuolee/server113_pip_repair.pyz --status
```

若上传/复制时校验文件出现 `$'\r'`，使用本页带 `tr -d '\r'` 的校验命令；不跳过SHA-256。WinSCP没有交互终端，继续使用后台短命令，不用 `tmux new` 或缺失的 `bin/activate`。

'''
content=content[:start]+intro+content[end:]
content=content.replace('**第一份：图片等输入。** 将 `D:\\00\\工作\\人体重建\\data\\hover175` 整个文件夹上传到服务器 `datasets`，得到 `datasets/hover175`。',
    '**第一份：图片等输入。** 将 `D:\\00\\工作\\人体重建\\data\\hover175` 整个文件夹上传到服务器 `datasets`，得到 `datasets/hover175`。包含它的 `route113` 子目录，尤其是 `route113/shared-observations.npz`；不能只传images和masks。已经漏传该文件时按页首补传。')
content=content.replace('- `datasets/hover175/training-inputs.json`','- `datasets/hover175/training-inputs.json`\n- `datasets/hover175/route113/shared-observations.npz`')
old='**本次改动与验证：** Git113仍为 **`5fda749`**。最新截图证明pip已修好，但依赖补装被旧入口的非交互检查停止；本次入口支持WinSCP后台安装，不启动训练。没有在公司服务器代执行，实际GPU和113模块仍须第5步验证。'
content=content.replace(old,'**本次改动与验证：** Git113仍为 **`5fda749`**。pip修复和后台入口已工作；最新安装停在nvdiffrast编译，具体原因等待完整编译日志。补传文件已与冻结113清单校验一致。尚未在公司服务器完成依赖或训练验证。')
assert '后台安装进程5090仍在运行' not in content
template=(OUT/'00_部署与训练指南.html').read_text(encoding='utf-8-sig')
body=markdown.markdown(content,extensions=['fenced_code','tables'])
page=re.sub(r'(?s)(<main>).*?(</main>)',lambda m:m.group(1)+body+m.group(2),template,count=1)
for suffix,value in [('md',content.strip()+'\n'),('html',page)]:
    (OUT/f'00_部署与训练指南.{suffix}').write_text(value,encoding='utf-8',newline='\n')
# Keep the existing generator consistent so a later update cannot restore stale status.
generator=HELP/'simplify_smplx_deployment_guide.py';code=generator.read_text(encoding='utf-8-sig')
tree=ast.parse(code);assign=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='content' for t in n.targets))
lines=code.splitlines(keepends=True);lines[assign.lineno-1:assign.end_lineno]=['content = '+repr(content)+'\n']
generator.write_text(''.join(lines),encoding='utf-8',newline='\n')
proof=dict(data_bytes=row['bytes'],data_sha256=row['sha256'],frozen_git='5fda749',
    concrete_compiler_error_received=False,server_dependency_install_passed=False,training_started=False,
    error_categories=['nvdiffrast build failure, cause not in excerpt','missing 113 observation file'],
    files=[str(dst),str(OUT/'00_部署与训练指南.html')])
(HELP/'113-build-observation-update.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(proof,ensure_ascii=False))
