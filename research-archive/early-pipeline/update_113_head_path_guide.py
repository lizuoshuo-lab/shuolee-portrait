"""Publish the checked head-path compatibility repair into the existing guide."""
from pathlib import Path
import ast
import json
import re
import markdown

ROOT = Path(__file__).parent
OUT = Path('D:/00/工作/SMPL-X')
CACHE = Path('D:/GS-SLAM-Data/shuolee-portrait-transfer-cache/pip-bootstrap')


def main():
    for name in ['head-source-path-validation.json', 'training-launcher-validation.json']:
        proof = json.loads((CACHE/name).read_text(encoding='utf-8'))
        assert proof['passed'], name
        assert not proof['real_cuda_or_training_tested'], name
    path = OUT/'00_部署与训练指南.md'
    text = path.read_text(encoding='utf-8-sig')
    assert '### 当前1：修复head源码路径' not in text, 'Already published; edit in place instead of adding twice.'
    title, old = text.split('\n\n', 1)
    current = r'''

**最新：已读取公司端 `head.log` 和 `head.json`。本轮 `prepare` 成功，`head` 在记录FLAME源码SHA时失败：它仍查找 `preparation/flame_model/flame.py`，实际源码已移到 `avatar_core/flame_model/flame.py`。不是模板权重缺失；日志已走过FLAME前向与初始化计算。**

`113_001` 已写入部分初始化结果，保留全部文件，不删除、不改名、不续跑。本次用小型路径兼容修复，再使用全新的 **`113_002`**。不改Git113基线 `5fda749`、跟踪源码、清单、相机、输入、模板或训练参数。

### 当前1：修复head源码路径

公司电脑00等WPS同步后，从 **`00 → 工作 → SMPL-X`** 用WinSCP二进制模式上传以下4个小文件，覆盖到 **`/home/zhouhaoyi/shuolee/`**：

- `server113_head_path_fix.py`
- `server113_head_path_fix.py.sha256`
- `server113_train.py`（新版支持 `--run-name`）
- `server113_train.py.sha256`

若本工程还有运行任务，先等它结束。然后在WinSCP“执行命令”复制整段：

```bash
cd ~/shuolee &&
tr -d '\r' < server113_head_path_fix.py.sha256 | sha256sum -c - &&
tr -d '\r' < server113_train.py.sha256 | sha256sum -c - &&
python3.10 server113_head_path_fix.py --apply
```

**看什么：** 两个文件都显示 `OK`，最后出现 **`HEAD_SOURCE_PATH_FIX_READY`**。工具先核对完整Git版本与Linux Git字节SHA；确认没有安装/训练任务后，只建立兼容目录链接 `preparation/flame_model → ../../avatar_core/flame_model`，使错误引用访问到同一份原始FLAME源码。不会复制权重或启动计算；重复执行正确链接时为幂等成功。

若出现 `HEAD_SOURCE_PATH_FIX_STOPPED`，保存全部输出并停下。已有普通目录、错误链接、源码SHA不符或任务仍运行时都不会覆盖。不要自己新建空的 `flame.py`，也不需要重装环境。

### 当前2：用新编号从头执行113

只有上一步成功后，复制：

```bash
python3.10 ~/shuolee/server113_train.py --start --run-name 113_002
```

**看什么：** `PIPELINE_STARTED PID=...` 和 `Run: /home/zhouhaoyi/shuolee/runs/113_002`。这只代表后台流程启动；仍先核查安装状态、Git基线、GPU0空闲，随后重新执行audit、doctor、prepare和全部拟合/训练步骤。原 `113_001` 不读取为初值，也不覆盖。

新版会把旧的总状态逐字归档到 `~/shuolee/logs/server113-state-history/`，总日志继续追加。`113_002` 已存在时仍会拒绝启动，不自动删除、续跑或换编号。

查询进度仍用：

```bash
python3.10 ~/shuolee/server113_train.py --status
```

**看什么：** `revision` 为 **`2026-10-27-winscp-training-2-fresh-run-name`**，`run` 指向 **`113_002`**。`stage: all` 包含准备、头部拟合、身体拟合等前序阶段；只有进入高斯训练后才有高斯步数。`training_started: false` 在旧设计中一直保留到最终导出成功，不能单独用来判断当前是否已经进行拟合或高斯更新。

当前失败日志位置改为 `~/shuolee/runs/113_002/stages/` 与 `logs/`；再失败先保存总状态/总日志以及对应阶段的JSON和日志，不重复 `--start`。完整下载与质量复核按本页第7–8步，那里已改为 `113_002`。

**验证范围：** 本地隔离环境已核验路径修复、Git LF身份校验、错误目标拒绝、幂等、并发拒绝、新编号透传、旧实验保留和状态归档；未在公司服务器执行本次修复，也未完成真实头部拟合或训练。`HEAD_SOURCE_PATH_FIX_READY` 不等于拟合通过，`PIPELINE_STARTED` 不等于训练完成。

---

**以下为此前CUDA补装和完整部署参考。当前已定位到head源码路径错误，先按上面的“当前1–2”操作，不重复此前的CUDA补装。**

'''
    text = title+current+old
    start = text.index('\n## 6. ')
    prefix, tail = text[:start], text[start:]
    tail = tail.replace('113_001', '113_002')
    tail = tail.replace('python3.10 ~/shuolee/server113_train.py --start\n',
                        'python3.10 ~/shuolee/server113_train.py --start --run-name 113_002\n')
    text = prefix+tail
    html_path = OUT/'00_部署与训练指南.html'
    html = html_path.read_text(encoding='utf-8-sig')
    body = markdown.markdown(text, extensions=['fenced_code', 'tables'])
    html, count = re.subn(r'(?s)(<main>).*?(</main>)', lambda m:m[1]+body+m[2], html, count=1)
    assert count == 1
    path.write_text(text, encoding='utf-8', newline='\n')
    html_path.write_text(html, encoding='utf-8', newline='\n')
    generator = ROOT/'simplify_smplx_deployment_guide.py'
    code = generator.read_text(encoding='utf-8-sig')
    node = next(n for n in ast.parse(code).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'content' for t in n.targets))
    lines = code.splitlines(keepends=True)
    lines[node.lineno-1:node.end_lineno] = ['content = '+repr(text)+'\n']
    generator.write_text(''.join(lines), encoding='utf-8', newline='\n')
    print(json.dumps({'guide':str(html_path), 'server_verified':False, 'training_started':False}))


if __name__ == '__main__':
    main()
