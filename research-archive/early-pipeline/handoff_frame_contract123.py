"""Create a small, immutable code handoff for the user-authorized 00 task."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil, zipfile
H=Path(__file__).parent
CO=H.parent/'shuolee-coordination'
P=Path('D:/GS-SLAM-Data/shuolee-portrait-new-capture-118')
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
A=T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route/training-interface-audit'
OUT=CO/'handoffs/frame-contract-123-20261029'
OUT.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
stamp=datetime.now(timezone.utc).isoformat()
files=['src/shuolee_portrait/frame_pose_contract.py','src/shuolee_portrait/camera_pixel_contract.py',
       'src/shuolee_portrait/alignment_inputs.py','scripts/check_frame_pose_inputs.py',
       'tests/test_frame_pose_contract.py','docs/逐帧高斯输入约定.md']
checks=read(A/'cpu-tests-final.json')
assert checks['passed'] and checks['tests_run']==27
assert checks['source_sha256']==sha(P/files[0]) and checks['test_sha256']==sha(P/files[4])
snapshot=OUT/'source'
records=[]
for name in files:
    dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(P/name,dest);assert sha(dest)==sha(P/name)
    records.append(dict(path=name,sha256=sha(dest),bytes=dest.stat().st_size))
shutil.copy2(A/'cpu-tests-final.json',OUT/'cpu-tests-final.json')
roadmap='''# 给00对话的部署与验证路线

用户本轮明确要求：将已经跑通的路线图同步给00窗口，让其做Git部署、实验，并反馈给原对话。

## 已跑通的部分与边界

| 阶段 | 实际证据 | 当前能交付什么 |
|---|---|---|
| 原生图像与完整K的像素约定 | 已有camera_pixel_contract；本轮解析检查保留偏心主点/skew并只转换一次半像素 | 通用坐标接口，不证明物理标定 |
| SMPLify-X共享体型、逐帧姿态 | 三帧真实拟合曾执行，但几何/脸部质量未通过；合成控制已执行 | 研究代码，不能称实拍已验收 |
| 根位姿初始化 | A/B合成几何结果几乎一致，梯度收敛未通过 | 原因分析结论，不重复启动 |
| 帧ID—图像—相机—网格姿态连接 | 本轮27项CPU检查含公共CLI全部通过，文件哈希绑定 | 本次小型代码包可提交独立Git分支并跨机器验证 |
| 共享自由衣物高斯随姿态运输 | 旧mesh adapter有基础能力；自由衣物点和正式训练连接尚未完成 | 暂不将新模块当成完整逐帧训练器 |
| 参考姿态静态PLY+camera | 仍需要新路线端到端协方差/SH/渲染回读验收 | 不能发布新的“已解决重影”模型 |

## 00负责的具体工作

1. 核验source-manifest.json和六个source文件哈希。本包不含视频、图片、权重或环境。
2. 检查你管理的实际仓库、分支和未提交改动。对 `https://github.com/lizuoshuo-lab/shuolee-portrait.git` 使用独立 `codex/frame-pose-contract-123` 分支或适合的隔离worktree，不覆盖113稳定版或正在运行代码。不重置他人更改。
3. 比较依赖和目标现状再移植六个文件。它们依赖Python 3.10、NumPy、Pillow，本次检查不需要CUDA。保留现有安装方式，不再安装大权重。
4. 在该隔离checkout运行：

```bash
python -B tests/test_frame_pose_contract.py --report frame-pose-tests.json
python -B scripts/check_frame_pose_inputs.py --help
```

成功标志是27项检查通过且JSON `passed: true`。`training_admitted: false`是正确结果，不能为跑通改成true。

5. 如有已经保存的逐帧输入，只做独立只读manifest接入检验：

```bash
python -B scripts/check_frame_pose_inputs.py /path/to/frame-pose-manifest.json
```

不补造视频ID、帧号、内参/外参、mask、体型或共享beta。旧静态结果也不能伪装成逐帧拟合。缺字段保存清单并反馈。

6. 检查源包和测试结果后提交推送私有仓库，回报commit、分支、文件哈希和完整测试日志。如果目标服务器没有可用授权连接，完成本地Git准备并给用户一条可复制命令，明确未实际部署服务器。
7. 本轮不重新运行相机/人体拟合或高斯训练，不改已有静态保护。后续真实实验由双方对齐质量证据与接口后开展，不能跳过已失败的实拍门槛。

## 反馈给原对话

原对话ID：01a0e316-7534-7693-a60c-ded52094785d。
00对话维护company00-status.json；原对话维护local-research-status.json，避免互相覆盖。

反馈至少包括：仓库/commit/分支、Python/NumPy/Pillow版本、实际执行主机、27项测试日志与JSON、真实输入接入结果或缺字段、修改和失败原因、服务器是否真正验证。只在实质完成/失败或需要决策时发回一次，不发确认循环。
'''
(OUT/'部署路线.md').write_text(roadmap,encoding='utf-8')
manifest=dict(schema='shuolee-code-handoff/1',created=stamp,source_project=str(P),
    source_head='06609e5ece8d8f2a2d75a08dfd6172ea52d5a349',source_branch='shuolee/new-capture-118',
    source_is_dirty=True,git_remote='https://github.com/lizuoshuo-lab/shuolee-portrait.git',files=records,
    tests=checks,scope='Validated CPU input contract only, not a complete quality-accepted training route.',
    source_user_authorization='将已经跑通的路线图同步给00窗口，让他做git上的部署以及实验，并作反馈给你',
    target_thread='01a17c8c-e70f-7822-ad39-440290b3d1da',return_thread='01a0e316-7534-7693-a60c-ded52094785d',
    expected_target_branch='codex/frame-pose-contract-123',GPU_training_authorized_by_this_package=False,
    physical_quality_passed=False,contains_subject_data=False,contains_weights=False)
save(OUT/'source-manifest.json',manifest)
zip_path=OUT/'frame-contract-123-source.zip'
with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
    for name in files:archive.write(snapshot/name,'source/'+name)
    for name in ['source-manifest.json','cpu-tests-final.json','部署路线.md']:archive.write(OUT/name,name)
with zipfile.ZipFile(zip_path) as archive:
    for record in records:assert hashlib.sha256(archive.read('source/'+record['path'])).hexdigest()==record['sha256']
save(OUT/'handoff-receipt.json',dict(created=stamp,zip=str(zip_path),sha256=sha(zip_path),bytes=zip_path.stat().st_size,files=len(files),Git_uploaded=False,target_executed=False))
p=CO/'local-research-status.json';s=read(p)
s.update(updated_at=stamp,company00_requested_work=dict(handoff=str(OUT/'source-manifest.json'),roadmap=str(OUT/'部署路线.md'),
    code_archive=str(zip_path),state='prepared_for_authorized_git_deployment_and_CPU_validation',
    feedback_required=True,quality_accepted=False))
save(p,s)
print(json.dumps(dict(handoff=str(OUT),zip=str(zip_path),sha256=sha(zip_path),bytes=zip_path.stat().st_size),ensure_ascii=False))
