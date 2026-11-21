"""Freeze the smallest reviewed CPU-unit patch for the authorized 00 handoff."""
from pathlib import Path
import hashlib, json, shutil, zipfile
from datetime import datetime, timezone

H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123'
S=REC/'smplifyx-reference-control/framewise-canonical-route/training-interface-audit/saved-snapshot'
P=Path('D:/GS-SLAM-Data/shuolee-portrait-new-capture-118')
C=H.parent/'shuolee-coordination'
OUT=C/'handoffs/world-units-123-20261029'
OUT.mkdir(parents=True,exist_ok=True)
def lp(p):return Path('\\\\?\\'+str(Path(p).resolve()))
def read(p):return json.loads(lp(p).read_text(encoding='utf-8-sig'))
def save(p,v):lp(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(lp(p).read_bytes()).hexdigest()

# Correct the source pointer only; the reviewed implementation and metrics are unchanged.
p=S/'transport-interface-review.json';v=read(p)
v['source_findings'][2]['path']=str(P/'src/avatar_core/scene/clothing_gaussian_model.py')
assert Path(v['source_findings'][2]['path']).is_file();save(p,v)
for d in [T/'shuolee/current/robust-portrait-122/video10-global-initializer',T/'viewer/diagnostics/video10-global-initializer',Path('D:/00/工作/查看器/diagnostics/video10-global-initializer')]:
    shutil.copy2(lp(p),lp(d/'training-interface-audit/saved-snapshot'/p.name))

review=read(S/'unit-review-final.json');rows=[]
for relative in ['src/shuolee_portrait/camera_world_units.py','src/shuolee_portrait/sequence_fitting.py','tests/test_camera_world_units.py']:
    source=P/relative;destination=OUT/'source'/relative
    expected=review['source_hashes'][source.as_posix()]
    assert sha(source)==expected
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,destination)
    rows.append(dict(path=relative,sha256=sha(destination),bytes=destination.stat().st_size))
doc='''# 共同相机单位：CPU部署与验证

这是逐帧接口的单位保护增量，尚未完成真实人物训练。

## 已验证

输入必须声明相机平移单位、共同正比例、依据和证据。相机平移统一到人体模型米制后再进行相机到世界变换；旋转和内参不变。任何额外SfM点也必须使用同一比例。

检查点保存单位来源、帧顺序和冻结相机。缺失或冲突必须拒绝加载，包括嵌套模块和非严格加载。

## 运行

使用已有Python 3.10环境，需NumPy和PyTorch；CPU即可，无需下载SMPL-X、VPoser权重或编译CUDA。

```bash
python -B tests/test_camera_world_units.py --report unit-tests.json
```

期望19项检查全部通过。报告中的合成尺度0.4是已知几何控制，绝不能作为真实视频参数。

## 仍未解决

实际视频共同尺度与物理标定、真实逐帧数据准入、自由衣物高斯姿态运输和完整渲染连接仍待验证。历史拟合尺度不是已验证标定，不自动复用。不能对已经混合单位生成的网格整体乘一个比例冒充修复。

只部署本包模块和测试，不执行旧拟合脚本；旧脚本没有传递新单位声明，且部分旧脚本在构造模块之前会加载模型。未来运行入口必须先检查单位与数据，再加载任何模型或启动GPU。

本接口是本地逐帧适配实现，不是SMPLify-X或GaussianAvatars_SMPLX完整官方复现。
'''
dest=OUT/'source/docs/共同相机单位与部署边界.md';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(doc,encoding='utf-8')
rows.append(dict(path='docs/共同相机单位与部署边界.md',sha256=sha(dest),bytes=dest.stat().st_size))
archive=OUT/'world-units-123-source.zip'
assert not archive.exists(), 'Frozen source archive already exists; do not overwrite.'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for row in rows:z.write(OUT/'source'/row['path'],row['path'])
stamp=datetime.now(timezone.utc).isoformat()
manifest=dict(updated=stamp,base_branch='smplx/frame-pose-contract-123',base_commit='3750cd76682ab29023875fb874ac27e31a8bc590',
    files=rows,archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
    code_scope='CPU camera-unit and checkpoint provenance guards only',training_admitted=False,
    do_not_overwrite=['frame_pose_contract.py','alignment_inputs.py','existing camera files','old fitting runners'],
    expected_unit_tests=19,real_input_scale_known=False,private_evidence_outside_source_archive=True)
save(OUT/'source-manifest.json',manifest)
for name in ['result.json','unit-regression-final.json','unit-review-final.json','saved-input-result.json']:
    shutil.copy2(lp(S/name),OUT/name)
handoff='''# 给00的共同单位增量交接

用户已授权在00负责Git部署、实验与反馈。以已发布3750cd7分支为基础，只增量移植source内四个文件；保留00已修复的图片解码、独立相机校验和Git兼容性。

1. 先核验source-manifest.json及ZIP哈希，检查目标Git状态与任务进程。不要覆盖正在运行的代码或113稳定版。
2. 此包只需Python 3.10、NumPy、PyTorch，在现有CPU环境运行19项检查，并运行该分支原有回归。必要时可以修复确证实现错误，保存前后证据。
3. 不运行旧逐帧拟合脚本、旧相机优化或任何高斯训练。实际共同尺度未知；旧尺度1.4259/1.3215不自动套用。禁止用常数1补齐声明。
4. 本目录result、review和saved-input等是内部含本机路径的证据，只共享给本聊天，不原样推公开仓库。source四个文件才是拟定提交范围。
5. 发布独立CPU接口版本后，反馈Git提交、真实运行环境、19项及全套回归结果、CI、任何失败和修改。服务器没实际执行就保持server_verified=false；任何训练前必须另行完成真实数据和尺度准入。

本对话已收到00的3750cd7部署结果，已合入返回的完整图像解码补丁并通过29项检查。此次是新尺度保护的实质增量，无需重复确认已收到。
'''
(OUT/'部署增量.md').write_text(handoff,encoding='utf-8')
p=C/'local-research-status.json';v=read(p);v['world_unit_handoff']=dict(updated=stamp,path=str(OUT),archive_sha256=manifest['archive_sha256'],state='prepared_for_authorized_00_deployment',training_admitted=False);save(p,v)
print(json.dumps(manifest,ensure_ascii=False))
