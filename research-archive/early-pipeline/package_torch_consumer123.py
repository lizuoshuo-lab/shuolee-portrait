from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,zipfile,shutil
H=Path(__file__).parent;C=H.parent/'shuolee-coordination'
P=Path('D:/GS-SLAM-Data/shuolee-portrait-new-capture-118')
out=C/'handoffs/torch-clothing-consumer-123-20261029'
out.mkdir(exist_ok=False)
doc='''# 逐帧自由衣物高斯的 Torch 消费层

本增量依赖 `smplx/free-clothing-transport-123` 的 d4a3d73fdf8dbec8aed4f661eccd3bbf95f9210b。它是通用工程接口，不是已完成的人物训练项目，也不是完整官方 GaussianAvatars_SMPLX 复现。

输入是共享参考姿态中的自由 XYZ、wxyz 四元数、log 标准差，与固定的 Gaussian ID/运动支撑三角形 ID。网格仅给运动支撑，不把点吸附到裸人体。每帧局部相似运动同时作用于中心、完整协方差及 SH 参考视线。一般剪切使用相似近似，支撑残差保留；不能据此断言真实形变已正确。

`FreeClothingTorch.from_cpu(source, camera_length_contract=contract, frame_source_w2c=cameras, device="cuda", dtype=torch.float32)` 只在准备时拷贝一次三角形运动表和相机。几何已经按声明统一，原始相机平移按显式比例转换一次；比例和依据必须给出，不隐式猜为物理标定。

每步调用 `consumer(frame_id, xyz, quats, log_scales, gaussian_ids=ids)` 得到 centers/quats/scales 和 reference_view_directions。SH 用参考方向求值，再把 RGB 交给渲染器；不得把参考 SH 当作 posed-world SH。原生路径默认不生成 covariance；审计时可指定 return_covariance=True。固定表使用 identity/version 守护，Torch 数学保持原设备/精度，关闭局部 autocast，不在热路径做 CPU/NumPy 转换或特征分解。

这是固定点数消费层，没有优化器、相机优化、增密/剪枝或初始化。未来改变 Gaussian 布局必须同步参数、Adam 状态、ID和支撑，不能偷偷按最近网格刷新。最终交付指定参考帧的高斯，不能把它当作所有动态观测的同一个静态姿态。

## 可直接执行的 CPU 复验

在完整仓库和已有 Python/Torch 环境中运行：

```bash
python tests/test_free_clothing_torch.py --report /tmp/torch-clothing-consumer-check.json
```

看到 `outcome: passed` 与 `tests_run: 14` 才算这批 CPU 控制完成；报告不覆盖已有文件，重复验证需另取报告路径。保存该 JSON 与实际 Git 提交。此命令不开始训练，不使用 CUDA。

## 已有证据与限制

- 14 项 CPU 控制通过：参考姿态、半周旋转、剪切残差、两尺度、身份、精度、表修改保护及独立梯度。
- 研究端另以冻结合成场景完成 11 次 CUDA 前向和 1 次反向，核对 CPU/CUDA数值、autocast和已知尺度；未在本测试命令复跑。
- 上述 CUDA 测试未调用原生光栅器，不能替代原生训练链、20 万点性能、真实相机或实拍人像质量验收。
- 更早的原生渲染14项合成控制使用 CPU 参考运输，并非本新消费层已经接入原生训练器。
- 本增量不包含人物图像、拟合档案、检查点、权重或本机绝对路径。
'''
stage=out/'source';(stage/'docs').mkdir(parents=True)
(stage/'docs/逐帧Torch消费层.md').write_text(doc,encoding='utf-8',newline='\n')
names=['src/shuolee_portrait/free_clothing_torch.py','tests/test_free_clothing_torch.py']
for name in names:
    (stage/name).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(P/name,stage/name)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(stage/names[0])=='c509cb70eeda9833f829dd4c5b74ed34b59686179fb73f0e50f4c1a7adfe630a'
assert sha(stage/names[1])=='900d307fd32ac1d889557113315407f52673cbe87629c91dfae97a28437bb112'
names+=['docs/逐帧Torch消费层.md']
bundle=out/'torch-clothing-consumer-123-source.zip'
with zipfile.ZipFile(bundle,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for name in names:z.write(stage/name,name)
manifest=dict(updated=datetime.now(timezone.utc).isoformat(),source_package_sha256=sha(bundle),bytes=bundle.stat().st_size,
    base_commit='d4a3d73fdf8dbec8aed4f661eccd3bbf95f9210b',files={name:sha(stage/name) for name in names},
    allowed_work='Independent generic-source Git deployment,14 targeted CPU checks and complete CPU regression, fresh-checkout verification. Return receipt and evidenced fixes.',
    GPU_or_training_work_requested=False,person_data_included=False,weights_included=False,production_trainer_included=False,
    research_scope='Newconsumer CPU14 and bounded CUDA tensor parity; not full native-training bridge or physical quality.')
(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(package=str(bundle),sha256=sha(bundle),bytes=bundle.stat().st_size),ensure_ascii=False))
