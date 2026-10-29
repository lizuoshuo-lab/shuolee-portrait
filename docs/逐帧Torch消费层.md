# 逐帧自由衣物高斯 Torch 接口

`FreeClothingTorch` 在准备阶段冻结三角形运动表和相机，逐帧将参考姿态中的自由 XYZ、wxyz 四元数和 log 标准差转换为当前姿态。接口保留参数梯度，没有优化器、初始化、增密或剪枝。

## Git 安装与 CPU 验证

使用已有 Python 3.10 环境，验证版本为 NumPy 1.26.4、Torch 2.7.1。以下 Linux Bash 命令拉取独立分支，在新目录运行测试。环境默认取 `~/.local/share/smpl-x/env-cu128`，兼容已有的 `shuolee-portrait/env-cu128`；其他位置可先设置 `SMPLX_ENV_DIR`。

```bash
bash -c 'set -eu; test ! -e "$HOME/smpl-x-torch-clothing"; git clone --depth 1 --filter=blob:none --no-checkout --branch smplx/torch-clothing-consumer-123 https://github.com/lizuoshuo-lab/shuolee-portrait.git "$HOME/smpl-x-torch-clothing"; cd "$HOME/smpl-x-torch-clothing"; git sparse-checkout init --cone; git checkout; git sparse-checkout set src scripts tests docs; CHECK_PY="${SMPLX_ENV_DIR:-$HOME/.local/share/smpl-x/env-cu128}/bin/python"; if [ ! -x "$CHECK_PY" ] && [ -z "${SMPLX_ENV_DIR:-}" ]; then CHECK_PY="$HOME/.local/share/shuolee-portrait/env-cu128/bin/python"; fi; test -x "$CHECK_PY"; export CUDA_VISIBLE_DEVICES=""; mkdir -p test-results; git rev-parse HEAD; "$CHECK_PY" -s -B tests/test_free_clothing_torch.py --report test-results/torch-clothing-consumer-cpu.json'
```

报告中的 `outcome: passed`、`tests_run: 19`、`failures: 0` 和 `errors: 0` 表示本次 CPU 控制完成。测试报告保存在 `test-results/torch-clothing-consumer-cpu.json`，拒绝覆盖已有报告。已有检出目录时命令停止，避免覆盖工作。无需上传源代码、人物资料或下载模型权重。

测试依赖完整基线仓库中的运输模块、单位换算和 SH 工具。在检出目录使用同一环境运行完整 CPU 回归：

```bash
CUDA_VISIBLE_DEVICES='' python -s -B -m unittest discover -s tests -p 'test_*.py'
```

CI 使用 `requirements-test.txt` 固定的 CPU 依赖。

## 准备与调用

```python
consumer = FreeClothingTorch.from_cpu(
    source,
    camera_length_contract=contract,
    frame_source_w2c=cameras,
    device="cpu",
    dtype=torch.float32,
)
out = consumer(frame_id, xyz, quats, log_scales, gaussian_ids=ids)
```

`source` 是已验证的 `FreeClothingTransport`。相机映射必须完整覆盖所有帧身份，每帧严格对应一个 `4×4` 外参矩阵。几何已经处于声明的统一米制坐标；准备阶段只将源相机平移按显式共同换算比例转换一次，不再次缩放几何。比例与来源依据必须提供，声明本身不代表物理标定通过。

输入张量须与冻结表的设备、float32/float64 精度和 Gaussian ID 顺序完全一致。接口支持明确选择 CPU 或 CUDA；上述部署命令仅验证 CPU。高斯中心可保留离人体表面的自由偏移，支撑三角只用于确定运动。

返回 `centers`、`quats`、`scales`、`log_scales` 和 `reference_view_directions`。旋转与尺度同步作用于完整协方差；SH 必须用返回的参考方向求值，再将 RGB 交给渲染器。参考 SH 不能直接当作当前世界坐标下的 SH 使用。

默认不生成完整协方差，以适配四元数和尺度输入。设置 `return_covariance=True` 可输出完整协方差，并额外进行 Cholesky 正定检查。一般剪切只作局部相似近似，支撑三角残差始终保留。

## 运行约定

逐帧数值运算留在表所在设备，不调用 NumPy、CPU 拷贝、文件读取或特征分解。局部关闭 autocast，保持声明精度与自动微分。输入验证使用 Torch 异步断言；CUDA 设备断言失败后，该次运行必须结束。

表的 Tensor identity/version 和梯度状态在调用时检查，检测普通原位修改、替换或重新启用梯度。即使在外层 `torch.inference_mode()` 中准备，也会生成可跟踪版本的常量表，并恢复调用者状态。绕过 Torch 的 `.data` 或底层 storage 写入不受此约定支持。输出可独立使用，不能通过修改输出重新定义冻结支撑。

四元数先按最大绝对分量缩放再归一化，避免直接求范数时的平方上溢或下溢，并保留梯度。零值、非有限输入及倒幅值已无法有限表示的极端输入会明确报错，不加 epsilon 改写旋转。参考帧中心和 log 标准差保持逐位恒等；四元数按稳定算法归一化。

点数和 Gaussian ID/三角支撑映射固定。改变布局必须由调用方同步参数、优化器状态、身份和支撑，不能静默重绑到最近网格。接口的 `report()` 返回准备来源、单位换算和限制；它不保存训练状态，也不核验原视频文件。

## 验证范围

19 项 CPU 控制涵盖参考恒等、半周旋转、剪切残差、共同尺度、身份、完整协方差、SH 方向、独立梯度、表修改与梯度状态保护、无 CPU 转换的逐帧路径、精度与推理上下文，以及极小和极大四元数幅值。

本模块尚未接入原生训练器，未验证 20 万点性能或实拍重建。合成张量一致性不能替代真实相机、材料对应和人像质量验收；`training_admitted` 与物理相机验收字段保持 false。最终参考帧高斯也不能被视为所有动态观测的同一个静态姿态。
