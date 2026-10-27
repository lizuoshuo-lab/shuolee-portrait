# 119 衣物表面自由高斯流程

本入口复用经过审阅的119表示和18,000步训练配方：新视频生成相机、人物遮罩和二维观测，重新拟合人体，再从衣物轮廓体积初始化20万个自由世界坐标高斯。人体仅提供本轮初始范围与尺度；高斯不绑定人体三角形，也不向裸模吸附。

代码由Git部署。业务输入只上传原视频；不上传旧118缓存、人物拟合、旧高斯或优化器。公开模型、许可人体模板和Python/CUDA是服务器环境依赖，放在代码目录外的资产目录，按资产清单校验。缺失依赖会报错，不生成替代遮罩或伪造成功。

## 目录和版本

下面示例约定：

```text
~/shuolee/portrait/           Git代码，固定到本地验收过的版本
~/shuolee/videos/person.mp4  本轮唯一上传的视频
~/shuolee/assets/            独立环境资产，参见模型配置说明
~/shuolee/runs/119_001/      全新运行目录，必须尚不存在
~/shuolee/logs/119_001.log   后台命令输出
```

从项目现有私有远端获取已验证版本；记录 `git rev-parse HEAD`，不把“拉取成功”当作本轮训练或公司00验收。已有失败目录保留，下一轮使用新名称。模型与Python环境不进入WPS交付目录。

先在本地完成安装和运行核验，随后才装配公司端。模型来源和许可见[模型配置](模型配置.md)。本页的新入口是 `run119.py`；`run.py reconstruct` 是历史的绑定式路线，不用于119自由点训练。

## 先做到拟合检查

在已激活且通过环境核验的Python环境中执行：

```bash
cd ~/shuolee/portrait
python run119.py reconstruct --video ~/shuolee/videos/person.mp4 --assets ~/shuolee/assets --config configs/video119.json --output ~/shuolee/runs/119_001
```

默认停在 `fit-gate`，生成真实同相机人体叠图，等待实际审阅。每5帧取一帧，最长边1600，12–300帧；重新拟合1200步，所有训练状态从头开始。不会自动删帧、插值补相机、覆盖或续接旧任务。

WinSCP命令窗口没有交互TTY；长阶段需要后台运行。例如已激活环境后：

```bash
mkdir -p ~/shuolee/logs
cd ~/shuolee/portrait
nohup python -u run119.py reconstruct --video ~/shuolee/videos/person.mp4 --assets ~/shuolee/assets --config configs/video119.json --output ~/shuolee/runs/119_001 > ~/shuolee/logs/119_001.log 2>&1 < /dev/null &
```

如果环境依赖 `cuda128.env`，先在同一次命令中 `source ~/shuolee/cuda128.env`；使用实际安装环境的Python。运行后只查状态，不重复启动：

```bash
python ~/shuolee/portrait/run119.py status --run ~/shuolee/runs/119_001
tail -n 80 ~/shuolee/logs/119_001.log
```

`status` 读取运行、准备和进度文件，包括准备失败的原因；它不根据历史PID猜测进程仍在运行。输入一致性可另外执行 `inspect --run ...`，该检查不加载GPU模型。

## 审阅后进入下一阶段

先打开 `runs/119_001/reviews/fit/` 的原图与网格叠图，检查人物位置、肩袖、裙摆、腿脚和头部。实际看过后记录判断；`--note` 写观察到的事实。记录只约束后续有限研究，不宣布标定或重建质量通过。

```bash
python run119.py review fit --run ~/shuolee/runs/119_001 --decision approve --note '已检查同相机叠图，人体范围可用于衣物初始化'
python run119.py stage surface --run ~/shuolee/runs/119_001 --assets ~/shuolee/assets
```

再检查 `clothing-init/initialization.ply`、对应原图和初始化渲染，确认衣物和脚部范围、点分布及截断情况。拒绝时把 `approve` 换为 `reject`，后续阶段会停止。

```bash
python run119.py review surface --run ~/shuolee/runs/119_001 --decision approve --note '已检查初始化点和渲染，衣物范围没有明显截断'
nohup python -u run119.py stage train --run ~/shuolee/runs/119_001 --assets ~/shuolee/assets > ~/shuolee/logs/119_001-train.log 2>&1 < /dev/null &
```

训练在1000步等待实际审阅，最多30分钟；同一进程保留优化器等待决定，超时或拒绝就停止并保留快照。查看 `gaussian-run/step-001000/` 的原图、渲染、`world.ply` 和中心分布；只有检查后才决定是否继续这一轮剩余预算：

```bash
python run119.py review checkpoint --run ~/shuolee/runs/119_001 --decision approve --note '已检查1000步同相机结果及中心分布，继续同轮预算'
```

审阅命令从实际文件计算SHA，不需要手写补丁或JSON。旧审阅与不匹配的产物不能解锁新阶段；审阅不会越过失败的数值门槛。停止后的运行不能用此命令跨轮续训。

确认 `status` 中实际完成步数为18,000、对应回读校验通过后导出：

```bash
python run119.py stage export --run ~/shuolee/runs/119_001 --assets ~/shuolee/assets
```

默认导出通用数据，不夹带Spark专用网页，也不依赖人体模板再次蒙皮：

- `export/point_cloud.ply`：常用3DGS三阶球谐字段、对数尺度、wxyz四元数、logit不透明度，包含世界坐标和完整协方差。
- `export/cameras.json`：常用3DGS相机数组，包含相机中心、相机到世界旋转和完整fx/fy/cx/cy；部分查看器会忽略额外cx/cy，须单独核对。
- `export/colmap/cameras.txt`、`images.txt`、`points3D.txt`：COLMAP PINHOLE文本相机，保留完整主点及Hamilton wxyz世界到相机外参。没有虚构稀疏点或二维轨迹，点文件为空，图像观测行也为空；不附原始图像。
- `export/camera_provenance.json`：可选来源记录，保存原完整K/w2c、共同世界变换、文件哈希及原始帧名到导出文件名的映射。重名、空格及不安全字符统一清洗并去重，不静默丢失原名。
- `export/validation.json`：原生/PLY一致性检查。

通用查看器可直接导入PLY；需要相机时使用它支持的3DGS数组或COLMAP文件。文件可读与几何精度是独立验收事项。历史绑定式路线仍保留原 `cameras.json` 来源记录合同。

## 相机和验收范围

默认相机前端从本轮新图像特征估计一个有限焦距候选，假定居中主点和相同fx/fy，再执行Essential/PnP及固定K的束调整。整条外观轨迹在几何估计前留出。它与历史119的厂家K裁剪假设不同，复用的是119衣物表示和训练配方。

已有可信输出视频K时，在 `video119.intrinsics` 使用 `fixed_pinhole`，必须声明与视频完全相同的 `image_wh`、K、像素中心约定和来源。不根据视频宽高猜厂家传感器裁剪，不对已去畸变视频再次套用未知畸变。相机候选能计算、留出残差或文件校验通过，都不等于物理标定通过。

历史119的体积计票方式在本版本保留：出画面区域仍可能被计作负轮廓证据。这个已知视场限制写入初始化审计，尚未静默替换为“修正后119”。体积是轮廓外包络，凹陷、遮挡和人物微动仍会影响结果。

已完成CPU输入/身份/像素约定和合成PLY变换回读。本次历史119沙发PLY也通过了产品查看器及两个独立第三方查看器的实际导入、视角操作和切换检查；这证明历史文件互操作。真实新视频成功SfM、GPU分割/先验/拟合/18,000步，以及新流程训练产物的第三方回读与公司00运行仍需分别核验。原生图像独立可信留出面部点逐点小于1像素尚未通过，界面和所有结果保持 `quality_accepted=false`、`formal_training_admitted=false`。

来源及行为差异见 `src/shuolee_portrait/route119-provenance.json` 和 `configs/video119-provenance.json`。失败时保存运行状态、`preparation.json`、对应日志及阶段产物；先根据原始异常定位，不在公司端反复加补丁重启。
