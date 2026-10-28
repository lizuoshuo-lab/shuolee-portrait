# smpl-x

一次部署，以后只上传视频。自动完成新相机和人物观测、1200步人体拟合、20万衣物表面自由高斯初始化、18000步训练和通用PLY导出。

在目标Linux服务器执行这一行：

```bash
git clone --filter=blob:none --no-checkout -b main https://github.com/lizuoshuo-lab/shuolee-portrait.git ~/smpl-x/app && git -C ~/smpl-x/app sparse-checkout init --cone && git -C ~/smpl-x/app checkout && bash ~/smpl-x/app/deploy.sh
```

随后把视频上传到 **`~/smpl-x/videos/`**。文件稳定30秒后自动排队，一次训练一个，关闭SSH或WinSCP后继续运行。相同内容的视频只处理一次。

有systemd时安装开机自启服务；没有systemd时自动后台运行，无需sudo。后一种环境在机器或容器重启后，执行 `bash ~/smpl-x/app/deploy.sh` 恢复接收视频；完成的任务不会重复训练。

当前采集相机的横屏、竖屏视频使用同一个文件夹和训练命令，按实际宽高自动换算已有标定的缩放与居中裁剪参数。

也可以在服务器手动训练指定视频：

```bash
bash ~/smpl-x/app/scripts/train.sh /path/to/person.mp4
```

结果在 **`~/smpl-x/runs/<视频名>-<内容标识>/export/point_cloud.ply`**，同时提供通用3DGS相机JSON和COLMAP相机。状态在 `~/smpl-x/queue/status.json`，每个视频的日志在 `~/smpl-x/queue/logs/`。

本地已用00数据集的横屏、竖屏各一段，从视频文件夹自动完成18000步训练与通用导出；分别为89帧和112帧。独立PLY/COLMAP读取检查通过，服务重启不会重复训练。

仓库公开，下载代码无需GitHub账号或凭据。目标环境需要Linux x86_64、NVIDIA驱动、Python 3.10、CUDA 12.8开发工具、FFmpeg及C++构建工具。脚本安装隔离依赖环境；已有环境自动核验复用。已有113服务器的模型和 `cuda128.env` 自动识别；缺少的公开模型自动下载到 `~/smpl-x/models/`，逐个校验后启动接收视频。旧模型目录保持不变。全新服务器需要预置已授权的SMPL-X与VPoser，详见[模型配置](docs/models.md)。仅安装systemd服务时需要sudo权限。

代码仅经Git部署，业务输入只有视频；没有查看器、Node.js、Spark专用格式或人工审核步骤。[流程与维护说明](docs/workflow.md) · [本地验证记录](provenance/deployment-validation.json)
