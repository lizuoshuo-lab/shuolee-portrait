# smpl-x

一次部署，以后只上传视频。自动完成新相机和人物观测、1200步人体拟合、20万衣物表面自由高斯初始化、18000步训练和通用PLY导出。

在目标Linux服务器执行这一行：

```bash
git clone --filter=blob:none --sparse -b codex/smpl-x https://github.com/lizuoshuo-lab/shuolee-portrait.git ~/smpl-x/app && bash ~/smpl-x/app/deploy.sh
```

随后把视频上传到 **`~/smpl-x/videos/`**。文件稳定30秒后自动排队，一次训练一个，关闭SSH或WinSCP后继续运行，服务器重启后服务自动启动。相同内容的视频只处理一次。

也可以在服务器手动训练指定视频：

```bash
bash ~/smpl-x/app/scripts/train.sh /path/to/person.mp4
```

结果在 **`~/smpl-x/runs/<视频名>-<内容标识>/export/point_cloud.ply`**，同时提供通用3DGS相机JSON和COLMAP相机。状态在 `~/smpl-x/queue/status.json`，每个视频的日志在 `~/smpl-x/queue/logs/`。

目标环境需要Linux x86_64/systemd、NVIDIA驱动、Python 3.10、CUDA 12.8开发工具、FFmpeg及C++构建工具。脚本安装隔离依赖环境；已有环境自动核验复用。已有113服务器的 `~/shuolee/models/113/` 和 `cuda128.env` 自动识别，通用预训练模型须满足[模型清单](docs/模型配置.md)。私有Git仓库使用服务器已有Git凭据；安装系统服务需要sudo权限。

代码仅经Git部署，业务输入只有视频；没有查看器、Node.js、Spark专用格式或人工审核步骤。[流程与维护说明](docs/workflow.md) · [本地验证记录](provenance/deployment-validation.json)
