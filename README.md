# Shuolee 119 视频重建

标准流程：视频 → 新相机、人物遮罩和二维观测 → 1200步人体拟合 → 20万衣物表面自由高斯 → 18000步训练 → 通用PLY和相机。

代码通过Git部署，业务输入只有原视频。部署内容仅包含重建，不安装查看器、Node.js或网页服务。全流程自动执行，没有人工审核、确认文件或逐阶段操作。

## 首次部署

目标环境：Linux x86_64 / WSL2、Python 3.10、NVIDIA GPU及驱动、CUDA 12.8开发工具、FFmpeg和C++构建工具。通用预训练模型按[模型配置](docs/模型配置.md)在目标环境一次性配置。

```bash
git clone --filter=blob:none --sparse --branch codex/reconstruction-119 https://github.com/lizuoshuo-lab/shuolee-portrait.git
cd shuolee-portrait
git sparse-checkout set configs scripts src third_party docs tests provenance .github
bash scripts/setup_server.sh --assets /data/shuolee-models
```

安装脚本创建独立Python环境，编译并实际检查CUDA算子，校验模型，保存部署配置。已有依赖环境使用 `--verify-existing --assets ...` 核验和绑定。自定义环境位置时设置 `SHUOLEE_ENV_DIR`。

## 导入视频并重建

```bash
bash scripts/train119.sh /data/videos/person.mp4
```

默认在 `~/shuolee/runs/` 生成独立运行目录并连续完成重建；可用 `--output /data/runs/person_001` 指定全新目录。代码更新使用 `git pull --ff-only`。

输出 `export/point_cloud.ply`、通用 `cameras.json`、COLMAP相机和格式核验记录。必要的输入、拟合、有限数值及导出一致性检查自动执行；异常直接停止并保留日志，不覆盖已有结果。详细运行和验证说明见[119流程](docs/119-workflow.md)。

模型、视频、环境、运行结果均保存在Git外。适用输入是围绕静止人物拍摄的针孔图像域视频；相机不足、遮挡或人物微动可能导致失败或质量下降。训练完成不自动等同于物理几何精度通过。
