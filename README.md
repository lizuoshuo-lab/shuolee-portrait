# Shuolee

视频重建与通用三维查看器。以“119 衣物表面自由点 · 18000步”实际流程为工程基线：静态共享世界高斯，20万初始点，1200步身体拟合，18000步高斯训练。

代码、配置、依赖锁和安装脚本通过 Git 部署。**本地只上传原视频**；目标机生成帧、相机、掩码、拟合和高斯，不复制旧实验输出作为新任务输入。模型权重按上游授权配置在 Git 外。

| 目录 | 内容 |
|---|---|
| `apps/viewer` | 独立查看器：本机文件导入，标准 PLY / COLMAP 导出 |
| `src/shuolee_portrait` | 视频与119重建流程 |
| `src/avatar_core` | 高斯和人体核心适配 |
| `configs`、`scripts` | 固定配置、安装与检查 |
| `third_party`、`upstream.lock.json` | 固定来源及保留的上游许可 |
| `tests`、`provenance`、`docs` | 测试、来源与验收记录 |

视频、权重、运行结果、虚拟环境和构建产物均不进 Git。

## 先运行查看器

Node.js 22 / 24 LTS：

```bash
git clone --branch codex/engineering-119-viewer https://github.com/lizuoshuo-lab/shuolee-portrait.git
cd shuolee-portrait/apps/viewer
npm ci
npm run check
npm start
```

打开 http://127.0.0.1:4870，选择模型，可同时选择配套相机 JSON。导出文件可由其他通用查看器直接读取。
使用说明与格式边界见 [查看器](apps/viewer/README.md)。

## 重建部署

Linux x86_64、Python 3.10、NVIDIA 驱动、CUDA 12.8 开发工具和 FFmpeg：

```bash
cd shuolee-portrait
bash scripts/setup_server.sh --preflight-only
bash scripts/setup_server.sh
source "$HOME/.local/share/shuolee-portrait/env-cu128/bin/activate"
unset PYTHONPATH SHUOLEE_EXTRA_PYTHONPATH
python run.py doctor --assets /data/shuolee-models
```

安装在目标机完成，CUDA扩展在目标机编译，不上传本机环境或补丁文件。
官方及已授权模型的位置与校验见 [模型配置](docs/模型配置.md)。
119入口及实际验证状态见 [119流程](docs/119-workflow.md)。

119是流程基线，历史人物的重影、背面缺损与物理标定尚未验收。当前工程改动也不代表00或113服务器已部署。
旧0.1候选说明保留在 [历史文档](docs/history/README-0.1.0-rc1.md)，其中旧诊断路线和手工传输说明不作为本版本部署步骤。
