# 119部署与运行

根目录README给出唯一部署路线：从Git稀疏检出重建代码，运行安装脚本，然后输入一个视频路径。目标环境不部署查看器，代码和环境不通过本机文件上传。

## 一次配置

```bash
bash scripts/setup_server.sh --assets /data/shuolee-models --runs /data/shuolee-runs
```

模型按[模型配置](模型配置.md)预置。安装完成会实际检查CUDA前向/反向和21个资产哈希，配置写入隔离环境的 `install/deployment119.json`。已有环境用 `--verify-existing`，中断的依赖安装用 `--resume-install`。自定义 `SHUOLEE_ENV_DIR` 时安装和运行使用同一设置。

## 一次运行

```bash
bash scripts/train119.sh /data/videos/person.mp4
```

程序自动选择配置的环境和模型，创建全新目录，完成全部阶段。没有人工审核模式、审批JSON或中途等待。运行开始打印本轮路径；追加 `--output /data/shuolee-runs/person_001` 可指定尚不存在的目录。

每5帧采样，最长边1600，12–300帧；人体拟合1200步；20万初始点；18000训练步；种子6060。高斯使用自由世界坐标，人体只提供本轮初始化范围和尺度。每轮从原视频重新估计相机、人物参数和优化器，不读旧人物结果。

输入完整性、拟合误差、轮廓体积、数值有限性、点数预算和原生/PLY渲染一致性自动检查。异常直接记录具体阶段和错误，不自动放宽阈值或覆盖旧运行。

需要关闭SSH/WinSCP终端后继续时：

```bash
mkdir -p ~/shuolee/logs
nohup bash scripts/train119.sh /data/videos/person.mp4 --output /data/shuolee-runs/person_001 > ~/shuolee/logs/person_001.log 2>&1 < /dev/null &
```

查询记录的阶段与错误：

```bash
~/.local/share/shuolee-portrait/env-cu128/bin/python -s run119.py status --run /data/shuolee-runs/person_001
tail -n 80 ~/shuolee/logs/person_001.log
```

自定义环境时对应替换Python路径。`status` 读取持久状态，不根据历史PID猜测进程仍存活。

## 输出

- `export/point_cloud.ply`：Graphdeco标准3DGS，完整SH、log-scale、logit-opacity和wxyz旋转。
- `export/cameras.json`：通用3DGS相机数组，保留完整fx/fy/cx/cy和相机到世界旋转。
- `export/colmap/`：COLMAP PINHOLE文本相机；未构造稀疏点和图像观测，相关文件/行为空。
- `export/validation.json`、`camera_provenance.json`：导出一致性和坐标来源。

没有Spark专用输出、查看器部署或人工审核页面。

## 适用范围与本地验证

默认相机前端用新图像SIFT轨迹估计单一焦距，假设居中主点和fx=fy，执行Essential/PnP及固定K束调整。已有可信输出视频K可在配置中声明 `fixed_pinhole`。不猜测未知鱼眼裁剪或重复去畸变。

保留119原有轮廓计票方式及其出画面区域计负证据的限制。遮挡、微动、低纹理和相机误差可能使流程停止或降低质量。结果的完成状态与质量验证字段分开，不附加人工审批步骤。

本地实际结果记录于 `provenance/deployment119-validation.json`。CPU编排测试、GPU算子检查和原视频全流程分别记录；环境复用也如实说明。00/113部署状态单独记录。
