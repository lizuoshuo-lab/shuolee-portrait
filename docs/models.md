# 服务器模型配置

`bash deploy.sh` 自动准备模型。已有完整目录经21个文件的大小和SHA-256校验后直接复用；旧113目录只有部分文件时，在 `~/smpl-x/models/` 组装完整模型，旧目录保持不变。缺少的公开依赖由服务器下载，SMPL-X和VPoser从已有授权目录复用。**模型不放Git。** 每个视频重新估计人物与相机，不加载旧人物的拟合或高斯。

完整路径、大小、SHA-256和下载来源以 `configs/server-assets.json` 为准。下载中断后重新执行部署命令会续传；只有校验通过的文件才正式启用。RTMW模型与归一化参数从同一个官方ZIP取得，解压校验后删除ZIP。服务器执行 `python run_smplx.py doctor --assets ~/smpl-x/models` 可复查。

旧113目录首次补齐约下载5.8GB；终端会显示下载进度，完成校验后才输出 `smpl-x READY`。再次部署会复用已校验的文件。网络代理沿用环境的 `HTTPS_PROXY`、`HTTP_PROXY` 或HTTP类型的 `ALL_PROXY`。

主要资源与上游入口：

| 资源 | 配置位置 | 来源 |
|---|---|---|
| HMR2 特征网络 | `gvhmr/hmr2/epoch=10-step=25000.ckpt` | [GVHMR 官方安装说明](https://github.com/zju3dv/GVHMR/blob/main/docs/INSTALL.md)，其中预训练模型目录 |
| ViTPose | `gvhmr/vitpose/vitpose-h-multi-coco.pth` | 同上 |
| GVHMR | `gvhmr/gvhmr/gvhmr_siga24_release.ckpt` | 同上 |
| SAM2.1 tiny | `sam2/sam2.1_hiera_tiny.pt` | [SAM2 官方项目](https://github.com/facebookresearch/sam2) |
| YOLOv8n | `detection/yolov8n.pt` | [Ultralytics 官方发布](https://github.com/ultralytics/assets/releases/tag/v8.2.0) |
| RTMW-L 384×288 与预处理配置 | `rtmw/rtmw-l-384x288.onnx`、`rtmw/normalization.json` | [RTMLib](https://github.com/Tau-J/rtmlib)，须使用相同模型包的归一化参数 |
| SMPL-X Neutral | `smplx/SMPLX_NEUTRAL.npz` | 用户已取得的官方授权资源；自动复用到GVHMR需要的位置 |
| VPoser V02_05 | `vposer/V02_05/` | [human_body_prior 官方项目](https://github.com/nghorbani/human_body_prior) |
| GVHMR 通用关节映射、均值参数 | `gvhmr/hmr4d/...` | 与锁定 GVHMR 源码版本对应的官方通用资源 |

GVHMR、HMR2和ViTPose的自动下载使用固定版本的[公开权重镜像](https://huggingface.co/camenduru/GVHMR/tree/21b32d5389e2e59c0737d4c4095bbc0b8c23f66b)，校验值与本地已验证模型一致；这不是作者独立发布的校验值。SAM2、YOLO、RTMW及GVHMR辅助资源使用官方来源。模型沿用各自的上游许可。

旧113服务器已有的 `smplx/SMPLX_NEUTRAL.npz` 和 `vposer/V02_05/` 自动识别，也可用 `SMPLX_ASSETS` 指定其他已授权目录。全新服务器须先按官方授权预置或挂载这三份许可文件；缺失或校验错误时会在下载公开权重和启动训练前停止。脚本不会自动登录或取得许可模型。日常只上传视频，不上传本机代码、环境、模型分卷或历史结果。
