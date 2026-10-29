# 服务器模型配置

`bash deploy.sh` 为官方 SMPLify-X 流程准备 7 个模型文件：YOLOv8n、SAM2、RTMW 及其归一化配置、SMPL-X Neutral、VPoser 的配置与权重。已有目录中的这 7 个文件通过大小和 SHA-256 校验后直接复用；缺项时在 `~/smpl-x/models/` 组装，旧目录保持不变。缺少的公开依赖由服务器下载，SMPL-X 和 VPoser 从已有授权目录复用。**模型不放 Git。** 每个视频重新估计人物与相机，不加载旧人物的拟合或高斯。

完整路径、大小、SHA-256和下载来源以 `configs/server-assets.json` 为准。下载中断后重新执行部署命令会续传；只有校验通过的文件才正式启用。RTMW模型与归一化参数从同一个官方ZIP取得，解压校验后删除ZIP。服务器执行 `python run_smplx.py doctor --assets ~/smpl-x/models` 可复查。

新流程不准备 GVHMR、HMR2、ViTPose 及其辅助文件。历史路线的完整 21 项清单和 `legacy` 检查仍保留；这些旧资产不会被删除。终端显示实际缺失文件的下载进度，完成环境和输入校验后才输出配置完成。网络代理沿用环境的 `HTTPS_PROXY`、`HTTP_PROXY` 或 HTTP 类型的 `ALL_PROXY`。

主要资源与上游入口：

| 资源 | 配置位置 | 来源 |
|---|---|---|
| SAM2.1 tiny | `sam2/sam2.1_hiera_tiny.pt` | [SAM2 官方项目](https://github.com/facebookresearch/sam2) |
| YOLOv8n | `detection/yolov8n.pt` | [Ultralytics 官方发布](https://github.com/ultralytics/assets/releases/tag/v8.2.0) |
| RTMW-L 384×288 与预处理配置 | `rtmw/rtmw-l-384x288.onnx`、`rtmw/normalization.json` | [RTMLib](https://github.com/Tau-J/rtmlib)，须使用相同模型包的归一化参数 |
| SMPL-X Neutral | `smplx/SMPLX_NEUTRAL.npz` | 用户已取得的官方授权资源 |
| VPoser V02_05 | `vposer/V02_05/` | [human_body_prior 官方项目](https://github.com/nghorbani/human_body_prior) |

SAM2、YOLO 和 RTMW 使用清单中锁定的官方来源。模型沿用各自的上游许可。旧 `legacy` 路线的 GVHMR、HMR2 和 ViTPose 镜像来源仍记录在完整清单中。

官方拟合源码通过 Git 获取到运行目录的 `sources/smplify-x`，固定提交为 `68f8536707f43f4736cdd75a19b18ede886a4d53`；实际使用的 7 个源码文件也须通过既有 SHA-256 核对。`SMPLX_OFFICIAL_SOURCE_ROOT` 可复用已有合法源码目录。校验后的路径与版本写入部署配置，供视频训练默认读取，不依赖开发机路径。此源码准备不安装另一套拟合器或修改求解参数。

旧113服务器已有的 `smplx/SMPLX_NEUTRAL.npz` 和 `vposer/V02_05/` 自动识别，也可用 `SMPLX_ASSETS` 指定其他已授权目录。全新服务器须先按官方授权预置或挂载这三份许可文件；缺失或校验错误时会在下载公开权重和启动训练前停止。脚本不会自动登录或取得许可模型。日常只上传视频，不上传本机代码、环境、模型分卷或历史结果。
