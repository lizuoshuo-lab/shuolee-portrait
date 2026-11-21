# VHAP 场景融合

## 从原视频一次运行全帧 VHAP 与 MSv2

服务器使用一个新目录接收本次运行的全部文件。

```bash
cd ~/projects/vhap-scene-fusion
python3 -u run_fresh_vhap_scene.py /path/to/portrait.mp4 \
  --output "runs/portrait-fresh-$(date -u +%Y%m%dT%H%M%SZ)"
```

入口要求空运行目录。先在原视频均匀采样的 80 帧上选择正面观测、拟合共享头型，再从同一视频按 3 FPS 重新登记全序列相机。RTMW 肩点参与裁片定位，临时 SMPL-X 提供头颈解剖先验。交付的头像使用 FLAME。共享网格、每帧位姿、颜色、场景种子和高斯均由本次运行生成。

复现 Demo2 历史观测视角时，原命令追加 `--split evaluation/demo2-observed.json`。该文件仅记录视频身份、帧序、评估分组及导出机位。新视频默认按观测角度选取机位。各阶段命令和输出摘要在 `run.json`，逐阶段日志在 `logs/`。最终 `delivery.grids` 指向 VHAP 原生逐帧 grid，`point-cloud-and-cameras.zip` 包含点云与相机。留出分组适用于各报告指定的阶段；所有帧参与过 VHAP 联合跟踪。

## 早期正面选帧流程

```bash
cd ~/projects/vhap-scene-fusion
python3 run_video.py /path/to/portrait.mp4
```

准备阶段从视频登记抽帧和真实时间戳，提取人物掩膜，建立静态场景的 COLMAP 相机。后续阶段使用该相机对应的去畸变原图，场景训练时遮掉头部，保留身体和房间。此入口适用于同一人物在固定位置、身体相对稳定且有环绕观察的视频。

头部候选在去畸变全图上用已标定相机和 STAR 关键点估计偏航。仅选 `abs(yaw) <= 30°`、两眼有足够掩膜支持的画面，按清晰度、脸部大小、重投影质量和角度/时间覆盖排序。默认目标为 16 张训练、4 张留出。留出代表帧的 ±0.6 秒不作为高斯训练的 RGB 目标。候选不足时查看 `selection/selection.json` 的逐帧原因和缺口。

VHAP 为选中的头部拟合 FLAME。训练帧用原图及掩膜，留出帧仅提供面部关键点做姿态。头部高斯直接从 FLAME 三角形和 VHAP 拟合的纹理初始化，随后与含身体的场景高斯在同一台 COLMAP 相机下联合训练。对留出原图的评估衡量有相机和面部姿态观测时的同片外观。

结果集中在 `runs/<视频名>-video-<UTC时间>/`。查看 `run.json` 的阶段和用时，`selection/selection.json` 的名单和偏航记录，`model/` 下的 PLY、配套相机、拟合检查点、留出近景和逐张渲染预览。默认预算为场景初始化 18,000 步、头部独立高斯训练 0 步、联合 520 步。用 `python3 run_video.py --help` 查看参数。指定 `--output` 可固定输出目录，`--resume` 使用同一个输入、目录和参数继续已完成阶段后的工作。

原始 `demo2.mp4` 的首次完整实测记录在 `runs/demo2-video-first`，使用 4,000 步场景初始化和 520 步联合拟合。最终结果与范围见该目录的 `director-brief.md`。

## 不依赖脸部关键点的侧面帧定位

```bash
python3 run_video.py /path/to/portrait.mp4 --side-poses
```

已有结果可在原命令上追加 `--resume --side-poses`。Demo2 的实测入口如下。

```bash
cd ~/projects/vhap-scene-fusion
python3 run_video.py ~/data/demo2_scene/src/demo2.mp4 \
  --output ~/projects/vhap-scene-fusion/runs/demo2-video-first \
  --background-steps 4000 --resume --side-poses
```

每帧按名字取原场景相机，从可信正脸帧的世界位姿取得初值。用原视频中间帧跟踪头部纹理；具备至少三张可信锚帧的纹理轨迹可三角化，经过遮挡、分布、留出纹理点和轮廓检查后更新转角及位置。其余帧沿原相机的图像平面用实拍头部外轮廓校正位置，沿用传播来的朝向与深度。该旁路面向身体相对稳定、头部连续可见且已有场景相机的环绕视频。

`side/side-poses.json` 保存逐帧位姿、纹理检验、轮廓得分和可用于有朝向监督训练的候选标记。自动生成的 `side/side-review.jpg` 按左右固定目标角度展示原图、世界初值和校正渲染。Demo2 在按初值角度固定的 33 张大侧角帧中有 6 张通过纹理位姿证据，其余 27 张用轮廓定位，平均轮廓 IoU 从 0.842 提高到 0.903。4 张隐藏脸部检测点的评估帧，其帧内中位重投影误差再取中位数，从 2.68 px 降到 1.02 px。接近 90° 的位置校正仍主要依赖轮廓，完整转角和侧面外观待更多可见证据验证。

传入 `--side-pose-weights ~/projects/vhap-scene-fusion/assets/6drepnet360/full-rotation.pth` 可启用 [6DRepNet360 Full Rotation](https://github.com/thohemp/6DRepNet360) 的 RGB 侧脸定位。头部掩膜确定裁片，正脸训练帧对齐网络和 FLAME 的坐标轴。侧面候选使用网络给出的偏航和轮廓给出的位置，和只调位置的渲染比较后再接收。结果在 `side-image/side-poses.json`，对照图从最侧的候选帧自动生成。已有训练可单独运行 `image_profile_pose.py` 并传入已保存的 `--baseline` 位姿文件。此阶段输出逐帧姿态，静态 PLY 保持原导出帧。

## VHAP 跨视角冷启动

对已登记相机的环绕视频，从 `guided_vhap_input.py` 生成输入。需要完整颈部时，先用 `prepare_head_neck_frames.py` 根据同帧肩点调整裁片，再将该输出目录作为 `sequence`。相机和一个全序列尺度在 VHAP 拟合期间固定。

```bash
project="$HOME/projects/vhap-scene-fusion"
cd "$project/upstream/VHAP-6e1e770"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONPATH="$PWD:$project" \
  "$HOME/venvs/vhap/bin/python" -u -B "$project/run_vhap_cold_head_pose.py" \
  --sequence "$sequence" --calibrated-start --output "$output"
```

输入的 `initialization.json` 从正脸 STAR 观测重建共用坐标系下的头部初值。缺少 STAR 的大侧角帧使用弱侧倾和光轴深度约束，侧倾作用于根旋转与颈旋转的组合。通过留出标定的 6DRepNet360 可为这些帧补充初始朝向。共享头型、网格偏移、颈部旋转和残差纹理按原版初始化，并由原版阶段重新优化。

已有自动检测的原生颈缘时，可同时传入 `--scene`、`--targets` 和 `--native-fallback`。管线核验图像与投影后，仅把拟合组的可见颈缘加入全序列优化。输出的 `initialization-proof.json`、`result.json` 和原版 `image_grid` 记录冷启动、训练以及轮廓实际使用情况。

可用 `fit_vhap_neck_sequence.py --local-mesh` 从同一次冷启动的网格继续精修颈部，保持合成头部朝向。跨视角检查应在同一固定相机下回读逐帧网格，并同时核对原相机的前喉和后颈。评估点须在所有相关阶段留出，单目绕拍的跨时刻回读没有同期真值。

## 联合训练

```bash
cd ~/projects/vhap-scene-fusion
./run_demo2.sh --train
```

沿用 [OmniRe 的场景图与联合光栅化](https://github.com/ziyc/drivestudio/blob/e59bda4fa681f829dbb1d65f0de582b0f633c450/models/trainers/base.py#L342-L432)。原始 600k 头部检查点提供绑定高斯，VHAP 提供逐帧关节；场景点云同时包含房间与身体。两个节点在同一 COLMAP 相机和同一深度排序下输出一张图。

默认跑 520 步。先训练头部与场景的颜色，104 步后放开头部三角形局部位置、轴长和透明度。头部局部位置由每帧绑定三角形传递，每轴变化最多为参考 FLAME 坐标的 0.004 单位，轴长比例限制为 exp(±0.18)。场景与身体使用同一静态点云，训练它在可观测区域的 DC 颜色，场景几何和全部相机固定。脸部的 L1 加 SSIM 损失只更新头部，身体与房间的 RGB 损失只更新场景。用联合排序后的头部有效可见度约束允许区外的覆盖。

默认用 13 帧优化，以 `00428.jpg`、`00439.jpg`、`00549.jpg`、`00560.jpg` 两段相邻帧作留出。共享尺度只用训练帧估计。留出仍通过原始 STAR 关键点取得头部姿态，因此衡量本轮未使用其 RGB 更新参数时的外观，属于同片、有姿态观测的评估。上游场景同样来自本段视频。

`runs/demo2-joint-<UTC时间>` 下保存训练参数、优化器状态、每帧指标、留出对照图、带身体和场景的静态融合 PLY 与对应相机。查看器请从同一个输出目录取 `point_cloud.ply` 和 `cameras.json`。静态 PLY 选用 `00494.jpg` 姿态，训练状态保留两个节点的共享残差。指定新输出目录和步数的示例为 `./run_demo2.sh --train runs/demo2-joint-test --steps 260 --warmup 52`。

服务器已验证的首轮结果位于 `runs/demo2-joint-first`。相对同一相机下的融合初始化，4 张留出图的脸部 RGB MAE 从 25.76 降到 7.51，颈部和衣领从 19.73 降到 12.26，身体均值从 7.59 降到 7.17，房间从 3.25 降到 2.84，均使用 0–255 量纲。两帧身体区域有小幅退化，侧脸眼镜和颈部仍有可见瑕疵。标准 PLY 的三视角回读最大 RGB 差为 0.000087（0–1）。

## 运行

服务器案例使用 `demo2_scene` 的去畸变原图、COLMAP 相机、带身体的场景点云，以及同一视频另一段训练得到的 GaussianAvatars 头部。VHAP 提供目标时刻的颈部、下颌、眼球与表情参数。融合方案来自 [streetGS：OmniRe](https://zerozero.feishu.cn/wiki/Th0RwQALWiekktk4djDcCyy1nGd)。

```bash
cd ~/projects/vhap-scene-fusion
./run_demo2.sh
```

默认将本次输出写入 `runs/demo2-<UTC时间>`。也可把一个新输出目录作为第一个参数。首次运行在 `inputs/demo2` 生成源图映射和原生 STAR landmarks，此后按原图与跟踪文件的 SHA-256 校验缓存。已验证结果在 `runs/demo2-final`。

| 文件 | 内容 |
| --- | --- |
| `point_cloud.ply` | `00494.jpg` 姿态的静态融合，保留场景 COLMAP 世界坐标 |
| `cameras.json` | 80 台同坐标场景相机，3DGS 相机到世界旋转和相机中心，`cx,cy` 用 COLMAP 半像素约定 |
| `metrics.json` | 17 帧姿态与投影、尺度留出、颜色对照、PLY 回读 |
| `*-review.jpg` | 同一原生场景相机的原图关键点叠加、融合渲染、头部 alpha 区域标记 |
| `before-after.jpg`、`head-neck-review.jpg` | 旧版和修复版整图，以及面部和领口近景 |
| `static-ply-views.jpg` | 从 PLY 和相机文件重新读取渲染的三个视角 |

## 坐标链路

1. 根据原始图像内容恢复 VHAP 时间编号对应的场景文件名，用名字关联背景相机，并与 COLMAP 原始外参逐个核对。STAR 在 `949×1687` 的去畸变原图上检测。
2. 保留 600k 检查点的 FLAME 共享形状、静态偏移和三角形绑定。把同一时刻的 VHAP 表情和关节参数交给原版 FLAME/三角绑定求值函数，用完整原生内参与脸部稳定点求解头部到 OpenCV 相机的刚体变换。
3. 从稳定面部点的跨视角一致性估计一个共享尺度，留出 4 张相机。以相机到场景的旋转和中心把已求值的高斯放进场景。中心应用相似变换，长度乘尺度，旋转左乘相同世界旋转。
4. FLAME 下缘三角形的高斯曾学习到衣领和肩部。该案例由场景资产负责衣领，去掉下缘的重复高斯。场景保留训练的 SH3，头部用 SH0，并固定几何做 156 步脸部颜色拟合。联合光栅化按同一场景深度合成。

旧版用裁片中心当主点。源帧 `00494.jpg` 的 512px 裁片主点约为 `(718,1504)`，旧设置为 `(256,256)`。旧评分还混用了源帧编号与重编号，最新旧版只写出两张掩膜。旧世界尺度从背景柱体最顶端选的 19 个点，其重投影位于原图顶边外。旧 600k 头部 PLY 已烘焙为中性姿态，使用整体旋转无法复现逐帧关节的形变。

## 已验证范围

- CUDA 偏心主点合成控制偏差约 0.00144 px。此数值用于检查投影实现。
- 17 帧脸部刚体拟合的帧内中位数再取中位数约 0.85 px，范围 0.59–1.56 px。独立于刚体拟合的嘴部观测帧中位数范围 0.76–2.50 px；它们来自同一段原视频的 STAR 检测。
- 4 张颜色留出帧的可见脸部平均绝对 RGB 误差由约 23.1 降至 10.9（0–255）。同一静态 3D 锚点的 4 张尺度留出帧中位数为 15.16、3.57、1.27、2.39 px，静态锚点检验仍受姿态和真实运动影响。
- 80 台背景相机的名称、旋转和位移经 COLMAP 交叉验证。参考帧 PLY 回读渲染最大 RGB 差约 0.000185（0–1），并以侧车相机文件重新渲染了正面及两边的三个视角。
- 全图的头已回到领口，眼镜、发际和颈部近景仍有瑕疵。当前尺度来自同一次采集的一致性估计，实际长度及完整绕拍外观仍待独立验证。静态 PLY 固定为 `00494.jpg` 的姿态，其余时刻提供同相机的逐帧渲染。
