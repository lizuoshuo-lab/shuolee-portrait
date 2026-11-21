---
id: "gslab-exp-tum-rgbd-mapping-20261009"
slug: "tum-rgbd-known-pose-mapping-depth-ablation"
title: "TUM 短序列建图：深度损失怎样改变几何"
order: 5
status: "completed_fixed_pose_mapping_and_depth_ablation"
updated: "2026-10-09"
sources:
  - "https://github.com/nerfstudio-project/gsplat/blob/937e29912570c372bed6747a5c9bf85fed877bae/gsplat/rendering.py"
  - "https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/scripts/splatam.py"
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats"
---

# 从 RGB-D 初始化到已知位姿高斯建图

在 TUM fr1_desk 的约1.97秒片段上，用10个训练视角深度初始化30,000个高斯，固定 GT 相机优化1000步。五个留出视角平均 RGB PSNR 从14.3256提升至21.5193dB，pooled 覆盖条件深度 RMSE 从0.193227降至0.089551m。

随后只移除深度损失，颜色 PSNR 仍接近21.48dB，但深度 RMSE为0.173505m。两组都用了深度初始化，因此消融只检验深度损失，不是纯RGB系统与RGB-D系统的对比。

## 配置与划分

- 数据为完整公开 TUM RGB-D fr1_desk 包，来源/校验/许可保存在数据清单；当前只使用一个短片段。
- 最近 RGB-depth 与 RGB-GT 差严格小于20ms，32Hz上限，禁止深度文件复用。549个可用关联帧中，选择frame_id30–86每4帧共15视角，跨度1.968082秒。
- 训练frame_id：30、34、42、46、54、58、66、70、78、82；留出：38、50、62、74、86。初始化、RGB/深度损失和梯度更新仅访问训练帧，留出GT仅作为评测相机。
- 图像160×120、深度PNG/5000为米，最近邻缩深度，K按像素中心仿射缩放。GT的xyzw四元数归一化并转换为c2w，再求w2c。
- 固定30,000高斯，每训练帧初始化3,000粒；seed0、1000步。颜色为常量RGB，无SH、增密、剪枝或相机优化。
- RGB+ED渲染；MSE RGB、覆盖条件深度MAE、有效深度像素alpha惩罚。深度权重0.1/0的两个实验，其余设置相同。

## 初末与消融结果

| 留出指标 | 相同初始化 | RGB+深度损失 | 无深度损失 |
| --- | --- | --- | --- |
| 平均RGB PSNR | 14.32556 dB | 21.51934 dB | 21.47949 dB |
| pooled覆盖条件深度RMSE | 0.193227 m | 0.089551 m | 0.173505 m |
| 有效深度覆盖 | 99.4219% | 99.6664% | 99.8898% |
| 逐视角含洞RMSE均值 | 0.224666 m | 0.139819 m | 0.179943 m |

加入深度损失时，当前覆盖条件深度RMSE约低48.4%，但同时必须报告覆盖差异；含洞补充指标也降低。此百分比只描述这两个真实运行，不能作为普适性能提升承诺。

训练视角的RGB PSNR为23.6911dB（有深度损失）与24.5376dB（无深度损失），pooled深度RMSE为0.063924m与0.176869m。更高训练颜色质量并没有带来更好的深度。

RGB+深度运行的优化循环4.5363秒，排除初始CUDA调用、数据准备、评估和文件导出。30k粒子与低分辨率小片段的时间不能外推为完整SLAM吞吐。

![留出视角 RGB 与深度初末对照](../outputs/tum-rgbd-gsplat-001/comparison.png)

![移除深度损失后的对照](../outputs/tum-rgbd-rgbonly-001/comparison.png)

图像来自TUM RGB-D fr1_desk，CC BY 4.0。办公室的大结构被恢复，细节仍模糊；图像预览只帮助观察，定量结论以指标定义为准。

## 指标统计口径

pooled RMSE先汇总各视角覆盖像素的平方误差，再除以覆盖像素总数开方，而不是直接平均各帧RMSE。含洞指标将未覆盖有效像素的预测深度设为零，再计算每个视角RMSE，最后取视角均值。二者统计口径不同，不能直接互减解释。

完整来源：[RGB+深度指标](../outputs/tum-rgbd-gsplat-001/metrics.json)、[无深度损失指标](../outputs/tum-rgbd-rgbonly-001/metrics.json)。两组初始指标完全一致，训练与留出RGB/深度文件互不复用，checkpoint参数数量与有限性检查通过。

## 可复现代码

[训练脚本](../scripts/train_tum_gs.py)、[数据与几何处理](../scripts/tum_rgbd.py)。

```powershell
.\.venv\Scripts\python.exe scripts\run_local.py scripts\train_tum_gs.py --run tum-rgbd-gsplat-new
.\.venv\Scripts\python.exe scripts\run_local.py scripts\train_tum_gs.py --depth-weight 0 --run tum-rgbd-rgbonly-new
```

run名应未被使用，原始结果不覆盖。输出包含逐视角RGB/深度/alpha、初末指标、配置、曲线、模型及后续位姿恢复用的tracking bundle。

## 适用边界

相机GT在训练与评测中固定，当前记录验证地图优化和深度损失，不报告轨迹精度。邻近帧留出仅评估这一段的视角插值；一个种子和两秒数据不能代表完整基准。RGB-depth时间差、最近GT、未去畸变及注册残差没有补偿；期望深度在边界也可能混合多个表面。

[RGB-D深度、覆盖与关联知识](../knowledge/07_rgbd_depth_coverage.md) 解释此实验的公式、代码和容易误解的地方。
