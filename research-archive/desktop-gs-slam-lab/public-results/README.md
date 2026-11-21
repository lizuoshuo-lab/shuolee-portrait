# 可公开的模型和实验结果

三个 SplaTAM 模型均来自公开 TUM RGB-D 数据集的完整关联序列。各目录下的 params.npz 是原训练输出，metrics.json 给出运行标识、点数、有效帧数和独立轨迹评估。上传前将每个模型的 SHA-256 与原评估输入记录比对。数据归属及 CC BY 4.0 许可见 splatam/DATA-LICENSE.json。模型是基于数据生成的衍生结果。

benchmark-metrics.json 汇集 151 份公开数据实验的标量记录，保留原实验相对路径。upstream-revisions.json 记录依赖来源和固定版本，upstream-patches 保存核查过的本地差异。source-snapshots 保存可公开的实验源码快照。原始数据可从数据提供方获取。人物实拍及其重建资产在 00 管理。
