from pathlib import Path
from datetime import datetime,timezone
import json,shutil
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-native-motion-tracks-075';S=Path('D:/Projects/EngineLab/engine-home')
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
assert read(R/'visual-review.json')['hand_box_true_fingertip_labels']==0
file=L/'knowledge/32_dynamic_human_scene_reconstruction.md';text=file.read_text(encoding='utf8')
addition='''局部图像跟踪也不能直接替代手指观测。沿一条近似直线的边缘滑动时，多个位置可能具有相似图块；往返跟踪一致、相关性很高，也未必对应同一个表面点。手部矩形框还可能包含衣服下摆。应先检查匹配是否唯一、点落在哪个可见物体上，再决定能否加入运动约束。[原生图块对应中的反例](../experiments/37_dynamic_human_reconstruction.md#局部图像对应的歧义)展示了这种区别。

'''
marker='### 相机与人物必须对应同一个拍摄时刻'
assert marker in text
if '局部图像跟踪也不能直接替代手指观测' not in text:file.write_text(text.replace(marker,addition+marker,1),encoding='utf8')
evidence=f'''

### 局部图像对应的歧义

用13台训练相机的七个偶数时刻、共91张原生照片，检查照片纹理的运动。没有更新高斯或人体参数，也没有使用另外三个机位和奇数帧。参考帧的人物内部来自此前核对过的保守标签；通过往返跟踪、直接与逐帧跟踪一致性、图块相关性和前景范围筛选，5933个特征种子留下7277条跨帧候选观测。它们是局部图像匹配，不是人体关节真值。

![前臂边界上图块相似但位置发生滑动的候选](../outputs/{R.name}/review/largest-model-disagreement.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap · 原照片上的特征标注，非重建效果图。

上图的候选沿手臂边界滑动。即使匹配图块的相关峰值达到0.9959，附近仍有几乎同样高的分离峰，峰值间隔只有约0.000089。相关性不是正确对应的概率。加入不依赖模型误差的局部匹配唯一性筛选后，这个候选被剔除，总候选数降为5117条。

两版冻结模型都具有足够源像素不透明度和人物贡献的共同子集为5072条。模型采用可见高斯中心投影位移的加权平均，与图像候选的端点差异中位数分别为0.793和0.801像素。这个比较偏向可跟踪的衣物纹理；高斯中心的加权位移也不等于每个有限高斯内部像素的精确表面运动，不能据此宣布人体运动已正确或模糊原因已经确定。

![位于手部框内但实际落在衣服下摆的特征](../outputs/{R.name}/unique-patches/review/detail00-camera03-frame000744.png)
> 图源：同上 · 前两列为原照片，后两列加特征标注；颜色标注不参与训练。

筛选后落在“手部框”的10条观测全部查看，其中8条实际属于衣服下摆或相邻裤子，另2条只在手掌附近。没有获得可作为独立指尖真值的标签。8张固定机位时间覆盖图、全部10条手部框候选和最大模型差异图均已查看，另检查了筛选前的两个诊断图。**这些候选没有送入手指训练。** 后续以原照片中手部、鞋部附近的颜色监督为对照，检验小区域的损失权重，而不把匹配不确定性转成强运动约束。

报告：[图像跟踪条件](../outputs/{R.name}/protocol.json)、[模型位移定义](../outputs/{R.name}/model-motion-protocol.json)、[唯一性筛选条件](../outputs/{R.name}/unique-patches/protocol.json)、[筛选结果](../outputs/{R.name}/unique-patches/summary.json)、[共同子集结果](../outputs/{R.name}/unique-patches/model-summary.json)、[实际看图](../outputs/{R.name}/visual-review.json)、[最终判断](../outputs/{R.name}/final-summary.json)。
'''
file=L/'experiments/37_dynamic_human_reconstruction.md';text=file.read_text(encoding='utf8')
if '### 局部图像对应的歧义' not in text:file.write_text(text+evidence,encoding='utf8')
assets=['protocol.json','model-motion-protocol.json','unique-patches/protocol.json','unique-patches/summary.json','unique-patches/model-summary.json','visual-review.json','final-summary.json','review/largest-model-disagreement.png','unique-patches/review/detail00-camera03-frame000744.png']
file=S/'scripts/sync-gs-lab.mjs';text=file.read_text(encoding='utf8')
if R.name+'/${name}' not in text:
    lines=text.splitlines(keepends=True);index=next(i for i,l in enumerate(lines) if 'dynamic-human-enerf-rgb-pose-correction-074/${name}' in l)+1
    lines.insert(index,'  ...'+json.dumps(assets)+f".map(name => [`outputs/{R.name}/${{name}}`, '原生图像运动对应诊断']),\n");file.write_text(''.join(lines),encoding='utf8')
save(R/'published-resources.json',dict(paths=assets,licensed_models_exported=False))
file=L/'manifests/dynamic-human-work.json';m=read(file);m.update(status='native_motion_diagnosis_complete_detail_weight_training_running',
    current_runs=[dict(name='076_native_detail_weight_training_and102known_camera_render',session=46567,linux_training_pid=318,status='verified_running')],
    outdoor_detail_weight='outputs/dynamic-human-enerf-detail-weight-076/protocol.json',last_recorded_utc=datetime.now(timezone.utc).isoformat())
save(file,m);shutil.copy2(__file__,R/Path(__file__).name)
print('075bounded knowledge and evidence prepared;076training status recorded.')
