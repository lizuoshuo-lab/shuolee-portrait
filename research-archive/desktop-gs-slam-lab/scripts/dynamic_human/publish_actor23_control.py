"""Use one bounded result in the existing knowledge chapter."""
from pathlib import Path
import json,shutil
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-actor23-independent-106';SITE=Path('D:/Projects/EngineLab/engine-home');read=lambda p:json.loads(p.read_text(encoding='utf8'))
c=read(O/'comparison/summary.json');t=read(O/'continuity/summary.json');v=read(O/'visual-review.json');assert c['PNG_checks']==1584 and t['PNG_checks']==300 and not v['clarity_accepted'];assert len(v['inspected'])>=12
p=L/'knowledge/32_dynamic_human_scene_reconstruction.md';s=p.read_text(encoding='utf8');shutil.copy2(p,O/'knowledge-before.md');begin=s.index('## 当前可见的重建效果');end=s.index('## 1. 静态背景与运动人物怎样共同表示',begin)
s=s[:begin]+'''## 当前可见的重建效果

![原照片与静态高斯、动态高斯、校正相机时间偏移后的动态高斯对照](../outputs/dynamic-human-actor23-independent-106/comparison/camera04-frame000613-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 机位04、留出帧613 · 原生像素，无增强。

同一组静态高斯同时拟合站立、抬脚和下蹲，容易把不同时刻的轮廓平均成重影。让局部三维结构随时间运动后，图中两个人的姿态、衣物和手持物更集中。这组对照使用同一采集地点的另一组人物，三种模型共享训练照片、初始化和更新次数；它说明动态表示怎样减轻跨时刻的监督冲突。

局部细节仍需单独检验：抬起的脚有拖影，张开的手指仍有混合与软化。第三列和第四列的区别是是否使用估计的相机时间偏移，不能仅凭这一张图判断校正对所有视角有效。灰色是检视人物的合成背景，建筑尚未与人物联合训练。[原生细节与连续回放](../experiments/37_dynamic_human_reconstruction.md#独立人物序列的静态与动态对照)保留完整条件、效果和失败案例。[连续运动表示](#时间基函数与局部运动)与[时间对应](#同一帧号不一定对应同一运动状态)解释其中的两个关键问题。

'''+s[end:]
# Separate image-selection offset from the opposite common-time convention.
begin=s.index('### 同一帧号不一定对应同一运动状态');end=s.index('### 连续人体运动如何带动高斯形状',begin);part=s[begin:end].replace('\\delta','\\eta');s=s[:begin]+part+s[end:]
needle='其中 $n$ 是图像文件的整数帧索引，$\\delta_k$ 是相对参考相机的时间偏移，以帧为单位。'
assert s.count(needle)==1;s=s.replace(needle,needle+'这里把图像索引映射到共同时间；前文的 $\\eta_k$ 则从共同时间选取图像索引。使用同一参考相机时，二者满足 $\\delta_k=-\\eta_k$，不能直接混用符号。',1)
s=s.replace('updated: "2026-10-15"','updated: "2026-10-16"',1);p.write_text(s,encoding='utf8')
p=L/'experiments/37_dynamic_human_reconstruction.md';s=p.read_text(encoding='utf8');shutil.copy2(p,O/'evidence-before.md');assert '### 独立人物序列的静态与动态对照' not in s
s+='''

### 独立人物序列的静态与动态对照

ENeRF-Outdoor 的 actor2_3 提供另一组人物，但仍是相同采集场地。15个机位在600至624帧中的13个偶数帧用于训练，共195张1920×1080照片；00、06、12机位在该区间只用于外观评测。静态高斯、同编号动态高斯、带时间偏移的动态高斯，各从相同初始化训练15000次。动态模型使用局部形变图，本次三组对照没有使用MANO或SMPL人体模板，效果不能归因于模板下载。

初始化、照片采样和背景采样相同；三组最终高斯数量不同，因此这里比较相同更新预算，未声称相同参数量或严格相同计算量。其他机位的时间偏移只在582至594帧的独立区间估计，未把校准图像用于高斯初始化或外观训练；这些机位属于外观留出，不属于完全未见过的传感器。偏移单位是图像索引，不是经过标定的硬件时间戳。

'''
labels={'validation_time':'已知机位的180张留出时刻','validation_camera_time':'其他机位的36张奇数帧','validation_camera':'其他机位的39张偶数帧'}
for key,label in labels.items():
    m=c['splits'][key]['foreground_psnr_db'];s+=f"{label}中，静态、动态、时间校正动态三组的平均前景PSNR分别为 **{m['static']:.2f}、{m['dynamic_nominal']:.2f}、{m['dynamic_clock']:.2f} dB**。\n\n"
s+='时间校正相对同编号动态模型，在180张已知机位留出图像中有170张改善、10张下降；最大下降发生在09机位601帧，约0.92 dB。其他机位的75张验证图像均提高，但平均分提升仍不能替代局部细节验收。\n\n'
s+='''![早期抬脚动作的完整人物对照，保留局部失败](../outputs/dynamic-human-actor23-independent-106/comparison/camera00-frame000601-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 外观留出机位00、帧601 · 原照片、静态、动态、时间校正动态。

动态模型减少了身体和下蹲姿态的重影，但抬起的脚仍有半透明拖影。下图手部上下文中，手指形状仍与原照片不完全一致。**本轮取得主体外观改善，尚未通过完整人物清晰度验收。**

![手部与物体上下文中的真实细节](../outputs/dynamic-human-actor23-independent-106/comparison/camera13-frame000613-crouching_hands_object_context.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 机位13、留出帧613 · 原生像素。

完整原分辨率评测包含264个情境，其中255张为正式验证照片、9张为明确标注的训练检查；72个细节窗口在查看模型输出前固定。PSNR使用原始前景掩码，也包含玩偶及掩码误差；矩形窗口是上下文，不是精确的手指或腿部语义标注。底部窗口可能截掉抬起的脚，故该失败必须结合完整人物图判断。

[机位00连续回放](../outputs/dynamic-human-actor23-independent-106/continuity/camera00-native-continuity.mp4)与[机位13连续回放](../outputs/dynamic-human-actor23-independent-106/continuity/camera13-native-continuity.mp4)各含25个连续帧，以5帧/秒慢放；这不是采集帧率。前者全部为外观留出，后者包含13张训练帧和12张留出帧。视频是压缩的观看副本，指标来自未增强的原生图像。

[完整结果](../outputs/dynamic-human-actor23-independent-106/final-summary.json)、[实际看图范围](../outputs/dynamic-human-actor23-independent-106/visual-review.json)保留校验与结论边界。该局部前景模型尚未与大场景背景共同优化，也未完成OmniRe接入。

独立复算检查了1584张评测PNG与300张连续序列PNG；实际查看15幅单帧对照和6幅连续帧抽查图。视频完成编码、帧数和解码检查，未作全部帧的视觉验收。
''';p.write_text(s,encoding='utf8')
files=['final-summary.json','visual-review.json','comparison/camera04-frame000613-full-person.png','comparison/camera00-frame000601-full-person.png','comparison/camera13-frame000613-crouching_hands_object_context.png','continuity/camera00-native-continuity.mp4','continuity/camera13-native-continuity.mp4']
p=SITE/'scripts/sync-gs-lab.mjs';s=p.read_text(encoding='utf8');assert O.name+'/${name}' not in s;lines=s.splitlines(keepends=True);i=next(i for i,line in enumerate(lines) if 'dynamic-human-hand-photo-pose-105/${name}' in line)+1;lines.insert(i,'  ...'+json.dumps(files)+'.map(name => [`outputs/'+O.name+'/${name}`, \'独立人物序列的静态与动态对照\']),\n');p.write_text(''.join(lines),encoding='utf8')
(O/'published-resources.json').write_text(json.dumps(dict(paths={O.name:files},licensed_models_exported=False),indent=2)+'\n',encoding='utf8');print('Existing chapter updated with one bounded comparison;7explicit resources,no models.')
