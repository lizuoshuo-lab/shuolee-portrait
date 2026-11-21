from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, shutil, numpy as np
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-head-motion-066';G=L/'outputs/dynamic-human-enerf-residual-graph-067';SITE=Path('D:/Projects/EngineLab/engine-home')
read=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,x:p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
pair=read(R/'paired-comparison/summary.json');temporal=read(R/'temporal-comparison/summary.json');pose=read(R/'projection-comparison.json')
assert read(R/'head_only/native-evaluation/independent-image-audit.json')['images']==102
small={k:v for k,v in pose.items() if k!='records'};small['full_local_report_sha256']=hashlib.sha256((R/'projection-comparison.json').read_bytes()).hexdigest();save(R/'projection-summary.json',small)
detail={n:dict(mean_psnr_delta_db=float(np.mean([r['delta_db'] for r in pair['details'] if r['region']==n])),improved=sum(r['delta_db']>0 for r in pair['details'] if r['region']==n),count=6) for n in ['dark_head','dark_forearm','light_forearm','dark_shoe']}
old=read(L/'outputs/dynamic-human-enerf-articulated-control-064/visual-review.json')
save(R/'visual-review.json',dict(status='12fixed_fullperson24fixed_details_and_worst_actually_viewed',full_person=old['full_person'],details=old['details'],additional_worst='camera05-frame000747-full-person.png',findings=['脸部、眼镜与头发边界较064明显集中；六个头部裁图均实际查看，仍有平滑。','两类前臂的外形大体保留，但摆动手指、皮肤纹理和袖口没有可靠恢复；鞋带与裤脚仍模糊。','四个固定机位的玩偶仍明显拉花、拖影，遮挡处腿部厚边尚在。','最差机位05帧747的玩偶仍横向扩散，浅色上衣和肢体边界存在偏差。'],temporal_visual_scope='Three separated validation frames per four cameras inspected; all4videos decoded, not claiming full playback visual review.',decision='Adopt066 as a development comparison starting point for residual motion, not a final clarity acceptance.',acceptance='not_passed',goal_complete=False))
save(R/'final-summary.json',dict(status='head_only_fit_and15000update_reconstruction102native_audits_complete_component_improvement',pose_projection=pose['summary'],validation_foreground=pair['validation_foreground'],fixed_detail_regions=detail,temporal_difference_residual=temporal['mean_temporal_residual'],interpretation='Head/neck isolation retains facial improvement with unchanged nonhead skeleton coordinates; global foreground mean+0.162885dB,62/90 improve. Hands, clothes, shoes and heldobjects remain insufficiently clear.',invariance='Otherposecoefficients,translation,shape,scale,timing frozen; nonhead bodyjoint coordinates unchanged. SMPL posecorrectives slightly move nonhead mesh vertices: do not claim the entire mesh is frozen.',next='Samebudget residualgraph motion on top of066frozenbodymotion; appearance validation still development only.',qualification='Single seed and repeatedly inspected development-time validation. Predicted keypoints and released mattes are notGT. No newcamera or largescene completion claim.',goal_complete=False))
p=L/'knowledge/32_dynamic_human_scene_reconstruction.md';text=p.read_text(encoding='utf8')
text=text.replace('上图中的头部重影明显减少，支持进一步检验头部运动自由度的作用。不过本轮同时重新拟合了身体姿态，不能把改善只归因于头部解锁；下一轮需要固定其他姿态参数进行对照。','上图中的头部重影明显减少，支持检验头部运动自由度的作用。不过这一组合修正同时重新拟合了身体姿态，不能把改善只归因于头部解锁。为区分两者，下面进一步固定身体运动，单独拟合头颈。')
text=text.replace('当前保留旧身体运动基线，下一步单独检验头颈运动，再处理不服从人体骨架的表面变化。','这一组合修正没有替换旧身体运动基线；随后单独检验头颈运动，以保留脸部改善并减少其他区域退化。')
addition=r'''

### 将头部运动与衣物、手持物运动分开

![只改变头颈运动后的全人物对照](../outputs/dynamic-human-enerf-head-motion-066/paired-comparison/camera04-frame000755-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实际训练 · 原照片、原身体运动、仅头颈修正 · 原生像素。

人体不同部位需要不同的运动自由度。先保持身体关节、世界平移、体形、相机与时间偏移不变，只更新颈部和头部的局部旋转，能够检查原来的脸部重影是否与这两处运动约束有关。本地对照将两个关节的轴角向量写为连续时间三次多项式：

$$
\boldsymbol\omega_j(\tau)=\sum_{k=0}^{3}\mathbf a_{j,k}\tau^k,
\qquad \tau=\frac{t-750}{6},\qquad j\in\{\mathrm{neck},\mathrm{head}\}.
$$

$t$ 表示带相机时间偏移的连续帧索引，不是秒；$\tau$ 无量纲。轴角向量的方向表示旋转轴，长度表示以弧度计的旋转角；系数 $\mathbf a_{j,k}$ 亦以弧度计。两个人、两个关节、四组系数、每项三个分量，共优化 48 个标量。其余身体骨架坐标在核对中保持不变；SMPL 的姿态修正项仍会使部分非头部表面产生小幅变化，所以不能说整个非头部网格完全不动。

![头颈隔离对照中的脸部细节](../outputs/dynamic-human-enerf-head-motion-066/paired-comparison/camera04-frame000755-dark_head.png)
> 图源：同上 · 与上方全图对应的脸部区域。

单独修正头颈后，眼镜、头发和脸部轮廓集中起来。六个固定头部区域的平均 PSNR 提高约 3.98 dB；全前景平均提高约 0.16 dB，90 张开发验证图中有 62 张改善。与上一种全身联合修正相比，这个结果支持保留头颈运动调整，但仍不能宣称手指和衣物细节已经恢复。

上方完整画面中的玩偶仍有横向拖影，因为它的运动并不由人体骨架唯一决定；衣服也可能相对身体滑动和弯曲。接下来的表示需要在骨架提供的大体运动之外加入局部变形，并约束相邻区域的运动一致性。只有在相同原图、训练预算和固定观察区域下真正恢复纹理，才能说明这层自由度有用。[完整对照与保留的负例](../experiments/37_dynamic_human_reconstruction.md#只优化头颈的隔离对照)。

'''
if '### 将头部运动与衣物、手持物运动分开' not in text:text=text.replace('### 连续运动中的同预算对照',addition+'### 连续运动中的同预算对照')
p.write_text(text,encoding='utf8')
p=L/'experiments/37_dynamic_human_reconstruction.md';text=p.read_text(encoding='utf8')
addition='''

### 只优化头颈的隔离对照

保留064的身体姿态、世界平移、体形、尺度、相机与时间偏移，只用七个偶数时刻的546个头部预测观测拟合48个颈部／头部旋转系数，更新3000次。白衣人物的鼻子和眼睛继续排除；预测关键点不等于真实关节。195个相机时刻组合在CPU完整复算，非头部身体关节坐标最大变化为0；SMPL姿态修正项仍导致部分非头部网格顶点移动，最大约0.00734场景单位。

预留时刻468个头部观测的投影误差中位数从7.550降至3.355像素，身体和脚部从5.938降至5.936像素。头部误差均方根仍为31.204像素，未解决错误预测和遮挡大残差。

使用相同105张训练图、高斯初值、抽样顺序、随机背景与15000次更新，得到246435个高斯。102张原生渲染均完成源数据、保存像素与指标复算；实际查看12张固定全图、24个细节区域和机位05第747帧最差退化样例。

![隔离头颈运动后的完整画面](../outputs/dynamic-human-enerf-head-motion-066/paired-comparison/camera04-frame000755-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实际训练 · 原照片、064、066。

90张开发时间验证图的全前景PSNR均值从24.1344增至24.2973dB，62张改善、28张退化。六个固定头部区域平均提高3.9776dB；黑衣前臂为−0.1861dB，白衣前臂为−0.1151dB，鞋部为+0.0618dB。区域矩形包含背景和遮挡，不能等同于精确人体部位分割。脸部改善明显，手指、玩偶和衣物局部仍未清晰恢复。

![保留的最差全前景退化样例](../outputs/dynamic-human-enerf-head-motion-066/paired-comparison/camera05-frame000747-full-person.png)
> 图源：同上 · 机位05、第747帧，PSNR变化−0.3284dB。

75组连续差分残差均值从0.04152降至0.04089；四段六帧对照视频均解码检查。该量包含真实运动、遮挡及掩码误差，不是光流补偿后的闪烁指标；只目视检查每个机位三个分离时刻，没有声称完整播放验收。视频：[01](../outputs/dynamic-human-enerf-head-motion-066/temporal-comparison/camera01-reserved-time-comparison.mp4)、[04](../outputs/dynamic-human-enerf-head-motion-066/temporal-comparison/camera04-reserved-time-comparison.mp4)、[08](../outputs/dynamic-human-enerf-head-motion-066/temporal-comparison/camera08-reserved-time-comparison.mp4)、[17](../outputs/dynamic-human-enerf-head-motion-066/temporal-comparison/camera17-reserved-time-comparison.mp4)。4帧/秒仅用于按顺序展示。

将066保留为下一轮局部变形的开发对照基础，**清晰人物重建仍未通过**。这些时间与相机已多次用于诊断，最终仍需分离的新时间窗口与相机验证。

报告：[结论](../outputs/dynamic-human-enerf-head-motion-066/final-summary.json)、[头部拟合条件](../outputs/dynamic-human-enerf-head-motion-066/protocol.json)、[完整人体复算](../outputs/dynamic-human-enerf-head-motion-066/independent-audit.json)、[隔离条件](../outputs/dynamic-human-enerf-head-motion-066/head-only-invariance-audit.json)、[投影摘要](../outputs/dynamic-human-enerf-head-motion-066/projection-summary.json)、[保存图像复算](../outputs/dynamic-human-enerf-head-motion-066/head_only/native-evaluation/independent-image-audit.json)、[成对指标](../outputs/dynamic-human-enerf-head-motion-066/paired-comparison/summary.json)、[连续检查](../outputs/dynamic-human-enerf-head-motion-066/temporal-comparison/summary.json)、[看图范围](../outputs/dynamic-human-enerf-head-motion-066/visual-review.json)。
'''
if '### 只优化头颈的隔离对照' not in text:p.write_text(text+addition,encoding='utf8')
p=SITE/'scripts/sync-gs-lab.mjs';text=p.read_text(encoding='utf8')
if 'dynamic-human-enerf-head-motion-066/${name}' not in text:
 lines=text.splitlines(keepends=True);i=next(i for i,l in enumerate(lines) if 'dynamic-human-enerf-direct-pose-065/${name}' in l)+1
 lines.insert(i,"  ...['final-summary.json','protocol.json','independent-audit.json','head-only-invariance-audit.json','projection-summary.json','head_only/native-evaluation/independent-image-audit.json','paired-comparison/summary.json','temporal-comparison/summary.json','visual-review.json','paired-comparison/camera04-frame000755-dark_head.png','paired-comparison/camera04-frame000755-full-person.png','paired-comparison/camera05-frame000747-full-person.png',...['01','04','08','17'].map(c => `temporal-comparison/camera${c}-reserved-time-comparison.mp4`)].map(name => [`outputs/dynamic-human-enerf-head-motion-066/${name}`, '头颈隔离与真实人物重建']),\n");p.write_text(''.join(lines),encoding='utf8')
p=L/'manifests/dynamic-human-work.json';m=read(p);m.update(status='head_motion_component_verified_residual_graph_training_running',current_runs=[dict(session_id=53116,script='run_enerf_graph_control.sh',output='outputs/dynamic-human-enerf-residual-graph-067')],current_best_continuous_control='outputs/dynamic-human-enerf-head-motion-066/head_only',outdoor_head_motion='outputs/dynamic-human-enerf-head-motion-066/final-summary.json',outdoor_residual_graph='outputs/dynamic-human-enerf-residual-graph-067/runtime-smoke.json',remaining_quality_issue='Head motion improves faces; hands, clothing, shoes, occlusion and heldobjects still not clear. Clarity acceptance remains false.',last_recorded_utc=datetime.now(timezone.utc).isoformat(),goal_turn_classification='Progress:066 trained and audited,12full24details plusworst visuallyreviewed;067actual graphtraining launched after CPUmath/CUDAgradient/density checks.',next_actions=['Finish067 samebudget residualgraph training and all102native evaluations; independentlyaudit pixels and fixedregions against066','Inspect heldobject/hand detail and temporal residuals, preserving all regressions','Verify new times and cameras after development clarity improves; do not claim large-scene completion'])
entry='066headonly15000update reconstruction and102nativeaudits,12full24detailworstreview;head+3.98dB,foreground+0.163dB;component adopted,overallclarity notpassed'
if entry not in m['completed']:m['completed'].append(entry)
save(p,m);shutil.copy2(__file__,R/Path(__file__).name)
print(json.dumps(dict(validation=pair['validation_foreground'],details=detail,temporal=temporal['mean_temporal_residual']),ensure_ascii=False))
