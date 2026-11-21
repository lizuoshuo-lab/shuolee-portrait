"""Publish one bounded mechanism explanation, keeping diagnostics in evidence."""
from pathlib import Path
import json,shutil
L=Path(__file__).resolve().parents[2];H=L/'outputs/dynamic-human-enerf-hand-observability-073';R=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074';SITE=Path('D:/Projects/EngineLab/engine-home')
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
assert read(H/'independent-audit.json')['old_hand_fraction_passing04']==1
pilots=read(H/'hand-pilot-rgb/observations.json')['records'];cams=sorted({x['camera'] for x in pilots})
review=dict(status='all14correctRGBpilot_panels_22hands_actually_reviewed',files=[f'hand-pilot-rgb/camera{c}-all-points.png' for c in cams],
 images='Fourcolumns:originalphoto,historicalwholebodyBGR,correctedwholebodyRGB,correctedhand-specificRGB. All21pointsdrawnunfiltered;notvisibilitytruth.',
 narrative_zh='局部手部模型在部分视角的手掌和外轮廓附近给出不同候选，但抓握与背面手指仍难逐点确认。全身模型的RGB修正没有消除遮挡猜测。没有像素级手指真值、时间连续性或跨视角独立验证，不能把候选直接作为训练约束。',
 adopt_as_hand_supervision=False,next='Onlyafterexplicitvisibilityandidentityscreening, reviewtrain-time handtracksandcrossviewconsistency;noautomaticthresholdacceptance.',goal_complete=False)
save(H/'hand-pilot-rgb/visual-review.json',review)
summary=read(H/'summary.json');summary.update(status='hand_observability_diagnostic_complete_no_hand_supervision_adopted',
 old_coordinate_convention='HistoricalwholebodyBGRinput mismatch established afterward in074; statistics and60sourcecrop review remain a diagnosis of archived data, not correctedRGBpose performance.',
 source_review='source-visual-review.json',pilot_review='hand-pilot-rgb/visual-review.json',new_model_sha256=read(H/'hand-model-acquisition.json')['model_sha256'],
 actual_review=dict(source_panels=15,source_hands=60,RGBpilot_panels=14,RGBpilot_hands=22),
 result_zh='原阈值未剔除任何手部预测；遮挡处仍会预测到玩偶或另一个人。局部手部候选未进入高斯训练。RGB预处理错误另行修正并开展同预算重训。',goal_complete=False)
save(H/'final-summary.json',summary)
save(R/'observation-visual-review.json',dict(status='bounded_correctRGB_observation_review',body_frame750_cameras=['01','07','17'],hand_pilot_panels=14,hand_pilot_hands=22,
 findings_zh='RGB修正前后可见身体主体大致相近，但被遮挡的暗衣人物头部和玩偶附近的手仍有猜测；不能凭输入修复宣称几何或外观已正确。',goal_complete=False))
block=r'''
#### 手部观测必须区分可见性与预测分数

![原生手部照片与全身姿态模型预测，遮挡处仍可能出现完整手指](../outputs/dynamic-human-enerf-hand-observability-073/source-panels/camera04-hands.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与 RTMW 输入诊断 · 机位04、帧750 · 原生像素；彩色点为预测，不是真值或重建结果。

上图每组左侧是照片，右侧是关键点预测。可见的裸露手与被玩偶、衣服遮住的区域，都可能得到完整的21点手部结构。因此，网络的响应分数不能直接解释成“这个指尖真实可见的概率”。多人交互时，还必须确认预测属于哪一个人；几条平滑曲线并不能证明对象对应正确。

对人物 $a$ 的关节 $j$，二维观测约束可概括为

$$
\mathcal L_{\mathrm{keypoint}}
=\sum_{c,t,a,j}v_{ctaj}\,u_{ctaj}\,w_{ctaj}\,
\rho\!\left(\left\|\pi_c\!\left(\mathbf X_{aj}(t)\right)-\mathbf x_{ctaj}\right\|_2\right).
$$

$\mathbf x_{ctaj}$ 是机位 $c$、时刻 $t$ 的二维预测，单位为像素；$\mathbf X_{aj}(t)$ 为三维关节，$\pi_c$ 为相机投影。$v_{ctaj}$ 表示可见性筛选，$u_{ctaj}$ 表示人物归属筛选，均可取0或1；$w_{ctaj}$ 是保留下来的观测权重。不可见或归属不确定时，不应仅因响应较高而令其成为强约束。鲁棒损失 $\rho$ 可以减弱离群值，不能替代这两项筛选。

细节还受到检测输入分辨率限制：原照片用于高斯训练时可以保留原生像素，但全身检测器会先缩放整个人，手指只占其中很小一块。局部手部检测能让网络观察更大的手部区域，却不能创造原照片中不存在的细节。新增手部自由度之前，应先检查可见指尖、跨机位对应和跨时刻连续性，再用真实渲染验证其作用。[输入检查与局部检测候选](../experiments/37_dynamic_human_reconstruction.md#手部观测与姿态输入核查)见实验附录。

'''
file=L/'knowledge/32_dynamic_human_scene_reconstruction.md';text=file.read_text(encoding='utf8')
if '#### 手部观测必须区分可见性与预测分数' not in text:
 marker='### 相机与人物必须对应同一个拍摄时刻';assert marker in text;text=text.replace(marker,block+marker);file.write_text(text,encoding='utf8')
evidence='''

### 手部观测与姿态输入核查

对既有195张原生训练机位图像的780只手预测进行检查。15个机位第750帧的60个原生手部区域已全部查看：遮挡处有落到玩偶、衣服和另一个人上的预测。旧RTMW输出的16380个手部点分数全部大于0.4，最小值约1.115；这个阈值没有筛掉任何点。它是未经概率校准的响应分数，不能按百分比或可见性解释。

旧预测的手部范围在原生图像中的长边中位数约50.4像素，在288×384全身检测输入里约20.5像素。这是预测范围统计，不是人工测量的真实手部尺寸。按旧身体时间偏移插值到参考时刻后，逐相机留出的手指预测重投影误差中位数约10.75像素，1200个检查中472个小于6像素；这些数值描述旧预测的相互一致性，不是真值误差，不能单独证明模糊原因。

![原生照片与局部手部候选](../outputs/dynamic-human-enerf-hand-observability-073/hand-pilot-rgb/camera08-all-points.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap、RTMW与RTMPose Hand · 输入观测比较，非重建结果。

在第750帧先按源图可见性选22只右手，用官方RTMPose Hand模型产生局部候选。14张四列对照全部查看；所有21个点均画出，避免不同模型的分数范围改变显示后造成假改善。候选未加入高斯训练；被玩偶遮住的左手也未用于先导。当前没有手指真值和时序验证，仍须检查对应关系。

核对两个模型的官方压缩包后，发现Normalize配置要求转成RGB，而旧调用链将BGR交给没有颜色转换的包装库。已用原生图像独立重建官方流程：修正后的归一化输入逐元素完全相同；旧调用也能精确复现历史坐标，证实问题发生在输入通道顺序。高斯训练图像此前一直是原始RGB，此问题作用于姿态检测。官方包内的部分裁剪尺寸字段过时，实际采用ONNX输入尺寸与对应解码尺寸，未照抄不一致的默认值。

已重提取195张图像的390个人物预测，并对原图哈希、相机参数、框和训练/留出时间划分复核。三维轨迹、相机时间偏移、2500步身体运动与3000步头部运动已重拟合，195个相机时刻的身体变换通过CPU独立复算。旧相机支持集合和规范人体参考保持固定，以便隔离这次修复；没有把手指候选混入训练。修正后的关键点及拟合一致性不等于重建清晰度，后者由单独的15000步高斯对照判定。

报告：[手部诊断](../outputs/dynamic-human-enerf-hand-observability-073/final-summary.json)、[原照片检查](../outputs/dynamic-human-enerf-hand-observability-073/source-visual-review.json)、[分数独立复算](../outputs/dynamic-human-enerf-hand-observability-073/independent-audit.json)、[局部候选检查](../outputs/dynamic-human-enerf-hand-observability-073/hand-pilot-rgb/visual-review.json)、[通道顺序核验](../outputs/dynamic-human-enerf-rgb-pose-correction-074/preprocessing-audit.json)、[同一目标的轨迹比较](../outputs/dynamic-human-enerf-rgb-pose-correction-074/same-target-trajectory-audit.json)、[运动重拟合条件](../outputs/dynamic-human-enerf-rgb-pose-correction-074/motion-refit-protocol.json)、[CPU变换复算](../outputs/dynamic-human-enerf-rgb-pose-correction-074/head/independent-audit.json)、[高斯对照条件](../outputs/dynamic-human-enerf-rgb-pose-correction-074/gaussian-ablation-protocol.json)。
'''
file=L/'experiments/37_dynamic_human_reconstruction.md';text=file.read_text(encoding='utf8')
if '### 手部观测与姿态输入核查' not in text:file.write_text(text+evidence,encoding='utf8')
assets={H.name:['final-summary.json','source-visual-review.json','independent-audit.json','hand-pilot-rgb/visual-review.json','source-panels/camera04-hands.png','hand-pilot-rgb/camera08-all-points.png'],R.name:['preprocessing-audit.json','same-target-trajectory-audit.json','motion-refit-protocol.json','head/independent-audit.json','gaussian-ablation-protocol.json']}
file=SITE/'scripts/sync-gs-lab.mjs';text=file.read_text(encoding='utf8')
for root,names in assets.items():
 assert all((L/'outputs'/root/n).is_file() for n in names)
 if root+'/${name}' in text:continue
 lines=text.splitlines(keepends=True);i=next(i for i,line in enumerate(lines) if 'dynamic-human-enerf-forearm-matte-repair-072/${name}' in line)+1
 lines.insert(i,'  ...'+json.dumps(names)+f".map(name => [`outputs/{root}/${{name}}`, '手部观测与姿态输入核查']),\n");text=''.join(lines)
file.write_text(text,encoding='utf8');shutil.copy2(__file__,H/Path(__file__).name);print('One knowledge mechanism section and bounded evidence prepared; no baseline replacement.')
