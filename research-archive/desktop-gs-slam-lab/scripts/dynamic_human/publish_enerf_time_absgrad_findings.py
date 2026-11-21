from pathlib import Path
import json,shutil
from datetime import datetime,timezone
LAB=Path(__file__).resolve().parents[2];SITE=Path('D:/Projects/EngineLab/engine-home')
OUT=LAB/'outputs/dynamic-human-enerf-absgrad-052'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,x:p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
summary=read(OUT/'paired-comparison/summary.json')
assert read(OUT/'paired-training-state-audit.json')['status'].startswith('paired6000updates_exact')
review=dict(status='six_whole_person_and17_fixed_native_detail_panels_actually_reviewed',
 whole_person_cameras=['00','06','12','01','08','17'],
 detail_regions=read(LAB/'outputs/dynamic-human-enerf-camera10-time-050/fixed-detail-regions.json')['regions'],
 findings='Some shirt print/folds change locally, but no coherent recovery of fingers or limb surface. Camera06crossed forearms remain stretched;00/12hanging arm retains wall-colored artifacts;heldobjects and airborne shoes remain smeared. Trainingviews also retain limb artifacts.',
 decision='No clear-human acceptance. The documented AbsGS threshold gives fewerGaussians and similar mean image scores here; this is not proof of geometry recovery or general method inferiority.',
 manipulation='Only native integer-coordinate crops and label padding; no enhancement or generated pixels.',goal_complete=False)
save(OUT/'visual-review.json',review)
save(OUT/'final-summary.json',dict(status='absgrad_matched6000updates_similar_image_scores_limb_clarity_unaccepted',
 baseline_points=143073,candidate_points=79659,point_reduction_percent=(1-79659/143073)*100,
 heldout_foreground_delta_db={r['camera']:r['foreground_delta_db'] for r in summary['records'] if r['split']=='heldout'},
 intervention='Absolute screen-space gradient accumulation with documented grow threshold0.0008 versus signed gradient threshold0.0002; fixed all input/init/camera/background draws and6000updates.',
 reference='https://arxiv.org/abs/2404.10484',implementation='Installed gsplat1.3DefaultStrategy; not complete AbsGS author benchmark.',
 state_audit='paired-training-state-audit.json',image_audit='native-evaluation/independent-image-audit.json',visual_review='visual-review.json',
 conclusion=review['decision'],goal_complete=False))
save(OUT/'author-geometry-availability.json',dict(status='provided_geometry_not_available_at_current_frame750',
 directory='D:/GS-SLAM-Data/enerf_outdoor/extracted/actor5_6/surfs',files=200,first='000000.ply',last='000199.ply',
 repo_config='configs/datasets/enerf_outdoor/actor5_6.yaml frame_sample=[0,200,1]',
 decision='Do not substitute another frame or assume a factor-six index mapping. Provided geometry has unknown view provenance and cannot silently enter strict heldout protocol.',
 used_in_current_training=False,goal_complete=False))
save(OUT/'execution-recovery.json',dict(training_completed=True,retrained_after_evaluation_failure=False,
 evaluation_failure='input-audit.json was missing in setup, copied byte-identical baseline audit; evaluation then completed without changing checkpoint.',
 audit_collision='Evaluator uses independent-state-audit.json; paired training audit now has distinct paired-training-state-audit.json, recomputed after evaluation.',
 original_setup_snapshot_preserved=True))
shutil.copy2(LAB/'scripts/dynamic_human/prepare_enerf_absgrad_control.py',OUT/'source-snapshot/prepare_enerf_absgrad_control-corrected.py')

p=LAB/'knowledge/32_dynamic_human_scene_reconstruction.md';text=p.read_text(encoding='utf8')
addition=r'''### 增密梯度与细节恢复

![原照片、标准梯度增密与绝对梯度增密的完整人物对照](../outputs/dynamic-human-enerf-absgrad-052/paired-comparison/camera06-frame000750-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站运行 · 机位 06 · 原生像素，无增强。

三列依次是原照片、标准三维高斯增密、采用 AbsGS 绝对梯度统计的增密。两种训练中的衣服主体都比早期点式结果清楚，但交叉手臂仍然被拉长，手指没有形成正确边界。这里改变了增密决策，未改变相机、图像、掩码与训练预算；不能将增密方式的改变直接等同于人体表面恢复。

增密依赖图像损失对高斯投影中心的梯度。记第 $i$ 个高斯的二维中心为 $\boldsymbol\mu_i^{\mathrm{2D}}$，像素 $p$ 对该中心的反向传播贡献为 $\mathbf g_{ip}$。标准梯度首先求和；若不同像素要求中心向相反方向移动，贡献可能抵消：

$$
\mathbf g_i=\sum_{p\in\mathcal P_i}\mathbf g_{ip},
\qquad s_i=\left\|\mathbf g_i\right\|_2.
$$

其中 $\mathcal P_i$ 是该高斯参与渲染的像素集合。AbsGS 的核心是先对每个贡献的两个坐标分量分别取绝对值，再累积：

$$
\mathbf a_i=
\begin{bmatrix}
\displaystyle\sum_{p\in\mathcal P_i}|g_{ip,x}|\\[3pt]
\displaystyle\sum_{p\in\mathcal P_i}|g_{ip,y}|
\end{bmatrix},
\qquad s_i^{\mathrm{abs}}=\left\|\mathbf a_i\right\|_2.
$$

这种统计能保留相反方向的细节需求。实现中还会按屏幕坐标尺度归一化，并在多次可见观测间累计，超过阈值的高斯再依据空间尺寸复制或分裂。阈值随损失归一化与梯度统计方式变化，不能把两种统计的相同数值视为相同的增密强度。本地采用 gsplat 文档的示例设置，标准统计阈值为 $2\times10^{-4}$，绝对梯度阈值为 $8\times10^{-4}$；这组设置最终产生了更少的高斯，不能笼统理解成“绝对梯度总会增加点数”。

增密解决的是表示分辨率的分配。相机观测之间的运动不一致、掩码中的背景残留、错误深度与遮挡关系，仍需要各自的约束。评价时应将原照片与完整人物、新视角、手臂和提物区域一起查看，不能只凭衣服花纹更锐利作出结论。方法来源见 [AbsGS](https://arxiv.org/abs/2404.10484)，本地条件与真实效果见[增密对照](../experiments/37_dynamic_human_reconstruction.md#绝对梯度增密与移动肢体)。

'''
anchor='### 同一帧号不一定对应同一运动状态'
if '### 增密梯度与细节恢复' not in text:
 assert text.count(anchor)==1;text=text.replace(anchor,addition+anchor)
p.write_text(text,encoding='utf8')

p=LAB/'experiments/37_dynamic_human_reconstruction.md';text=p.read_text(encoding='utf8')
text=text.replace('下一项是对机位 10 的局部候选做固定初始化、固定预算的训练观测对照。','机位 10 的局部成对训练已经完成，结果见下文。')
addition='''

### 单机位时间偏移与联合相机图

只将训练机位 10 的第 750 帧换成第 749 帧，使用相同的 SAM2 提示和掩码处理规则；其余 14 个训练输入、原始标定、留出照片均保持不变。两组从完全相同的高斯参数出发，固定全部 6,000 次目标机位与背景颜色抽样，分别完成 6,000 次更新。三个留出机位 00、06、12 的前景 PSNR 变化为 −0.373、+0.085、+1.003 dB。

![原照片、同帧训练与仅机位 10 前移一帧的留出效果](../outputs/dynamic-human-enerf-camera10-time-050/paired-comparison/camera12-frame000750-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站运行 · 机位 12 · 无增强。

六张完整人物对照及预先确定的 17 个人体细节区域均已查看。机位 12 的衣服、部分前臂和提物轮廓改善，但垂下的手臂仍有背景残留与拉伸；机位 00 的手和抬起的鞋变差，机位 06 的交叉手臂未恢复。因此不把单机位改帧作为清晰重建方案。见[结论](../outputs/dynamic-human-enerf-camera10-time-050/final-summary.json)、[成对状态核验](../outputs/dynamic-human-enerf-camera10-time-050/independent-state-audit.json)、[完整与细节指标](../outputs/dynamic-human-enerf-camera10-time-050/paired-comparison/summary.json)及[逐图审阅](../outputs/dynamic-human-enerf-camera10-time-050/visual-review.json)。

进一步只使用 740、750 帧的固定轨迹拟合 15 个训练相机的一致相对帧索引，限制偏移为 −2 至 +2 帧，机位 01 固定为零；29 条有效相机边形成连通图。联合解建议机位 02、05、10 分别取 −2、−1、−1，其余不变。760 帧的 28 条有效复核边平均误差由 0.651 降至 0.579 像素，但新增 02、05 偏移涉及的四条有效边中仅一条改善、三条变差。因此未开展全局改帧训练，也未修改原始数据。这只是图像索引假设，不是硬件时钟标定。见[联合解与复核](../outputs/dynamic-human-enerf-time-graph-051/final-summary.json)和[固定轨迹重算](../outputs/dynamic-human-enerf-time-graph-051/independent-audit.json)。

### 绝对梯度增密与移动肢体

在同帧基线的原生输入、掩码、初始化、6,000 次抽样与更新预算下，切换到 gsplat 1.3 文档提供的 AbsGS 增密选项：启用绝对梯度，并将对应增密阈值设为 0.0008，基线阈值为 0.0002。这是本地增密策略对照，不是 AbsGS 原论文完整基准复现。实际 CUDA 绝对梯度缓冲区已确认存在、有限且非零。

![原照片、标准梯度与绝对梯度增密的完整留出人物](../outputs/dynamic-human-enerf-absgrad-052/paired-comparison/camera06-frame000750-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站运行 · 机位 06。

最终高斯数量由 143,073 降为 79,659，减少约 44.3%；三个留出机位的前景 PSNR 变化仅为 −0.036、−0.003、+0.020 dB。六张完整人物对照和相同的 17 个细节区域均已查看，交叉手臂、手指、玩具及抬起的鞋仍有拉丝或模糊。这说明当前设置的模型更小，并未证明人体更清晰。见[结果](../outputs/dynamic-human-enerf-absgrad-052/final-summary.json)、[训练状态核验](../outputs/dynamic-human-enerf-absgrad-052/paired-training-state-audit.json)、[原生图像核验](../outputs/dynamic-human-enerf-absgrad-052/native-evaluation/independent-image-audit.json)、[逐图审阅](../outputs/dynamic-human-enerf-absgrad-052/visual-review.json)及[固定区域指标](../outputs/dynamic-human-enerf-absgrad-052/paired-comparison/summary.json)。

下载包附带的作者表面点云只覆盖文件编号 000000～000199，当前行走对照使用第 750 帧；没有证据支持把点云编号乘以六对应到原图，因此未将它接入当前模型。下一步检查训练相机中的前景残差、表面深度和轮廓约束，再选择几何监督方法。当前 GPU 可完成这些有界对照，没有新增硬件或数据下载阻塞。
'''
if '### 单机位时间偏移与联合相机图' not in text:text+=addition
p.write_text(text,encoding='utf8')
sync=SITE/'scripts/sync-gs-lab.mjs';text=sync.read_text(encoding='utf8')
if 'dynamic-human-enerf-absgrad-052/${name}' not in text:
 lines=text.splitlines(keepends=True);idx=next(i for i,l in enumerate(lines) if 'dynamic-human-enerf-representation-048/${name}' in l)+1
 extra="  ...['final-summary.json','independent-state-audit.json','visual-review.json','paired-comparison/summary.json','paired-comparison/camera12-frame000750-full-person.png'].map(name => [`outputs/dynamic-human-enerf-camera10-time-050/${name}`, '单机位时间观测对照']),\n"
 extra+="  ...['final-summary.json','independent-audit.json'].map(name => [`outputs/dynamic-human-enerf-time-graph-051/${name}`, '训练相机相对帧索引图']),\n"
 extra+="  ...['final-summary.json','paired-training-state-audit.json','native-evaluation/independent-image-audit.json','visual-review.json','paired-comparison/summary.json','paired-comparison/camera06-frame000750-full-person.png'].map(name => [`outputs/dynamic-human-enerf-absgrad-052/${name}`, '绝对梯度增密与人物细节']),\n"
 lines.insert(idx,extra);sync.write_text(''.join(lines),encoding='utf8')
mp=LAB/'manifests/dynamic-human-work.json';m=read(mp)
m['status']='time_and_absgrad_controls_complete_limb_geometry_unaccepted';m['current_runs']=[]
for item in ['051globaltraining-camera frame-index graph solved;87trackarrays recomputed; additional02/05 shifts have3of4coveredcheckedgesworse, globalretiming not adopted.',
 '052actual6000update AbsGS option compared with050baseline;24nativePNGs12panels6wholeperson17details verified/reviewed.44.3percentfewerGaussians, no clear-limb recovery.']:
 if item not in m['completed']:m['completed'].append(item)
m['outdoor_global_time_graph']='outputs/dynamic-human-enerf-time-graph-051/final-summary.json'
m['outdoor_absgrad_control']='outputs/dynamic-human-enerf-absgrad-052/final-summary.json'
m['next_actions']=['Decompose training-view limb residuals into foreground-mask support, depth/surface and appearance; choose train-only geometric supervision from that diagnosis.',
 'Do not assume author surfs000000-000199 correspond to current750; no undocumented frame mapping or silent heldout geometry use.',
 'Extend time continuity and full-scene composition only after visible limb surfaces improve.']
m['goal_turn_classification']='Progress:050-052 actual matched training, joint graph diagnosis, six-view native renders and fixed detail reviews; clarity unmet, no external blocker.'
m['external_input_blocker']=None;m['last_recorded_utc']=datetime.now(timezone.utc).isoformat();save(mp,m)
shutil.copy2(__file__,OUT/'source-snapshot'/Path(__file__).name)
print('050-052knowledge/evidence/allowlist/manifest updated, goal remains active')
