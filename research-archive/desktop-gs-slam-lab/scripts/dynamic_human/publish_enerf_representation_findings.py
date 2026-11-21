from pathlib import Path
import json,hashlib,shutil
from datetime import datetime,timezone
LAB=Path(__file__).resolve().parents[2];SITE=Path('D:/Projects/EngineLab/engine-home')
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
time=LAB/'outputs/dynamic-human-enerf-actor56-time-049';a=read(time/'independent-audit.json')
save(time/'visual-review.json',dict(status='four_native_triplet_sheets_24_fixed_matches_reviewed',
 files=['identity-01-10-t760.png','identity-10-11-t760.png','identity-04-09-t760.png','identity-05-07-t760.png'],
 selection='Fastest source tracks separated by40px, not ranked by residual improvement.',
 finding='Sampled patches largely preserve toy, shoe, printed shirt, sleeve and head identity. Perspective and motion change patches; center match identity alone cannot establish a physical clock.',
 decision='No global retiming adopted. Camera10 relative to01/11 is a candidate for a bounded training-observation control; several other candidate shifts fail independent760check.',goal_complete=False))
save(time/'final-summary.json',dict(status='local_relative_time_candidates_mixed_validation_no_global_retiming',
 track_files=a['track_files'],matched_tracks=a['matched_tracks'],candidate_pairs=a['candidate_pairs'],zero_lag_candidates=a['zero_lag_candidates'],
 nonzero_candidates=a['nonzero_candidates'],nonzero_with_check_coverage=a['nonzero_with_check_coverage'],nonzero_improve_check=a['nonzero_improve_check'],nonzero_worsen_check=a['nonzero_worsen_check'],
 fit_frames=[740,750],check_frame=760,heldout_appearance_opened=False,
 conclusion='22of39pairs prefer zero lag;17prefer nonzero. Of14nonzero with>=12movingchecktracks,9improve and5worsen. No blanket timing failure or global frame shift established.',
 next='Isolate camera10 candidate lag in a matched training observation control before changing the full dataset. Keep all heldout750photos unchanged.',goal_complete=False))

knowledge=LAB/'knowledge/32_dynamic_human_scene_reconstruction.md';text=knowledge.read_text(encoding='utf8')
text=text.replace('当前对照只证明部分边缘颜色受到无效源观测影响，尚未证明手指和衣物细节已恢复。',
 '固定模型检查与同起点的成对训练均只改善了部分边缘颜色，尚未恢复手指和衣物细节。')
addition=r'''### 纹理清楚与运动表面正确需要分别检验

![原照片、点式图像驱动渲染与直接优化三维高斯的同一机位对照](../outputs/dynamic-human-enerf-representation-048/paired-comparison/camera06-frame000750-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站运行 · 机位 06 · 原生像素，无图像增强。

三列依次为原照片、基于 4K4D 点表示的本地单帧分支、直接优化各向异性三维高斯的单帧分支。右列的衣服褶皱、头发与腿部轮廓更清楚，但手臂出现拉长，交叠人物的肩部仍有错误结构。因此，纹理更锐利不能替代对肢体位置与遮挡关系的检查。这是两套完整训练流程的比较，不能把差异单独归因于高斯形状。

设相机 $s$ 的内参和世界到相机变换分别为 $\mathbf K_s$、$\mathbf T_s$，场景参数为 $\Theta$，渲染算子为 $\mathcal R$，观测为 $\mathbf I_s(t_s)$。单帧多视角优化可以写成

$$
\Theta^*=\underset{\Theta}{\operatorname{arg\,min}}\;
\sum_{s\in\mathcal S_{\mathrm{train}}}
\mathcal L\!\left(\mathcal R(\Theta;\mathbf K_s,\mathbf T_s),\mathbf I_s(t_s)\right).
$$

这个目标隐含了共同场景状态的要求。即使图像文件的编号相同，不同观测也可能包含不同运动状态；标定误差、遮挡、分割缺失和错误表面同样可能使观测难以由一套几何共同解释。模型可能用伸长的高斯、厚表面或混合颜色降低训练误差。仅凭渲染中的重影，不能确定其中哪一种原因成立。

因此，需要同时查看原生分辨率下的训练视角与未参与训练的相机：前者检验是否学到纹理，后者检验这种纹理能否依附在合理的空间表面上。对运动一致性的判断还应跟踪同一纹理点，并在另一时刻复核相机之间的对应，不能为每个候选时间偏移重新挑选一组更容易匹配的点。这里的单帧比较尚不代表连续动态人物或大场景联合重建已经完成。

'''
anchor='### 同一帧号不一定对应同一运动状态'
if '### 纹理清楚与运动表面正确需要分别检验' not in text:
 assert text.count(anchor)==1;text=text.replace(anchor,addition+anchor)
knowledge.write_text(text,encoding='utf8')

page=LAB/'experiments/37_dynamic_human_reconstruction.md';text=page.read_text(encoding='utf8')
addition2='''

### 行走人物：训练、覆盖范围与表示对照

前景支持筛选的成对训练已经完成。两组从同一个 2,000 次更新模型及 Adam 状态出发，各增加 1,000 次更新；记录的目标与源机位序列一致。筛选组在六个视角均提高前景 PSNR，三个留出机位 00、06、12 分别提高 0.324、0.248、0.234 dB。六张原生人物对照与三张完整留出渲染均已查看：肩部部分黑边减轻，手、衣服图案和遮挡仍不清楚。见[成对训练结论](../outputs/dynamic-human-4k4d-source-gate-training-044/final-summary.json)、[模型状态核验](../outputs/dynamic-human-4k4d-source-gate-training-044/independent-state-audit.json)。

原始半径投影到图像后，参与渲染点的半径中位数约为 2.57～2.97 像素。保持点位置和颜色不变，将半径乘以 0.75 或 0.5，六个视角的前景指标均下降；头发、衣服和皮肤出现更多孔洞，手部没有恢复。见[覆盖范围对照](../outputs/dynamic-human-4k4d-point-footprints-046/final-summary.json)与[图像、半径公式核验](../outputs/dynamic-human-4k4d-point-footprints-046/independent-audit.json)。随后用当前模型的源相机深度排除遮挡取色，也在六个视角都变差；由于深度来自当前不完善的几何，此结果不能否定准确可见性监督的价值。见[可见性检查](../outputs/dynamic-human-4k4d-visibility-047/final-summary.json)。

为检查另一种表示，采用完全相同的 15 个训练机位、原生 RGB、候选掩码和标定，训练直接优化各向异性三维高斯的独立分支，包含自适应分裂与剪枝。固定 6,000 次更新后共有 142,735 个高斯。它改变了初始化、表示、损失与训练预算，是方法替代对照，不是单因素消融，也不是官方 4K4D 复现。

![原照片、4K4D 点式分支与直接优化的 3DGS：留出机位 06](../outputs/dynamic-human-enerf-representation-048/paired-comparison/camera06-frame000750-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站运行 · 保留手臂拉长与遮挡错误。

六个视角均已检查。三个训练诊断视角的前景 PSNR 提高约 1.50～3.59 dB，但留出机位 00、06、12 分别变化 −1.466、−0.066、+0.952 dB。头发、衣服褶皱和大面积表面变清楚，移动手臂、手指、提着的玩具和抬起的鞋仍有拉长或重影。因此该分支仍未通过清晰人物验收。见[训练协议](../outputs/dynamic-human-enerf-representation-048/protocol.json)、[结果与局限](../outputs/dynamic-human-enerf-representation-048/final-summary.json)、[24 张原生图像核验](../outputs/dynamic-human-enerf-representation-048/native-evaluation/independent-image-audit.json)及[六组图像审阅](../outputs/dynamic-human-enerf-representation-048/visual-review.json)。

### 行走片段的跨相机运动一致性

同帧输入检查采用 25 组相邻训练相机，共 2,879 个前景互匹配点，不用对极误差筛选匹配；其中 17 组的 Sampson 误差中位数小于 1 像素。原图检查同时发现少量错误匹配，以及某些移动鞋部的数像素残差。因此不能笼统归咎于全部数据不同步，见[同帧核验](../outputs/dynamic-human-enerf-foreground-geometry-045/final-summary.json)。

进一步固定同一组对应点，跟踪前后 3 帧，以 740、750 帧选候选偏移、760 帧复核。共保存并重算 117 个轨迹数组、5,594 条对应。39 组相机中，22 组偏好零偏移，17 组偏好非零偏移；非零候选中有 14 组在复核帧达到至少 12 个运动点，其中 9 组改善、5 组变差。四组原生三列图的 24 个对应已查看，未据此改写数据时间轴。下一项是对机位 10 的局部候选做固定初始化、固定预算的训练观测对照。见[轨迹核验](../outputs/dynamic-human-enerf-actor56-time-049/independent-audit.json)与[当前结论](../outputs/dynamic-human-enerf-actor56-time-049/final-summary.json)。
'''
if '### 行走人物：训练、覆盖范围与表示对照' not in text:page.write_text(text+addition2,encoding='utf8')
sync=SITE/'scripts/sync-gs-lab.mjs';text=sync.read_text(encoding='utf8')
if 'dynamic-human-enerf-representation-048/${name}' not in text:
 lines=text.splitlines(keepends=True);idx=next(i for i,l in enumerate(lines) if "source-gate-training-044/protocol.json" in l)+1
 added="  ...['final-summary.json', 'independent-state-audit.json'].map(name => [`outputs/dynamic-human-4k4d-source-gate-training-044/${name}`, '同起点的前景支持成对训练']),\n"
 for slug in ['dynamic-human-enerf-foreground-geometry-045','dynamic-human-4k4d-point-footprints-046','dynamic-human-4k4d-visibility-047','dynamic-human-enerf-actor56-time-049']:
  added+=f"  ...['final-summary.json', 'independent-audit.json'].map(name => [`outputs/{slug}/${{name}}`, '原生人物与几何控制检查']),\n"
 added+="  ...['protocol.json', 'final-summary.json', 'visual-review.json', 'native-evaluation/independent-image-audit.json', 'paired-comparison/camera06-frame000750-full-person.png'].map(name => [`outputs/dynamic-human-enerf-representation-048/${name}`, '直接优化各向异性3DGS的人物对照']),\n"
 lines.insert(idx,added);sync.write_text(''.join(lines),encoding='utf8')

mp=LAB/'manifests/dynamic-human-work.json';m=read(mp)
m['status']='single_frame_representation_controls_complete_limb_clarity_unaccepted'
m['current_runs']=[]
for entry in ['044 both1000additionalupdates/3000history;48nativePNGs24panels and6pairedpanels verified,9images reviewed. Supportgate modest edge improvement only.',
 '04525train-camera pairs2879foregroundmatches checked,6nativepatchsheets reviewed;17pairmedians<1px, local dynamic residuals/outliers remain.',
 '046 radius075/050 controls:18nativePNGs6panels and radii checked;3heldout panels show holes, reject shrink-only fix.',
 '047 sourcevisibility controls:15modeldepthmaps,18nativePNGs6panels checked;3heldout panels reviewed, bothrules worse all6views.',
 '048 directanisotropic3DGS6000updates142735Gaussians;6nativeviews24PNGs12panels verified,6pairedpanels reviewed; broadsurfaces sharper but movinglimbs/heldoutquality unresolved.',
 '049 fixed-track117arrays5594matches checked;4native identity sheets24matches reviewed. Nonzero lag candidates mixed760validation; no globalretiming adopted.']:
 if entry not in m['completed']:m['completed'].append(entry)
m['next_actions']=['Build a bounded camera10 relative-time training-observation control from049; same048initialization,6000budget and unchanged heldout750reference. Recompute candidate mask at selected source time, preserve original.',
 'Judge hands, sleeves, toy and airborne shoe in real native heldout renders; keep direct3DGS and4K4Dresults separate.',
 'Only after single-frame cross-view clarity improves, extend time continuity and static-scene integration.']
m['external_input_blocker']=None;m['goal_turn_classification']='Progress:044-049 executed, audited and visually reviewed; no hardware/input blocker, clear moving-human goal remains unmet.'
m['last_recorded_utc']=datetime.now(timezone.utc).isoformat()
for name,slug in [('official4k4d_source_gate_training','dynamic-human-4k4d-source-gate-training-044'),('outdoor_foreground_geometry','dynamic-human-enerf-foreground-geometry-045'),('official4k4d_footprint_control','dynamic-human-4k4d-point-footprints-046'),('official4k4d_visibility_control','dynamic-human-4k4d-visibility-047'),('outdoor_direct3dgs_actor56','dynamic-human-enerf-representation-048'),('outdoor_actor56_timing','dynamic-human-enerf-actor56-time-049')]:m[name]=f'outputs/{slug}/final-summary.json'
save(mp,m)
print('Knowledge, evidence, resource allowlist and active-work manifest updated through049')
