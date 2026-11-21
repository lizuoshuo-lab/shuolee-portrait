from pathlib import Path
import json,shutil
from datetime import datetime,timezone
LAB=Path(__file__).resolve().parents[2];OUT=LAB/'outputs/dynamic-human-enerf-surface-long-056';SITE=Path('D:/Projects/EngineLab/engine-home')
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
r=read(OUT/'paired-comparison/summary.json');a=read(OUT/'training-state-audit.json');assert a['step']==30000
assert read(OUT/'native-evaluation/independent-image-audit.json')['full_pngs']==24
save(OUT/'visual-review.json',dict(status='6wholeperson_and17fixed_native_details_actually_reviewed',
 cameras=['00','06','12','01','08','17'],detail_regions=read(LAB/'outputs/dynamic-human-enerf-camera10-time-050/fixed-detail-regions.json')['regions'],
 findings='Training01/08/17 shirt, hands andtoy details fitbetter; heldout00/06/12 show morefine streaking/floaters atarms, clothingedges andtoys. Allthree FGmetrics regress.00darkforearmstillstretched,06crossedforearmswrong,12shirtpartialimprovement doesnotrepairhand/footgeometry.',
 decision='Do not promote30000schedule as clarity improvement; stop adding iterations/capacity onthis same representation asdefaultnextstep. Improve reliable articulated correspondence and bodyalignment.',goal_complete=False))
save(OUT/'final-summary.json',dict(status='30000updates_training_fit_improves_development_views_regress_not_adopted',
 updates=30000,points=a['points'],checkpoint_sha256=a['checkpoint_sha256'],
 heldout_foreground_delta_db={x['camera']:x['foreground_delta_db'] for x in r['records'] if x['split']=='heldout'},
 training_foreground_delta_db={x['camera']:x['foreground_delta_db'] for x in r['records'] if x['split']=='training_diagnostic'},
 intervention='Longeroptimization plus LR/density schedules changed together. Same original15native traininginputs and initialparameters; first6000draws equal055.',
 recovery='cap-recovery.json;14901checkpoint preserved,805000guardallows528pointovershoot. No optimizer/RNG resets.',
 scope='Single-frame750 foreground, development-camera comparison; not complete dynamic/full-scene reconstruction.',goal_complete=False))
p=LAB/'experiments/37_dynamic_human_reconstruction.md';text=p.read_text(encoding='utf8')
addition='''

### 更长训练与持续增密

在相同原始图像、掩码、相机和初始面元上，重新训练有表面约束的模型至 30,000 次更新；同时把增密延长到第 15,000 次附近，并按更长训练预算衰减位置学习率。这是训练计划整体调整，不能解释为只改变迭代次数。第 14,901 次更新后面元数量达到 800,528，超过预设资源上限 528 个，保存完整检查点后将保护上限调为 805,000 并恢复；参数、优化器、随机状态和损失均未重置，后续不再增密。峰值训练分配显存约 2.3 GB，未发生显存不足。

最终三个训练诊断机位 01、08、17 的前景 PSNR 相对 6,000 次有约束版本提高 2.677、2.234、3.858 dB；但未参与训练的开发机位 00、06、12 分别下降 0.749、0.122、0.450 dB。24 张原生输出和 12 张原生比较图通过独立像素与指标核验；六张完整人物对照与同一组 17 个预定细节区域已查看。

![原照片、六千次表面训练与三万次训练的原生人物对照](../outputs/dynamic-human-enerf-surface-long-056/paired-comparison/camera06-frame000750-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站运行 · 机位 06 · 无增强。

训练照片上的衣服、手和玩具更接近观测，但留出视角的前臂、衣服边缘与玩具出现更多细碎条纹和漂浮结构；交叉手臂的位置错误仍在。该结果不作为清晰度提升版本，也不继续把扩大点数和迭代次数作为默认方向。下一步应改善可靠的人体对应与几何约束，而非把训练视角拟合得更好当成完成重建。

见[结果汇总](../outputs/dynamic-human-enerf-surface-long-056/final-summary.json)、[事前计划](../outputs/dynamic-human-enerf-surface-long-056/protocol.json)、[中断与恢复](../outputs/dynamic-human-enerf-surface-long-056/cap-recovery.json)、[最终训练核验](../outputs/dynamic-human-enerf-surface-long-056/training-state-audit.json)、[原生图像重算](../outputs/dynamic-human-enerf-surface-long-056/native-evaluation/independent-image-audit.json)和[实际图像审阅](../outputs/dynamic-human-enerf-surface-long-056/visual-review.json)。
'''
if '### 更长训练与持续增密' not in text:
 anchor='### 训练照片中的人体关节与粗略人体先验';assert anchor in text;text=text.replace(anchor,addition+'\n'+anchor);p.write_text(text,encoding='utf8')
p=LAB/'knowledge/32_dynamic_human_scene_reconstruction.md';text=p.read_text(encoding='utf8')
addition='\n训练拟合与跨视角恢复也可能分离。同一批观测上的更长训练与更大面元数量，使训练照片的拟合明显改善，三个开发留出机位却都变差，手臂仍有条纹。这类现象提示当前模型仍在用视角相关的外观和局部几何解释不一致观测；它不能确定唯一误差来源，但已不支持继续单纯增加训练预算。见[更长训练对照](../experiments/37_dynamic_human_reconstruction.md#更长训练与持续增密)。\n\n'
if '三个开发留出机位却都变差' not in text:
 anchor='### 掩码中的未知区域与几何观测覆盖';text=text.replace(anchor,addition+anchor);p.write_text(text,encoding='utf8')
p=SITE/'scripts/sync-gs-lab.mjs';text=p.read_text(encoding='utf8')
if 'dynamic-human-enerf-surface-long-056/${name}' not in text:
 lines=text.splitlines(keepends=True);idx=next(i for i,l in enumerate(lines) if 'dynamic-human-enerf-surface-gs-055/${name}' in l)+1
 lines.insert(idx,"  ...['final-summary.json','protocol.json','cap-recovery.json','training-state-audit.json','native-evaluation/independent-image-audit.json','visual-review.json','paired-comparison/camera06-frame000750-full-person.png'].map(name => [`outputs/dynamic-human-enerf-surface-long-056/${name}`, '二维高斯更长训练对照']),\n");p.write_text(''.join(lines),encoding='utf8')
p=LAB/'manifests/dynamic-human-work.json';m=read(p)
m['outdoor_surface_long_training']='outputs/dynamic-human-enerf-surface-long-056/final-summary.json'
m['outdoor_body_observations']='outputs/dynamic-human-enerf-body-observations-057/final-summary.json'
m['outdoor_body_prior']='outputs/dynamic-human-enerf-body-prior-058/final-summary.json'
m['current_runs']=[];m['external_input_blocker']=None
m['status']='surface_and_long_training_reviewed_coarse_body_prior_alignment_requires_refinement'
m['goal_turn_classification']='Progress:055pairedsurfacecontrol plus056actual30000training,057train-onlybodyobservations and058SMPLfit completed/audited. Humanclarity remainsunaccepted; noexternalblocker.'
m['next_actions']=['Improve body observations with per-person/high-resolution pose inference, including occludedelbow/head/hand/foot orientation; preserve uncertainty and training-only separation.',
 'Refine soft articulatedbody alignment before testing body-conditioned Gaussian appearance; do not force inaccurate058mesh onto clothing/hands/toys.',
 'After a credible single-frame body representation, extend to continuous walking and reservedtime/camera validation, then full-scene integration.']
for s in ['05630000updates800528points completed with preserved14901caprecovery; trainviewsimprove butall3developmentheldoutFGscoresregress. No clarity promotion.',
 '05715training bodydetections and23triangulatedjoint hypotheses verified;0582000updateSMPLprior fitted,5projectionviews show head/elbow/hand/foot errors. No hardconstraint adoption.']:
 if s not in m['completed']:m['completed'].append(s)
m['last_recorded_utc']=datetime.now(timezone.utc).isoformat();save(p,m)
shutil.copy2(__file__,OUT/'source-snapshot'/Path(__file__).name)
print('056 final findings recorded, all current jobs ended, continuation targets saved')
