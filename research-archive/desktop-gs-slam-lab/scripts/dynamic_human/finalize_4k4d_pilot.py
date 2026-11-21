from pathlib import Path
import json,hashlib,shutil
from datetime import datetime,timezone
LAB=Path(__file__).resolve().parents[2];OUT=LAB/'outputs/dynamic-human-4k4d-official-035'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
metrics=read(OUT/'native-evaluation/metrics.json');audit=read(OUT/'native-evaluation/independent-image-audit.json');state=read(OUT/'independent-state-audit.json')
assert audit['full_pngs']==24 and state['optimizer_max_step']==1000
viewed=[f'native-evaluation/camera{c}-frame000750-full-person-comparison.png' for c in ['00','06','12','01','08','17']]
viewed += [f'native-evaluation/camera{c}-frame000750-trained.png' for c in ['00','06','12']]
for f in viewed:assert (OUT/f).is_file()
review=dict(status='single_frame_pilot_not_accepted',images=viewed,actual_reviewed_images=9,enhancement=False,
    observations=['Main human forms, shirt print and broad folds are recognizable.','Point-like black boundaries remain around hair, shoulder and shoe silhouettes.','Hands, held toys, crossed legs and overlapping regions remain soft, mixed or fragmented in allthreeheldout comparisons and sampled training-view diagnostics.',
    'Reference mattes themselves include wall residuals; these cannot be counted as clean human geometry.'],
    scope='One actor5_6time750; no temporal continuity evaluation. Image panels retain native pixels, wide displays may be downscaled by viewer. No claim of improvement over a matched GS baseline because thisframe has not had that control.')
save(OUT/'visual-review.json',review)
summary=dict(status='official4k4d_singleframe1000complete_clarity_unaccepted',completed_utc=datetime.now(timezone.utc).isoformat(),updates=1000,points=262144,
    method='Official4K4D R4DVSampler with explicit headless PyTorch3D point-renderer path and2of8sourceCNNgradienttracking; not unmodifiedpaperhardwarebackend, not anisotropic3DGS.',
    resolution=[1920,1080],training='15trainingcameraRGBandmatte inputs;train-onlyvisualhull, noauthorsurface. Released optimizedcalibration unchanged.',
    evaluation='3heldout and3training diagnostic views,8trainingtexture sources each;targetcameraexcluded from all6source sets. TargetRGB/matte removed from forwardbatch;fullnativeK/H/W, no targetcrop.',
    metrics=[dict(camera=r['camera'],split=r['split'],foreground_psnr=r['metrics']['foreground_matte_gt0_5']['psnr_db'],full_valid_psnr=r['metrics']['full_valid']['psnr_db']) for r in metrics['records']],
    audit='native-evaluation/independent-image-audit.json',visual_review='visual-review.json',state_audit='independent-state-audit.json',
    hardware_peak_pytorch_bytes=7035564544,decision='Keep1000update result as bounded failedclaritypilot. Run fixed9000newupdatecontinuation fromexactparent withsameinputs before extendingmotion.',goal_complete=False)
save(OUT/'final-summary.json',summary)
for dest,names in [(OUT/'source-snapshot',['finalize_4k4d_pilot.py']),
    (OUT/'continuation-10000/source-snapshot',['prepare_4k4d_continuation.py','train_4k4d_continuation.py','run_4k4d_continuation.sh'])]:
    dest.mkdir(exist_ok=True,parents=True)
    for n in names:shutil.copy2(LAB/'scripts/dynamic_human'/n,dest/n)
live=read(OUT/'continuation-10000/training-live.json')
mfile=LAB/'manifests/dynamic-human-work.json';m=read(mfile)
m.update(status='official4k4d_pilot_failed_clarity_longer_fit_running' if live['status']=='training' else 'official4k4d_pilot_failed_clarity_continuation_attention',updated=summary['completed_utc'],
    official4k4d_pilot='outputs/dynamic-human-4k4d-official-035/final-summary.json',
    current_runs=[dict(name='official4k4d_frame750_history10000',directory='outputs/dynamic-human-4k4d-official-035/continuation-10000',session_id=82968,status=live['status'],last_observed_new_updates=live['completed_updates_this_process'])],
    remaining_quality_issue='Official4K4D1000update singleframe also retains black point boundaries and blurred hands/toys/overlap; no accepted human or fullscene model.',
    pending=['Complete9000newupdatecontinuation;fullnative6view evaluation and independent audit, compare sameframe035withoutsharpening.','Only after sufficientsingleframefit, extend to continuouswalking/occlusion with explicit source/time separation.','Continuecompletecamera/timechecks and background/personcomposition;034candidatePnP remainsunapplied.'])
m['next_actions']=m['pending']
line='035:official4K4Druntimecompiledandnativeoneframe1000updatescompleted;6fullnativeoutputsand12panelsaudited;9imagesreviewed,clarityunaccepted. Exactparentcontinuation9000newupdateslaunched.'
if line not in m['completed']:m['completed'].append(line)
save(mfile,m)
page=LAB/'experiments/37_dynamic_human_reconstruction.md';text=page.read_text(encoding='utf8')
section=r'''

### 官方 4K4D 的原生单帧人物验证

为检查图像特征能否保留人物细节，接入 4K4D 官方 R4DVSampler，以 actor5_6 第 750 帧完成 1,000 次优化。其表示是每时刻点云、空间与时间特征、投影到源图的颜色融合及球谐残差，并非传统各向异性 3DGS。当前在 WSL 使用官方提供的 PyTorch3D 点渲染路径；每次八张源图中两张保留图像编码器梯度，因此不是论文默认硬件后端的完整复现。

初始化几何仅由 15 个训练机位的轮廓生成，没有使用作者预制表面；外观训练与源图也仅使用这 15 个机位。00、06、12 三个机位留出外观。训练保留原生像素并同步调整裁剪后的主点；额外评测恢复完整 1920×1080 内参和画面，从模型输入中移除目标 RGB 和掩膜，避免把目标轮廓用于最终画面裁剪。另三个训练诊断视角也从纹理来源中排除了自身照片。

![官方模型在原生分辨率下的真实人物对照](../outputs/dynamic-human-4k4d-official-035/native-evaluation/camera00-frame000750-full-person-comparison.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站官方模型运行 · 留出机位 00；左为作者透明度合成参照，右为真实渲染，无增强。

三路留出相机在作者前景透明度大于 0.5 的像素上，PSNR 分别为 21.57、18.84、19.44 dB；包含大面积灰底的完整有效画面为 29.10、27.49、27.83 dB。两种口径差异说明，灰色背景会抬高全图分数，人物细节必须单独观察。衣物图案和主要形状可辨，但头发、肩部和鞋边仍有颗粒状黑边，手、玩偶、交叉腿及遮挡区域仍模糊。六组人物对照和三张留出完整图均已查看，尚未通过清晰度要求；这个单帧结果也不能证明时间连续性。

全部 24 张源图／参照／渲染／alpha PNG、12 张原生比较图和各区域指标已复核；模型参数有限，优化器更新次数为 1,000。保留这一结果后，另启动从相同模型与 Adam 状态继续 9,000 次更新的固定预算验证，使用相同图像和相机。当前这些视角属于开发验证，不作为从未查看过的最终测试集。参见[完整机位 06](../outputs/dynamic-human-4k4d-official-035/native-evaluation/camera06-frame000750-trained.png)、[实际看图范围](../outputs/dynamic-human-4k4d-official-035/visual-review.json)、[模型状态核验](../outputs/dynamic-human-4k4d-official-035/independent-state-audit.json)、[原生图像核验](../outputs/dynamic-human-4k4d-official-035/native-evaluation/independent-image-audit.json)、[阶段结论](../outputs/dynamic-human-4k4d-official-035/final-summary.json)、[后续固定预算](../outputs/dynamic-human-4k4d-official-035/continuation-10000/protocol.json)。
'''
if '### 官方 4K4D 的原生单帧人物验证' not in text:page.write_text(text.rstrip()+section+'\n',encoding='utf8')
sync=Path('D:/Projects/EngineLab/engine-home/scripts/sync-gs-lab.mjs');text=sync.read_text(encoding='utf8')
if 'outputs/dynamic-human-4k4d-official-035/${name}' not in text:
    lines=text.splitlines(keepends=True);index=next(i for i,line in enumerate(lines) if 'outputs/dynamic-human-enerf-matte-background-034/${name}' in line)+1
    lines.insert(index,"  ...['final-summary.json', 'visual-review.json', 'independent-state-audit.json', 'native-evaluation/independent-image-audit.json', 'native-evaluation/camera00-frame000750-full-person-comparison.png', 'native-evaluation/camera06-frame000750-trained.png', 'continuation-10000/protocol.json'].map(name => [`outputs/dynamic-human-4k4d-official-035/${name}`, '官方4K4D原生人物单帧试验；手部与边界仍未达标']),\n")
    sync.write_text(''.join(lines),encoding='utf8')
print(json.dumps(dict(pilot_status=summary['status'],continuation_status=live['status'],new_updates=live['completed_updates_this_process'])))
