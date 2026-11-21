"""Record completed RGB-pose control and publish bounded, visually reviewed results."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

L = Path(__file__).resolve().parents[2]
R = L / 'outputs/dynamic-human-enerf-rgb-pose-correction-074'
S = Path('D:/Projects/EngineLab/engine-home')
read = lambda p: json.loads(p.read_text(encoding='utf8'))
def save(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

known = read(R/'paired-comparison/summary.json')
held = read(R/'heldout/paired-comparison/summary.json')
kt = read(R/'temporal-comparison/summary.json')
ht = read(R/'heldout/temporal-comparison/summary.json')
assert read(R/'residual_graph/native-evaluation/independent-image-audit.json')['images'] == 102
assert read(R/'heldout/native-evaluation/independent-image-audit.json')['images'] == 18
assert len(kt['transitions']) == 75 and len(ht['transitions']) == 15
assert read(R/'training-protocol-audit.json')['updates'] == 15000
files = list(read(R/'visual-review-plan.json')['files'])
assert len(files) == 37
for prefix, result in [('',known), ('heldout/',held)]:
    w = result['worst_change']
    name = f"{prefix}paired-comparison/camera{w['camera']}-frame{w['frame']:06d}-full-person.png"
    if name not in files:
        files.append(name)
assert len(files) == 38 and all((R/p).is_file() for p in files)

review = dict(
    status='38_fixed_and_worst_case_comparison_images_actually_reviewed',
    reviewed_files=[dict(path=p, sha256=sha(R/p)) for p in files],
    review_method='Actual image inspection, original photograph / baseline / corrected RGB pose. Native pixel crops were not sharpened or enhanced. Full-person display may be downscaled by the viewing tool; native detail crops inspected separately.',
    findings_zh=[
        '固定已知机位完整人物12张与8张细节对照中，两版整体外观相近。头部、前臂和鞋部没有一致的可见细节改善，指缘和鞋带仍偏软。',
        '另外三个机位的9张固定全身图中，衣服轮廓基本保留。机位06第745帧黑衣人物胸前的浅色条纹仍在，第755帧交叉手臂和手部仍有明显重影。',
        '另外机位的8张前臂区域对照没有显示稳定改善。区域名称来自固定投影框，遇遮挡时会包含另一人物，不是手臂可见性真值或独立分割。',
        '已知机位最差退步为08第755帧，另三个机位最差为06第753帧；两张均已查看。胸前图案、遮挡和鞋部仍未达到清晰重建要求。'
    ],
    videos='Seven videos mechanically decoded during numeric audit. This record does not claim actual video playback review.',
    decision_zh='保留正确RGB姿态预处理；本轮渲染模型不替换067开发基线。单次训练没有显示稳定清晰度收益。',
    adopt_as_development_baseline=False, goal_complete=False)
save(R/'visual-review.json', review)
summary = read(R/'summary.json')
summary.update(
    status='RGB_pose_control_complete_clarity_not_accepted',
    native_review_complete=True,
    native_images=dict(known=102,other_cameras=18),
    actual_comparison_images_reviewed=38,
    known_validation=known['validation_foreground'],
    other_camera_validation=held['validation_foreground'],
    temporal_difference_residual=dict(known=kt['mean_temporal_residual'],other_cameras=ht['mean_temporal_residual']),
    checkpoint_sha256=sha(R/'residual_graph/resume.pt'),
    decision=review['decision_zh'], adopt_as_development_baseline=False,
    RGB_preprocessing_fix_retained=True,
    conditions='Same original105images, masks, initialization, graph384nodes, optimizer, schedule, density settings and one seed73; only frozen body motion changed after corrected pose inference and clock/body/head refit. Old support and canonical shape retained. Reused development cameras and times, not independent final test.',
    remaining='Fingers, airborne shoes and occlusion remain soft or ghosted; static scene integration remains unfinished.',
    goal_complete=False)
summary['gaussian_training']['status']='15000_updates_and120_native_renders_complete'
save(R/'summary.json', summary)
save(R/'final-summary.json', summary)

knowledge=L/'knowledge/32_dynamic_human_scene_reconstruction.md'
text=knowledge.read_text(encoding='utf8')
old='新增手部自由度之前，应先检查可见指尖、跨机位对应和跨时刻连续性，再用真实渲染验证其作用。'
new=old+' 本地修复姿态网络的RGB输入后，同预算重新拟合运动并训练，仍未消除指缘模糊与人物交叉处的重影。输入正确是必要条件；局部手指运动、表面对应和遮挡关系仍须分别检验，不能由关键点分数或一次输入修复代替。'
assert old in text
if new not in text:
    text=text.replace(old,new,1)
    knowledge.write_text(text,encoding='utf8')

evidence=f'''

### 正确颜色输入后的完整重建对照

修正姿态检测的颜色通道后，身体、头部运动与相机时间偏移已重新拟合，再以相同105张原始训练照片、初始化、384节点变形图、抽样顺序和15000次更新训练高斯。掩码、损失权重和增密设置保持一致，没有加入局部手部候选。本轮得到228414个高斯，完成102张原机位、18张另外机位的1920×1080原生渲染；保存的预测与原图裁剪经过数值复算，按预先固定范围及最差退步图实际查看38张对照。

![人物交叉处的原照片、基线和修正姿态后的重建](../outputs/{R.name}/heldout/paired-comparison/camera06-frame000745-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本地实际渲染 · 机位06、第745帧 · 原生像素，未锐化。

原训练机位的90张留出时间图，平均前景PSNR从{known['validation_foreground']['before_mean_psnr_db']:.3f}变为{known['validation_foreground']['after_mean_psnr_db']:.3f}dB，38张改善；另外三个机位的18张图从{held['validation_foreground']['before_mean_psnr_db']:.3f}变为{held['validation_foreground']['after_mean_psnr_db']:.3f}dB，8张改善。两组像素时间差分残差也略有增大。时间差分包含运动、遮挡和掩码影响，不能单独当作闪烁指标。

上图胸前的浅色条纹仍在。固定完整人物与局部对照显示，两版外观多数相近，手指、头发和抬起的鞋仍偏软；第755帧人物交叉处仍有明显手臂重影。下图保留鞋子的真实细节差距，不用平均分掩盖局部问题。

![原生鞋部细节对照](../outputs/{R.name}/paired-comparison/camera01-frame000751-dark_shoe.png)
> 图源：同上 · 机位01、第751帧 · 原照片、基线、修正姿态后的重建。

**正确的RGB预处理继续保留，本轮渲染模型不替换既有开发基线。** 这次修复没有带来稳定的人体清晰度改善。此前的支持集合和规范人体形状为控制变量而保持固定，并非所有上游几何都重建了一遍。以上机位与时间片段已经反复用于开发，且只有一个训练随机种子，不是独立最终测试。清晰手指、鞋部、复杂遮挡及完整大场景仍未完成。

报告：[最终结论](../outputs/{R.name}/final-summary.json)、[实际看图范围](../outputs/{R.name}/visual-review.json)、[相同训练条件核验](../outputs/{R.name}/training-protocol-audit.json)、[训练后变形独立复算](../outputs/{R.name}/trained-graph-independent-audit.json)、[原机位数值](../outputs/{R.name}/paired-comparison/summary.json)、[另外机位数值](../outputs/{R.name}/heldout/paired-comparison/summary.json)、[原机位时间变化](../outputs/{R.name}/temporal-comparison/summary.json)、[另外机位时间变化](../outputs/{R.name}/heldout/temporal-comparison/summary.json)。
'''
file=L/'experiments/37_dynamic_human_reconstruction.md'
text=file.read_text(encoding='utf8')
if '### 正确颜色输入后的完整重建对照' not in text:
    file.write_text(text+evidence,encoding='utf8')
assets=['final-summary.json','visual-review.json','training-protocol-audit.json','trained-graph-independent-audit.json','paired-comparison/summary.json','heldout/paired-comparison/summary.json','temporal-comparison/summary.json','heldout/temporal-comparison/summary.json','heldout/paired-comparison/camera06-frame000745-full-person.png','paired-comparison/camera01-frame000751-dark_shoe.png']
file=S/'scripts/sync-gs-lab.mjs'
lines=file.read_text(encoding='utf8').splitlines(keepends=True)
index=next(i for i,line in enumerate(lines) if R.name+'/${name}' in line)
if 'final-summary.json' not in lines[index]:
    # Merge into the existing explicit allowlist; never publish model binaries.
    prefix,tail=lines[index].split('.map(name',1)
    current=json.loads(prefix.strip()[3:])
    lines[index]='  ...'+json.dumps(current+assets)+'.map(name'+tail
    file.write_text(''.join(lines),encoding='utf8')
save(R/'published-resources.json',dict(paths=assets,licensed_models_exported=False))

file=L/'manifests/dynamic-human-work.json'
m=read(file)
m.update(status='RGB_pose_control_complete_clarity_unaccepted',current_runs=[],
    last_recorded_utc=datetime.now(timezone.utc).isoformat(),
    outdoor_RGB_pose_repair='outputs/'+R.name+'/final-summary.json',
    outdoor_RGB_pose_website_qa='outputs/'+R.name+'/website-qa.json',
    remaining_quality_issue=summary['remaining'],
    next_actions=[
        '保持正确RGB预处理；067仅保留为图像质量比较基线，不复用其错误输入流程。',
        '优先检验可见手部的跨时刻对应、可见性和人物归属，避免将遮挡猜测送入训练。',
        '把表面运动不一致与细节表示能力分开验证；只有真实原生细节及另外机位同时改善才接纳新模型。',
        '人物清晰度通过后再接回静态背景和大场景；当前不宣称完成。'],
    goal_turn_classification='Progress: RGB preprocessing defect fixed;15000updates120native renders38actual comparison reviews completed. Clarity still unaccepted.')
entry='074correctRGBpose control15000updates120nativeimages38actualpanels;correctinputretained,newcheckpointnotadopted'
if entry not in m['completed']:
    m['completed'].append(entry)
save(file,m)
shutil.copy2(__file__,R/Path(__file__).name)
print(json.dumps(dict(status=summary['status'],reviewed=len(files),known_delta=known['validation_foreground']['mean_delta_db'],other_delta=held['validation_foreground']['mean_delta_db']),ensure_ascii=False))
