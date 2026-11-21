from pathlib import Path
import json,shutil
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-hand-photo-pose-105';SITE=Path('D:/Projects/EngineLab/engine-home')
read=lambda p:json.loads(p.read_text(encoding='utf8'))
r=read(O/'final-summary.json');assert r['visual_review_complete'] and not r['adopted']
p=L/'knowledge/32_dynamic_human_scene_reconstruction.md';s=p.read_text(encoding='utf8');shutil.copy2(p,O/'knowledge-before.md')
anchor='本例的两组候选均未替换原模型。'
assert s.count(anchor)==1
s=s.replace(anchor,anchor+'随后固定全部高斯属性，仅用原照片优化手部关节和手腕，仍未恢复清晰指缝。这说明本例中有限的姿态修正不足以修复现有表面外观，不能由训练损失下降推断细节恢复；也不能据此认定姿态已经准确。[姿态修正的原生图像检验](../experiments/37_dynamic_human_reconstruction.md#固定外观后的手部姿态修正)保留对应条件与画面。',1)
p.write_text(s,encoding='utf8')
p=L/'experiments/37_dynamic_human_reconstruction.md';s=p.read_text(encoding='utf8');shutil.copy2(p,O/'evidence-before.md');assert '### 固定外观后的手部姿态修正' not in s
c=r['comparison']['splits'];t=r['temporal']['metrics'];run=r['training']
s+='''

### 固定外观后的手部姿态修正

从上一节轮廓先验模型出发，固定全部新旧高斯的颜色、尺度、偏移和不透明度，以及旧人体运动、相机和时间参数；仅允许连续手部关节与手腕作有界修正。每次优化共同使用全部105张训练照片，完成60次L-BFGS外层更新。图像先按1920×1080完整渲染，再取原生160×160手部上下文计算损失，保持训练与评测的栅格化方式一致。局部窗口也包含前臂、衣物或玩偶，不能当作精确的手部语义掩码。

![固定外观前提下的手部姿态优化对照](../outputs/dynamic-human-hand-photo-pose-105/native-articulated-hand-comparison.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 04、13机位，留出时刻731帧 · 原照片、原模型、固定手部姿态、优化手部姿态 · 原生像素。

'''
s+=f"训练目标从{run['initial_objective']:.6f}降至{run['final_objective']:.6f}，约下降0.18%。手部上下文的留出时刻PSNR在04机位由{t['04']['heldout_time']['fixed_pose']['psnr_db']:.3f}变为{t['04']['heldout_time']['photo_pose']['psnr_db']:.3f} dB，在13机位由{t['13']['heldout_time']['fixed_pose']['psnr_db']:.3f}变为{t['13']['heldout_time']['photo_pose']['psnr_db']:.3f} dB。实际画面仍有斑块、碎片和指间混合，**本候选不采用，原模型保留**。这只说明该起点和修正范围下的姿态单独优化不足，不证明其他联合优化必然无效。\n\n"
s+='''134个完整原分辨率情境、三模型的804张颜色／透明度PNG、26个连续情境和24组相邻帧变化均已复核；实际查看21个手部上下文、6个连续抽查画面与3组全身图。两段视频完成编码与解码验证，视频本身未作逐帧视觉验收。此区间经过反复开发，不能作为独立场景测试。新演员序列的预处理另存本地，尚未产生训练结果。[完整结果](../outputs/dynamic-human-hand-photo-pose-105/final-summary.json)、[实际看图范围](../outputs/dynamic-human-hand-photo-pose-105/visual-review.json)。
'''
p.write_text(s,encoding='utf8')
files=['final-summary.json','visual-review.json','native-articulated-hand-comparison.png']
p=SITE/'scripts/sync-gs-lab.mjs';s=p.read_text(encoding='utf8');assert O.name+'/${name}' not in s
lines=s.splitlines(keepends=True);i=next(i for i,line in enumerate(lines) if 'dynamic-human-enerf-contour-surface-103/${name}' in line)+1
lines.insert(i,'  ...'+json.dumps(files)+'.map(name => [`outputs/'+O.name+'/${name}`, \'固定外观后的手部姿态修正\']),\n');p.write_text(''.join(lines),encoding='utf8')
(O/'published-resources.json').write_text(json.dumps(dict(paths={O.name:files},licensed_models_exported=False),indent=2)+'\n',encoding='utf8')
shutil.copy2(__file__,O/Path(__file__).name);print('Existing theory paragraph and evidence appendix updated; three explicit public resources.')
