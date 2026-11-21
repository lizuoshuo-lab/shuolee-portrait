from pathlib import Path
import json, shutil
L=Path(__file__).resolve().parents[2]; O=L/'outputs/dynamic-human-enerf-contour-surface-103'; SITE=Path('D:/Projects/EngineLab/engine-home')
read=lambda p:json.loads(p.read_text(encoding='utf8')); result=read(O/'final-summary.json')
assert result['visual_review_complete'] and not result['clarity_accepted']
p=L/'knowledge/32_dynamic_human_scene_reconstruction.md'; s=p.read_text(encoding='utf8'); shutil.copy2(p,O/'knowledge-before.md')
start=s.index('本地对照在每个手部三角面放置一个高斯'); end=s.index('判断新表面是否真正参与图像',start)
s=s[:start]+'''本地对照在每个手部三角面放置一个高斯，共1538个。两组分别采用仅由预测关键点拟合的手部姿态，以及额外利用图像轮廓约束拟合的姿态；颜色、不透明度、局部偏移与尺度使用同样的105张照片优化6000次。两组均固定关节先验，并在训练全程把6330个旧邻域高斯的不透明度限制在不高于0.02，以检验改善先验轮廓能否改善真实图像。

![两种手部先验在相同训练条件下的原生渲染对照](../outputs/dynamic-human-enerf-contour-surface-103/native-articulated-hand-comparison.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 上为04机位，下为13机位，均为留出时刻731帧 · 原照片、原模型、关键点先验、增加轮廓约束的先验 · 原生像素。

**轮廓拟合更好，并不保证手指渲染更清楚。** 右侧两组仍有明显斑块和错误指缝，限制旧表面后还损伤了部分手部与玩偶边界。新增表面参与合成的程度提高，但其单独渲染也没有恢复准确的手指外观。因此，轮廓覆盖、表面可见性和最终重建清晰度必须分别检查，不能用前两者代替真实图像效果。

'''+s[end:]
start=s.index('因此，前方旧表面接近不透明时'); end=s.index('\n### ',start)
s=s[:start]+r'''因此，前方旧表面接近不透明时，新表面即使位于合适的投影位置，也可能只能获得很小的颜色梯度。对每个旧高斯设置较低的不透明度上限，并不意味着整层旧表面变得透明。若某条光线经过的100个旧高斯在该像素处均满足$\alpha_j=0.02$，其剩余透射率为

$$
T_{\mathrm{old}}=(1-0.02)^{100}\approx0.133.
$$

这个合成示例说明，多个低不透明度高斯仍可共同遮挡后方表面。降低旧表面的贡献时，还要检查其是否同时承载前臂、衣物或道具；几何邻域不能直接当作语义手部标签。另一方面，即使可见性改善，表面位置、手指对应或共享颜色拟合不准确，仍会形成斑块。本例的两组候选均未替换原模型。[先验与表面表示](../experiments/37_dynamic_human_reconstruction.md#手部先验与关节表面高斯的图像检验)、[不透明度初始化对照](../experiments/37_dynamic_human_reconstruction.md#旧表面遮挡与不透明度初始化)和[轮廓先验的真实图像检验](../experiments/37_dynamic_human_reconstruction.md#轮廓先验与真实手部渲染)保留实验条件及原生图片。

'''+s[end:]
p.write_text(s,encoding='utf8')
p=L/'experiments/37_dynamic_human_reconstruction.md'; s=p.read_text(encoding='utf8'); shutil.copy2(p,O/'evidence-before.md'); assert '### 轮廓先验与真实手部渲染' not in s
c=result['comparison']['splits']; t=result['temporal']['metrics']
s+='''

### 轮廓先验与真实手部渲染

手部轮廓先验采用SAM产生并逐图检查的弱掩码：只保留较可靠的内部正样本和邻域负样本，排除边缘及腕部不确定区。22张原生裁图中18张参与轮廓损失，04、13机位的4张仅用于轮廓诊断；但它们的预测关键点仍参与拟合，因此不能称为独立留出相机。控制组与候选组均使用105组HaMeR预测关键点、相同起点及1500次更新，固定体形和尺度。诊断图的已知区域交并比由0.8847提高至0.9422，实际叠加仍有细指间隙不准。该指标衡量弱掩码覆盖，不是三维手指准确率。

随后用两种先验分别初始化1538个表面高斯，固定关节先验，学习颜色、不透明度、局部偏移与尺度。两组使用同一105张训练照片、同一6000步采样顺序，并持续限制6330个旧邻域高斯的不透明度不超过0.02，其余246654个旧高斯的不透明度及全部旧几何、颜色和运动保持逐项一致。两组之间只改变先验；与此前允许旧不透明度重新增长、允许光度姿态修正的模型相比，则同时改变了其他条件，不能称为单一先验消融。

![轮廓先验与关键点先验的留出时刻原生效果](../outputs/dynamic-human-enerf-contour-surface-103/native-articulated-hand-comparison.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 原照片、原模型、关键点先验、增加轮廓约束的先验 · 04、13机位731帧。

'''
s+=f"两组各完成134张1920×1080原生图像评测。已知机位留出时刻的人物区域平均PSNR分别为{c['validation_time']['foreground_psnr_db']['prior_control']:.4f}和{c['validation_time']['foreground_psnr_db']['contour_prior']:.4f} dB，留出机位为{c['validation_camera']['foreground_psnr_db']['prior_control']:.4f}和{c['validation_camera']['foreground_psnr_db']['contour_prior']:.4f} dB。21个手部窗口及连续序列中的6个固定时刻均未显示清晰手指恢复。两组的局部上下文也退化：04机位连续留出时刻的该区域PSNR由原模型的{t['04']['heldout_time']['parent']['psnr_db']:.2f}降至轮廓组的{t['04']['heldout_time']['contour_prior']['psnr_db']:.2f} dB，13机位由{t['13']['heldout_time']['parent']['psnr_db']:.2f}降至{t['13']['heldout_time']['contour_prior']['psnr_db']:.2f} dB。区域含前臂和玩偶，不是精确手部语义掩码。\n\n"
s+='''只读分解另外保留8组画面，分别显示共同合成、旧表面单独渲染和新表面单独渲染；新表面自身仍有斑块与指缝误差。该反事实渲染不重新训练，不能证明遮挡对训练没有影响，也不是可替代完整场景的新模型。两组均未通过清晰度验收，原模型保留。

134组四模型的指标及1072张PNG经过独立复算；另核验26个连续情境、24组相邻变化及两段5帧/秒慢放视频。视频速度用于检查，不代表采集频率。当前区间仍为开发集，独立场景与完整背景联合训练尚待完成。[完整结果](../outputs/dynamic-human-enerf-contour-surface-103/final-summary.json)、[实际看图范围](../outputs/dynamic-human-enerf-contour-surface-103/visual-review.json)、[可见贡献](../outputs/dynamic-human-enerf-contour-surface-103/contribution/summary.json)、[只读分解](../outputs/dynamic-human-enerf-contour-surface-103/visibility/summary.json)、[04机位分解示例](../outputs/dynamic-human-enerf-contour-surface-103/visibility/contour_prior-camera04-frame000731.png)。
'''
p.write_text(s,encoding='utf8')
files=['final-summary.json','visual-review.json','native-articulated-hand-comparison.png','contribution/summary.json','visibility/summary.json','visibility/contour_prior-camera04-frame000731.png']
p=SITE/'scripts/sync-gs-lab.mjs'; s=p.read_text(encoding='utf8'); assert O.name+'/${name}' not in s
lines=s.splitlines(keepends=True); i=next(i for i,l in enumerate(lines) if 'dynamic-human-enerf-hand-visibility-100/${name}' in l)+1
lines.insert(i,'  ...'+json.dumps(files)+'.map(name => [`outputs/'+O.name+'/${name}`, \'轮廓先验与原生手部渲染\']),\n'); p.write_text(''.join(lines),encoding='utf8')
(O/'published-resources.json').write_text(json.dumps(dict(paths={O.name:files},licensed_models_exported=False),indent=2)+'\n',encoding='utf8')
shutil.copy2(__file__,O/Path(__file__).name); print('Updated existing section and six explicitly allowed public resources.')
