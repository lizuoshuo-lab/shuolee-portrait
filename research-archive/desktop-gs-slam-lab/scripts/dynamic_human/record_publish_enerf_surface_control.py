from pathlib import Path
import json,shutil
from datetime import datetime,timezone
LAB=Path(__file__).resolve().parents[2];OUT=LAB/'outputs/dynamic-human-enerf-surface-gs-055';SITE=Path('D:/Projects/EngineLab/engine-home')
read=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,x:p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
r=read(OUT/'paired-comparison/summary.json');a=read(OUT/'paired-training-state-audit.json')
assert all(read(OUT/b/'native-evaluation/independent-image-audit.json')['full_pngs']==24 for b in ['surface_only','surface_regularized'])
save(OUT/'visual-review.json',dict(status='all6wholeperson_and17fixed_native_detail_panels_actually_reviewed',
 cameras=['00','06','12','01','08','17'],detail_regions=read(LAB/'outputs/dynamic-human-enerf-camera10-time-050/fixed-detail-regions.json')['regions'],
 findings={'00':'White-shirt exposed arm smoother, less streaking; dark-shirt hanging arm remains elongated. Print and airborne shoe still smeared.',
 '06':'Back forearm becomes smoother, shirt less wavy; crossed forearms and occluded dark shoulder retain wrong shape. Shoe/toy remain blurred.',
 '12':'Right exposed arm smoother; dark hanging forearm remains elongated with background texture. Print worsens numerically and visibly unresolved.',
 'training':'01/08/17 mostly fitted but all three foreground PSNR decline with regularization; hand boundaries, toys and airborne shoe not recovered.'},
 conclusion='Regularization improves all three development heldout foreground scores and selected surface streaks, but does not meet clear moving-human acceptance.',goal_complete=False))
save(OUT/'final-summary.json',dict(status='surface_regularization_partial_development_gain_human_clarity_still_unaccepted',
 branches=a['branches'],heldout_foreground_delta_db={x['camera']:x['foreground_delta_db'] for x in r['records'] if x['split']=='heldout'},
 training_foreground_delta_db={x['camera']:x['foreground_delta_db'] for x in r['records'] if x['split']=='training_diagnostic'},
 paired_input_audit='paired-training-state-audit.json',visual_review='visual-review.json',
 method='gsplat1.5.3 2DGS, L1 depth-distortion variant and opacity-weighted normal consistency on eroded training foreground; not exact original2DGS reproduction.',
 interpretation='Gains are regularizer effect within this paired2Dsurface pipeline. A2Dvs3D comparison would include renderer, runtime version and initialization differences.',
 limits='Single frame750. Views00/06/12 are repeatedly used development views, not fresh final test evidence. No temporal or large-scene completion.',goal_complete=False))
p=LAB/'knowledge/32_dynamic_human_scene_reconstruction.md';text=p.read_text(encoding='utf8')
addition=r'''
### 表面表示与表面一致性约束

![原照片、仅图像拟合的二维高斯与增加表面约束后的完整人物对照](../outputs/dynamic-human-enerf-surface-gs-055/paired-comparison/camera06-frame000750-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站运行 · 机位 06 · 原生像素，无增强。

二维高斯泼溅（2D Gaussian Splatting，2DGS）用位于三维空间的有向面元表示表面。这里的“二维”指每个高斯的局部支撑平面，不是把场景降为一张平面图像。设面元中心为 $\boldsymbol\mu$，正交单位切向量为 $\mathbf t_u,\mathbf t_v$，两个正尺度为 $s_u,s_v$，则面元位置与局部高斯权重写成

$$
\mathbf X(u,v)=\boldsymbol\mu+s_u u\,\mathbf t_u+s_v v\,\mathbf t_v,
\qquad G(u,v)=\exp\!\left[-\frac{u^2+v^2}{2}\right].
$$

$u,v$ 无量纲；中心与尺度采用场景坐标单位，单位法线为 $\mathbf n=\mathbf t_u\times\mathbf t_v$。像素的视线与面元求交后，可以计算局部权重并按深度合成颜色。它使局部表面的方向明确，但仍需要损失函数约束面元之间的关系。

深度畸变约束鼓励同一像素的主要颜色贡献集中在较窄的深度范围；法线一致性约束鼓励面元法线与深度图所导出的表面法线一致。在本站采用的实现变体中，令 $w_i$ 为透明度合成权重，$z_i$ 为按深度排列的面元深度，$\mathcal P$ 为腐蚀后的有效前景像素集，则深度项为

$$
\mathcal L_{\mathrm{dist}}=
\frac{1}{|\mathcal P|}\sum_{p\in\mathcal P}
2\sum_{i>j}w_i(p)w_j(p)\bigl(z_i-z_j\bigr).
$$

排序成立时，它等于双重求和形式的加权绝对深度差；量纲为场景长度单位。这是所用渲染器的 $L_1$ 变体，应与原论文的平方深度差设计区分。法线项使用透明度加权后的渲染法线与深度法线，不能简单等同于两条单位向量的余弦损失。实验采用深度项权重 $0.01$、法线项权重 $0.05$，数值依赖场景尺度和损失归一化方式，不能作为通用参数直接照搬。

上图中，右列的部分前臂条纹和衣服起伏减轻，但交叉手臂的位置、被遮挡的肩部及手指仍不正确。表面正则项提供的是几何偏好，并没有新增人体观测或正确骨架。应把“表面更平滑”和“人体结构恢复正确”分开判断。方法参考 [2DGS](https://surfsplatting.github.io/)；本地成对条件及效果见[表面约束对照](../experiments/37_dynamic_human_reconstruction.md#二维高斯表面与几何正则项)。

'''
if '### 表面表示与表面一致性约束' not in text:
 anchor='### 掩码中的未知区域与几何观测覆盖';assert text.count(anchor)==1;text=text.replace(anchor,addition+anchor);p.write_text(text,encoding='utf8')
p=LAB/'experiments/37_dynamic_human_reconstruction.md';text=p.read_text(encoding='utf8')
addition='''

### 二维高斯表面与几何正则项

在独立运行环境中完成 gsplat 1.5.3 的二维高斯 CUDA 渲染、反向传播与两组正式训练。两组各更新 6,000 次，使用完全相同的 52,109 个轮廓壳层初始点、局部 PCA 切平面方向、15 台训练相机的原生图像与掩码，以及全部相机和背景颜色采样。增加约束的一组在有效前景内部加入深度畸变和法线一致性项；已验证约束对几何产生有限、非零梯度。

最终未加约束组为 205,318 个面元，加约束组为 208,340 个。三个未参与训练的开发机位 00、06、12 前景 PSNR 分别提高 0.614、0.628、0.355 dB；训练机位 01、08、17 则分别下降 1.259、0.732、1.325 dB。这里的留出相机已在多轮开发中反复查看，不能当作全新的最终测试集。两组 48 张原生输出、24 张比较图的像素和指标已独立重算，六张完整人物对照与预先确定的 17 个细节区域已逐张查看。

![原照片、无表面约束与有表面约束的完整人物效果](../outputs/dynamic-human-enerf-surface-gs-055/paired-comparison/camera06-frame000750-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站运行 · 无增强。

部分白衣人物的前臂条纹减轻，衣服表面更平顺；黑衣人物的下垂前臂仍有拉伸，交叉手臂、遮挡肩部、衣服图案和抬起的鞋仍有明显失真。结论是局部改善，清晰移动人物仍未验收。与此前三维高斯的整套流程比较还同时涉及渲染器版本、面元表示和初始方向，不能写成单独改变维度的因果结论。

见[结果汇总](../outputs/dynamic-human-enerf-surface-gs-055/final-summary.json)、[事前协议](../outputs/dynamic-human-enerf-surface-gs-055/protocol.json)、[成对训练核验](../outputs/dynamic-human-enerf-surface-gs-055/paired-training-state-audit.json)、[无约束图像核验](../outputs/dynamic-human-enerf-surface-gs-055/surface_only/native-evaluation/independent-image-audit.json)、[有约束图像核验](../outputs/dynamic-human-enerf-surface-gs-055/surface_regularized/native-evaluation/independent-image-audit.json)和[实际图像审阅](../outputs/dynamic-human-enerf-surface-gs-055/visual-review.json)。
'''
if '### 二维高斯表面与几何正则项' not in text:p.write_text(text+addition,encoding='utf8')
p=SITE/'scripts/sync-gs-lab.mjs';text=p.read_text(encoding='utf8')
if 'dynamic-human-enerf-surface-gs-055/${name}' not in text:
 lines=text.splitlines(keepends=True);idx=next(i for i,l in enumerate(lines) if 'dynamic-human-enerf-surface-tracks-054/${name}' in l)+1
 lines.insert(idx,"  ...['final-summary.json','protocol.json','paired-training-state-audit.json','surface_only/native-evaluation/independent-image-audit.json','surface_regularized/native-evaluation/independent-image-audit.json','visual-review.json','paired-comparison/camera06-frame000750-full-person.png'].map(name => [`outputs/dynamic-human-enerf-surface-gs-055/${name}`, '二维高斯表面约束对照']),\n")
 p.write_text(''.join(lines),encoding='utf8')
p=LAB/'manifests/dynamic-human-work.json';m=read(p)
m['outdoor_surface_control']='outputs/dynamic-human-enerf-surface-gs-055/final-summary.json'
m['current_runs']=[dict(name='056surface regularized30000',exec_session=22690,state='training_running_verified_step3100')]
m['status']='surface_constraint_partial_gain_longer_optimization_running'
m['goal_turn_classification']='Progress:055paired6000updates and48nativePNGs verified; all6wholeperson17details reviewed.056longertraining active. No external blocker.'
m['next_actions']=['Finish05630000updates and inspect same native limb/occlusion regions.','Keep single-frame development gains separate from multi-time and full-scene acceptance.']
entry='055paired2DGS6000updates complete: all3development heldout FG scores improve .355-.628dB withsurface regularization; limbclarity stillunaccepted.'
if entry not in m['completed']:m['completed'].append(entry)
m['last_recorded_utc']=datetime.now(timezone.utc).isoformat();save(p,m)
shutil.copy2(__file__,OUT/'source-snapshot'/Path(__file__).name)
print('055 reviewed and source pages updated;056 retained as running')
