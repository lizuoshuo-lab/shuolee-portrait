"""Publish a completed local matte comparison, never a success inferred from loss."""
from pathlib import Path
from datetime import datetime,timezone
import json,shutil
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-forearm-matte-repair-072';S=Path('D:/Projects/EngineLab/engine-home')
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
f=read(R/'final-summary.json');v=read(R/'visual-review.json');p=read(R/'training-protocol-audit.json');raw=read(R/'raw-forearm-evaluation/summary.json')
assert f['goal_complete'] is False
block=r'''
#### 掩码怎样改变颜色监督

![原照片、原训练目标、局部修正目标与改变的像素](../outputs/dynamic-human-enerf-forearm-matte-repair-072/candidates/camera01-frame000750-review.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap，SAM2.1局部分割及本站核验 · 机位01、帧750 · 输入诊断，非重建结果。

前景掩码既影响轮廓，也可能直接改变颜色监督。令 $M(\mathbf p)\in[0,1]$ 为训练前景透明度，$\mathbf I(\mathbf p)$ 为原照片，$\mathbf b$ 为随机背景色，$\mathbf C_\Theta$ 为已经按可见性和不透明度加权的高斯前景颜色，$A_\Theta$ 为累计不透明度。当前训练的合成图与目标分别为

$$
\widehat{\mathbf I}_{\mathbf b}=\mathbf C_\Theta+(1-A_\Theta)\mathbf b,
\qquad
\mathbf I^*_{\mathbf b}=M\mathbf I+(1-M)\mathbf b.
$$

所有颜色采用归一化RGB，$M$ 和 $A_\Theta$ 无量纲。真实皮肤若被误标为 $M=0$，目标颜色就变成背景色，同时轮廓损失还会要求该处透明。此时仅增加高斯或运动节点，仍在优化错误的目标。修正只应针对原照片确实可见的表面；分割预测不能用来推断被遮挡的手或衣服。

本地对照只修正一个训练机位三张偶数帧中的固定前臂区域，保留其余照片和原有模型。原照片、原掩码、初始化与评估输入均单独保留。候选经逐帧看图接受，仍属于额外弱标签。下图比较实际重新训练后的原生渲染。

![原照片、原掩码目标、基线渲染与局部修正后的真实渲染](../outputs/dynamic-human-enerf-forearm-matte-repair-072/raw-forearm-evaluation/camera01-frame000751-raw-comparison.png)
> 图源：同一公开数据与本站实际训练 · 留出的奇数帧751 · 四列原生像素，无增强。

__CONCLUSION__

对照原照片的误差只在事先选定、明确可见的皮肤内部计算；它不会把原掩码误删的皮肤排除，也不评价手指、轮廓和隐藏表面。全体机位继续保留原指标，避免更换评估口径后把两次成绩直接混比。[训练条件与完整结果](../experiments/37_dynamic_human_reconstruction.md#三张训练图的局部掩码修正)见实验附录。

'''
block=block.replace('__CONCLUSION__',v['narrative_zh'])
file=L/'knowledge/32_dynamic_human_scene_reconstruction.md';text=file.read_text(encoding='utf8')
if '#### 掩码怎样改变颜色监督' not in text:text=text.replace('### 增密梯度与细节恢复',block+'### 增密梯度与细节恢复')
intro='''## 当前可见的重建效果

![原照片、身体运动基线与保留的局部变形模型在新机位的实际效果](../outputs/dynamic-human-enerf-heldout-cameras-068/paired-comparison/camera12-frame000751-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实际训练 · 机位12、帧751 · 原生像素，无增强。

当前保留的局部变形模型已经恢复人物主体、衣物图案和手持物的大体外观，但手指、鞋部及交叉遮挡还没有达到清晰重建要求。灰色是用于检查人物的合成背景，此处尚未联合重建建筑。近期的[运动节点密度对照](#运动控制节点的密度与观测支持)与[局部掩码修正](#掩码怎样改变颜色监督)分别检验模型容量和监督质量；具体取舍见各自的真实画面对照。

'''
if '## 当前可见的重建效果' not in text:text=text.replace('## 1. 静态背景与运动人物怎样共同表示',intro+'## 1. 静态背景与运动人物怎样共同表示')
file.write_text(text,encoding='utf8')
k=f['known'];h=f['heldout'];tr=f['training']
evidence=f'''

### 三张训练图的局部掩码修正

072从067完全相同的52109高斯和384节点图开始，只改变机位01第750、752、754帧中固定矩形内的训练matte；其他102张训练matte与全部原始RGB保持不变。SAM2.1 Hiera Tiny先在七张原生偶数帧上产生候选，七张全部看图，接受其中三个明显漏掉前臂的候选。规则为SAM前景腐蚀3×3后，在固定区域与有效像素内与原matte取最大值，共改变11561个原生像素，其中5962个原值小于0.5。区域包含少量袖口和衣服边缘；不是纯皮肤真值，也不用于补全隐藏表面。旧039、053试验已有掩码质量的局部或混合结果，此次不是首次发现掩码问题，而是当前动态模型上的隔离验证。

![三帧修正中一个实际输入例子](../outputs/dynamic-human-enerf-forearm-matte-repair-072/candidates/camera01-frame000750-review.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap、SAM2.1及本站筛选 · 输入诊断。

实际完成15000次更新，其中{p['matte_supervised_updates']}次抽到三张修正图，最终{tr['points']:,}个高斯，训练{tr['elapsed_seconds']:.2f}秒。相同初始参数、随机背景、抽样序列、身体与时间模型、学习率和增密日程均已复算。三张matte既改变合成颜色目标也改变alpha目标，不能把效果进一步拆成其中一项的独立作用。

90张已知机位留出时间图沿用原matte指标，平均前景PSNR为{k['before_mean_psnr_db']:.4f}→{k['after_mean_psnr_db']:.4f}dB，{k['improved_images']}/90张改善。18张另外机位图为{h['before_mean_psnr_db']:.4f}→{h['after_mean_psnr_db']:.4f}dB，{h['improved_images']}/18张改善。原matte可能漏掉真实皮肤，所以另外固定四张奇数帧的可见皮肤内部多边形，直接对照原照片；只看源图选区，在任何新模型评估之前保存并查看，未用于训练。

这四个内部区域共{raw['total_native_pixels']}个原生像素，合并MSE后计算的PSNR为{raw['pooled_psnr_db']['before']:.4f}→{raw['pooled_psnr_db']['after']:.4f}dB。该数值只描述同一已知机位的可见皮肤内部，不代表轮廓、手指、新机位或全身清晰度。没有按预测alpha除法还原颜色，比较的是模型实际合成到0.5灰背景的像素。

![留出时刻的前臂原生比较](../outputs/dynamic-human-enerf-forearm-matte-repair-072/raw-forearm-evaluation/camera01-frame000751-raw-comparison.png)
> 图源：同上 · 原照片、原matte目标、067基线、072修正。

{v['narrative_zh']} 采用结论：{v['decision_zh']}。当前机位与时段反复用于开发，单种子结果不代表重复性或独立最终测试。120张原生结果、90组时间变化均保留；实际看图范围和数值复算范围分别记录。

![固定遮挡机位中的完整对照](../outputs/dynamic-human-enerf-forearm-matte-repair-072/heldout/paired-comparison/camera06-frame000745-full-person.png)
> 图源：同上 · 原照片、067基线、072局部修正。

报告：[输入修正规则](../outputs/dynamic-human-enerf-forearm-matte-repair-072/protocol.json)、[七张候选看图](../outputs/dynamic-human-enerf-forearm-matte-repair-072/mask-visual-review.json)、[输入独立复算](../outputs/dynamic-human-enerf-forearm-matte-repair-072/mask-independent-audit.json)、[预先固定训练条件](../outputs/dynamic-human-enerf-forearm-matte-repair-072/ablation-protocol.json)、[实际训练核验](../outputs/dynamic-human-enerf-forearm-matte-repair-072/training-protocol-audit.json)、[已训练变形复算](../outputs/dynamic-human-enerf-forearm-matte-repair-072/trained-graph-independent-audit.json)、[已知机位](../outputs/dynamic-human-enerf-forearm-matte-repair-072/paired-comparison/summary.json)、[另外机位](../outputs/dynamic-human-enerf-forearm-matte-repair-072/heldout/paired-comparison/summary.json)、[原照片区域定义](../outputs/dynamic-human-enerf-forearm-matte-repair-072/raw-forearm-evaluation/protocol.json)、[原照片直接比较](../outputs/dynamic-human-enerf-forearm-matte-repair-072/raw-forearm-evaluation/summary.json)、[已知机位时间变化](../outputs/dynamic-human-enerf-forearm-matte-repair-072/temporal-comparison/summary.json)、[另外机位时间变化](../outputs/dynamic-human-enerf-forearm-matte-repair-072/heldout/temporal-comparison/summary.json)、[实际看图](../outputs/dynamic-human-enerf-forearm-matte-repair-072/visual-review.json)、[最终判断](../outputs/dynamic-human-enerf-forearm-matte-repair-072/final-summary.json)。
'''
file=L/'experiments/37_dynamic_human_reconstruction.md';text=file.read_text(encoding='utf8')
if '### 三张训练图的局部掩码修正' not in text:file.write_text(text+evidence,encoding='utf8')
assets=['protocol.json','mask-visual-review.json','mask-independent-audit.json','ablation-protocol.json','training-protocol-audit.json','trained-graph-independent-audit.json','paired-comparison/summary.json','heldout/paired-comparison/summary.json','raw-forearm-evaluation/protocol.json','raw-forearm-evaluation/summary.json','temporal-comparison/summary.json','heldout/temporal-comparison/summary.json','visual-review.json','final-summary.json','candidates/camera01-frame000750-review.png','raw-forearm-evaluation/camera01-frame000751-raw-comparison.png','heldout/paired-comparison/camera06-frame000745-full-person.png']
assert all((R/x).is_file() for x in assets)
file=S/'scripts/sync-gs-lab.mjs';text=file.read_text(encoding='utf8')
if R.name+'/${name}' not in text:
 lines=text.splitlines(keepends=True);i=next(i for i,line in enumerate(lines) if 'dynamic-human-enerf-fine-graph-071/${name}' in line)+1
 lines.insert(i,'  ...'+json.dumps(assets)+f".map(name => [`outputs/{R.name}/${{name}}`, '前臂监督修正对照']),\n");file.write_text(''.join(lines),encoding='utf8')
file=L/'manifests/dynamic-human-work.json';m=read(file);m.update(status='fine_graph_and_local_matte_controls_complete_clarity_unaccepted',current_runs=[],last_recorded_utc=datetime.now(timezone.utc).isoformat(),outdoor_fine_graph='outputs/dynamic-human-enerf-fine-graph-071/final-summary.json',outdoor_local_matte_repair='outputs/'+R.name+'/final-summary.json',outdoor_local_matte_website_qa='outputs/'+R.name+'/website-qa.json',remaining_quality_issue=v['narrative_zh'],next_actions=v['next_actions'],goal_turn_classification='Progress:071and072fixed15000update controls completed with120native images each; actual visual decisions recorded. Overall moving-human clarity not accepted.')
if v['adopt_as_development_baseline']:m['current_best_continuous_control']='outputs/'+R.name+'/residual_graph'
for entry in ['071finer1536node graph completed120native images and64visualpanels,notadopted','072three localtrainingmatte corrections15000updates120nativeimages plusrawskininteriorcheck,clarityunaccepted']:
 if entry not in m['completed']:m['completed'].append(entry)
save(file,m);shutil.copy2(__file__,R/Path(__file__).name);print('072knowledge/evidence/allowlist/manifest prepared for export and build.')
