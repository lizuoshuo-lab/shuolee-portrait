from pathlib import Path
import datetime, hashlib, json, shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LAB=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab')
B=LAB/'outputs/mini-splatting2-mip-depth-001/shuolee'; V=B.parent/'viewer'
A=B/'current/static-white-edge-audit-068'; R=B/'current/person-static-alpha-fresh-069'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=datetime.datetime.now().astimezone().isoformat(timespec='seconds')
def save(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8');t.replace(p)

before=read(A/'boundary-before/boundary-audit.json'); after=read(A/'boundary-after/boundary-audit.json')
review=read(A/'after/alpha-review.json'); grad=read(A/'objective-diagnosis/objective-gradients.json')
assert grad['completed'] and grad['optimizer_updates']==0 and grad['model_unchanged']
assert read(R/'gaussian-run/result.json')['steps']==18000
fields=['inner_boundary_undercovered_fraction','inner_boundary_without_support_fraction','outer_boundary_overcovered_fraction']
means={label:{k:float(np.mean([r[k] for r in data['rows']])) for k in fields} for label,data in [('064',before),('069',after)]}
ratios=[r['attribution']['xyz']['actual_alpha_to_photo_norm_ratio'] for r in grad['records']]
eligible=[r['normal_eligible_contour_fraction'] for r in grad['records']]
diagnosis=dict(updated=now,subject='共享静态穿衣人物的边界学习',optimizer_updates_in_diagnosis=0,
    summary='069外部亮边减少，但局部内侧覆盖下降及红黑碎条仍存在，未修复整体边界。下一轮先完成投影、遮挡与多视图支持归因。',
    evidence=[
      dict(fact='白边存在于原生渲染，不能单独归因于网页或PLY加载。',source='after/observed-023-backgrounds.jpg'),
      dict(fact='064位置、尺度、旋转与透明度均有轮廓误差梯度；不是所有边界参数都不可学习。',source='boundary-before/boundary-audit.json'),
      dict(fact='064实际加权透明度项相对RGB项的位置梯度范数较小。',range=ratios,source='objective-diagnosis/objective-gradients.json',limit='冻结的末态、固定邻居、原始梯度；没有Adam预条件，不能等同每步位移或历史成因。'),
      dict(fact='11像素软轮廓带仅约9%像素可直接参与法线项；多视图项还要求双侧内部与整块有效。',eligible_fraction=eligible,source='objective-diagnosis/objective-gradients.json'),
      dict(fact='真实方位覆盖约181度，观察高度角跨度约2.84度。',source='../static-surface-comparison-067/observation-coverage.json',limit='围绕固定目标的相机高度角，不是相机光轴俯仰角；175图像不等于175个独立标定相机。')],
    hypotheses=[
      dict(id='H1',cause='显示器或查看器造成白边',status='不足以解释',reason='原生光栅器复现。'),
      dict(id='H2',cause='边界完全没有梯度',status='被当前可见轮廓证据否定',reason='四类几何/覆盖参数均有非零有限梯度；不推广到没有观测支持的表面。'),
      dict(id='H3',cause='轮廓监督在全图归一化中被弱化',status='有直接机制证据，单独充分性未证明',reason='只占约2.4%至3.2%的轮廓像素与3.3%至4.2%的相对位置梯度；069同时改变三项。'),
      dict(id='H4',cause='相机、分割或真实微动造成各视角不一致',status='待核验',reason='有相反的逐点梯度，但可见性和切向变化也会造成相反梯度；不能据此认定标定错误。原标定第0视图和插值第23视图均有局部残差。'),
      dict(id='H5',cause='缺失拍摄方向导致隐藏几何欠约束',status='覆盖缺口已证实，具体伪影归因未完成',reason='未拍摄俯仰只作诊断；不能仅靠增点或更强轮廓项恢复真实隐藏表面。'),
      dict(id='H6',cause='颜色与累计覆盖共同补偿误差',status='理论成立，组合试验支持局部外观改善',reason='069不能区分预乘重采样、双背景和轮廓归一化各自效果；外观改善不等于几何修复。'),
      dict(id='H7',cause='边界点的观测剔除或增密策略留下支持空洞',status='未测量',reason='源码每1000步移除少于2帧实际观测的点；须核验具体问题点，不能直接停用剔除。')],
    round069=dict(from_scratch=True,initial_points=200000,steps=18000,points=249326,single_term_ablation=False,
        actual_images_inspected=['after/observed-023-backgrounds.jpg','boundary-after/boundary-000.jpg','boundary-after/boundary-023.jpg','after/below25-backgrounds.jpg','after/right90-backgrounds.jpg'],
        observed_metrics=review['summary'],coverage_means=means,
        boundary_repair_accepted=False,geometry_accepted=False,quality_accepted=False,
        decision='不晋升为最优，不部署00；作为反例和必要对照保留。'),
    next_diagnostic=dict(training_started=False,
      question='哪些真实视角要求保留同一处边界，而另一些要求移除它；冲突是否来自投影或可见性？',
      method=['固定0、23、69、99、135、174相机与头后、脸前、衣领、鞋边区域，不按结果改选样本。',
              '对产生过量或缺失覆盖的高斯计算实际alpha合成贡献；不能用中心是否落在遮罩内代替贡献与遮挡。',
              '在59个已标定与116个插值相机分别核查可见性、轮廓支持和投影残差；回看原图确定分割是否截断头发或衣物。',
              '再检查涉及点在剔除/增密阶段的观测支持。若证据只支持监督稀释，下一次只改变轮廓归一化，不再同时改变颜色管线。'],
      limitation='不同视角的轮廓点未必是同一物理表面点；遮挡后方的点不能被错误视为违反前景遮罩。'),
    iteration_policy=dict(analysis_before_training=True,new_training_ready=False,user_approval_required=False,
      required_before_next_training=['固定症状区域和可复现视角','竞争假设及各自证据与反证','不改变模型的诊断结论','单一主变量与固定条件','预先定义正反预测、主体保护标准和停止条件','从头初始化、有限预算与可独立加载PLY'],
      launch_rule='证据足以区分候选原因后自主执行最小对照；证据不足继续诊断，不扫描权重或盲目叠模块。'))
save(A/'causal-analysis.json',diagnosis)
save(R/'phase-conclusion.json',diagnosis['round069']|dict(updated=now,analysis=str(A/'causal-analysis.json')))

# Actual same-pixel photographic comparison; no synthesized or repainted content.
mask=np.asarray(Image.open(A/'after/observed-023-mask.png'))
yy,xx=np.where(mask>127);top=int(yy.min());head_end=top+round((yy.max()-top)*.28)
hy,hx=np.where((mask>127)&(np.indices(mask.shape)[0]<head_end))
box=(max(0,int(hx.min())-36),max(0,top-24),min(mask.shape[1],int(hx.max())+36),min(mask.shape[0],head_end+22))
files=['observed-023-source.png','observed-023-064-bg0.00.png','observed-023-069-bg0.00.png']
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23)
canvas=Image.new('RGB',(1260,430),(245,245,245));dr=ImageDraw.Draw(canvas)
for i,(name,label) in enumerate(zip(files,['实拍原图','064：原静态模型','069：边界组合对照'])):
    im=Image.open(A/'after'/name).crop(box);im.thumbnail((408,388))
    canvas.paste(im,(i*420+(420-im.width)//2,40));dr.text((i*420+12,5),label,font=font,fill=(20,28,30))
canvas.save(A/'head-boundary-comparison.jpg',quality=96)
save(A/'head-comparison-provenance.json',dict(crop_box=box,common_crop=True,sources={n:sha(A/'after'/n) for n in files},pixel_content_modified=False))

save(A/'review-decision.json',dict(actually_inspected=True,quality_accepted=False,boundary_repair_accepted=False,
    ppt_effect='轮廓外亮度降低，但覆盖位置仍需独立核验；不是边界修复完成。',
    ppt_summary='共享静态场恢复穿衣外观；边界监督组合降低部分轮廓外亮度。',
    ppt_limit='头后仍有碎条，部分轮廓内缺失增加；仰视与鞋底仍未通过。\n先核查投影、遮挡和多视图支持，再选择单变量对照。',
    evidence=diagnosis['round069'],analysis_file='causal-analysis.json'))
save(B/'current/static-surface-comparison-067/review-decision.json',dict(actually_inspected=True,
    display_candidate='066',display_candidate_label='取消多视图约束',quality_accepted=False,
    ppt_effect='同相机保留主体外观；关闭约束后局部衣物和鞋边表现发生变化。',
    ppt_summary='三个静态分支均完成原生同相机对照。',ppt_limit='鞋底缺口与仰视重影仍存在，未证实整体改善。',
    inspected=['right90.jpg','below25.jpg','front-shoes.jpg'],continuous_path_fully_reviewed=False))

for round_id,runname in [('066','person-static-no-multiview-fresh-066'),('069','person-static-alpha-fresh-069')]:
    rr=B/'current'/runname;pr=rr/'review'
    assert all((pr/x).exists() for x in ['standalone-gaussian.png','standalone-centers.png'])
    save(pr/'viewer-validation.json',dict(updated=now,entry='打开文件',local_viewer=True,pc00_opened=False,
        gaussian_mode=True,point_center_mode=True,default_up_axis='+Y',quality_accepted=False,
        evidence={x:sha(pr/x) for x in ['standalone-gaussian.png','standalone-centers.png']}))

plan=B/'long-term-plan.md';text=plan.read_text(encoding='utf8');start=text.index('<!-- GVHMR_059_ACTIVE -->');end=text.index('<!-- END_GVHMR_059_ACTIVE -->')+len('<!-- END_GVHMR_059_ACTIVE -->')
active='''<!-- GVHMR_059_ACTIVE -->
## 当前执行：先归因边界失效，再选择静态训练改动

用户已明确废弃动态路线，眼镜专项停止，不同步E盘，本地验证通过后才考虑00部署。175个有效实拍帧只监督一个穿衣人物静态场，其中59个相机已标定、116个为插值；插值不会补充未拍摄的俯仰观测。每轮从原始观测重新拟合粗范围，重新生成20万高斯和优化器，训练18000步，旧结果仅作只读对照。

本阶段先定位具体区域、观察方向和失效环节，再列竞争假设、直接证据、反证与仍未知的部分。先做不改变模型的覆盖、实际梯度、可见性和投影诊断；只有诊断足以支持一种原因，才预先固定单一主变量、数据/预算、支持和否定预测、主体保护标准与停止条件，启动下一轮。多项一起变化只能判定组合效果，不作单项因果结论。无需增加用户逐轮批准。

064原生渲染有后脑过量覆盖，边界参数有非零梯度。冻结末态的7视图诊断中，透明度项对相关点的位置梯度约为RGB项的3.3%—4.2%；11像素软轮廓带仅约8.6%—9.0%直接参与法线项。它们支持监督分配不均的解释，但不能证明只加权即可修复。相机、遮罩、可见性、真实微动与剔除策略仍须逐项核验。

069已从头完成18k步、249326点，同时改了预乘重采样、双背景合成与轮廓归一化，不是单项消融。6个已见视图边界alpha误差均值由0.12891降至0.09446，轮廓外亮度下降，但第0视图轮廓内部覆盖不足从4.75%升至13.89%，后脑出现红黑碎条，仰视和鞋部未通过。069不晋升最优、不部署00。065取消尺度压缩、066取消多视图约束亦未获得明确总体改善。

下一步是只读归因：固定头后、脸前、衣领、鞋边，按真实合成贡献追踪问题高斯，在已标定和插值相机分别核查遮挡与轮廓支持。不能仅凭点中心投影或相反梯度判定相机错误。若只支持监督稀释，下一次仅改变轮廓归一化；若支持投影或分割冲突，先修复输入。070尚未启动，不扫描权重。每轮仍提供标准+Y PLY，并分别记录格式兼容与质量验收。

证据与分析：current/static-white-edge-audit-068/causal-analysis.json；网站body-smplx.html#boundary-analysis。历史动态计划不再作为后续执行项。
<!-- END_GVHMR_059_ACTIVE -->'''
plan.write_text(text[:start]+active+text[end:],encoding='utf8')
status=read(B/'work-status.json');status.update(updated=now,updated_at=now,current_experiment='069 reviewed; boundary causal analysis',
    current_phase='static_boundary_causal_analysis',dynamic_route_abandoned=True,quality_accepted=False,
    next_action=diagnosis['next_diagnostic']['question']+' 先按固定视角核查实际合成贡献、遮挡、59标定/116插值相机与遮罩一致性；原因未区分前不启动070、不扫描损失权重。',
    iteration_policy=diagnosis['iteration_policy'],boundary_analysis=str(A/'causal-analysis.json'))
status['latest_local_execution'].update(state='completed_reviewed_not_accepted',running=False,steps=18000,points=249326,quality_accepted=False)
save(B/'work-status.json',status)

assets=V/'body-boundary-current';assets.mkdir(exist_ok=True)
for src,name in [(A/'head-boundary-comparison.jpg','head-comparison.jpg'),(A/'boundary-after/boundary-000.jpg','alpha-coverage.jpg'),(A/'after/below25-backgrounds.jpg','below25-comparison.jpg'),(A/'causal-analysis.json','analysis.json')]:shutil.copy2(src,assets/name)
page=V/'body-smplx.html';t=page.read_text(encoding='utf8');s=t.index('<!-- CLOTHING_PIPELINE_CURRENT_BEGIN -->');e=t.index('<!-- CLOTHING_PIPELINE_CURRENT_END -->')+len('<!-- CLOTHING_PIPELINE_CURRENT_END -->')
old=t[s:e].replace('<!-- CLOTHING_PIPELINE_CURRENT_BEGIN -->','').replace('<!-- CLOTHING_PIPELINE_CURRENT_END -->','').replace('id="pose-clothing-current"','id="static-064-history"').replace('id="clothing-delivery"','id="clothing-delivery-064"')
new='''<!-- CLOTHING_PIPELINE_CURRENT_BEGIN -->
<section id="pose-clothing-current"><h2 id="boundary-analysis">边界为什么没有被正确学到</h2>
<p>当前目标是所有真实观察方向共同约束一个穿衣人物静态场。边界修复必须让高斯的覆盖位置与有效观测一致，并保护头发、衣服和鞋边；颜色变暗或换成白色背景不能证明修复。</p>
<p class="notice"><strong>先分析，再迭代。</strong>每轮先确定失效区域和视角，比较竞争解释并用只读诊断筛选，再固定一个主变量做最小对照。069已完成，但未通过边界质量验收，尚未启动下一轮训练。</p>
<h3>从现象区分外观与覆盖</h3>
<figure><a href="body-boundary-current/head-comparison.jpg"><img src="body-boundary-current/head-comparison.jpg" alt="同像素裁片：实拍原图、064原静态、069边界对照"></a><figcaption>同一观察相机与裁片范围。后脑白壳减轻，但出现红黑碎条，前侧轮廓仍有缺失；全部为实际输出。</figcaption></figure>
<p>渲染颜色由前景颜色贡献与剩余透射背景共同组成。两者能在图像误差中相互补偿，所以图像看起来更接近，不保证高斯支持的位置正确。应同时看颜色、累计不透明度和可见性。</p>
<figure><a href="body-boundary-current/alpha-coverage.jpg"><img src="body-boundary-current/alpha-coverage.jpg" alt="原图、目标人物遮罩、学习覆盖、红色多余与蓝色缺失"></a><figcaption>第0观察视图：红色表示轮廓外过量覆盖，蓝色表示目标内部覆盖缺失。目标来自分割，不能当作物理透明度真值。</figcaption></figure>
<h3>已有证据能说明什么</h3>
<p>白边在原生渲染中也存在；位置、尺度、旋转与透明度均有非零轮廓梯度，因此不能概括为边界完全不可学习。原目标把轮廓误差平均到整幅图，冻结末态的7个视图中，相关点上透明度项的位置梯度范数约为图像项的3.3%—4.2%。11像素软轮廓带仅约9%直接进入法线约束，多视图项还要求双侧内部和整块有效。这说明边界与内部接受的监督不同。</p>
<p>这些是末态的原始梯度，不包含优化器预条件，也不能还原整个训练过程。部分跨视图逐点梯度相反，但遮挡与表面切向变化也会如此；还不能把它归结为相机标定错误。</p>
<h3>本轮为什么不算修复完成</h3>
<p>069从20万个新点训练18,000步，最终249,326点，同时改变预乘前景缩放、双背景合成和轮廓归一化。6个已见视图的平均边界覆盖误差从0.12891降至0.09446，但第0视图轮廓内部覆盖不足由4.75%增至13.89%，仍有明显碎条。因此它只能说明组合改动影响了外观与覆盖，不能证明任何单项解决了根因。</p>
<details><summary>查看同一仰视方向的实际结果</summary><figure><a href="body-boundary-current/below25-comparison.jpg"><img src="body-boundary-current/below25-comparison.jpg" alt="064与069同相机仰视，黑灰白背景"></a><figcaption>上064，下069。鞋底与人体重影仍需处理；没有这一视角的实拍真值。</figcaption></figure></details>
<h3>下一轮的依据</h3>
<p>先追踪头后、脸前、衣领、鞋边的实际高斯合成贡献，再分别核查59个已标定相机与116个插值相机的遮挡和轮廓支持，回看原图确认分割是否可靠。若主要是监督被稀释，只做轮廓归一化单项对照；若存在投影或分割冲突，先修输入。看不见的表面仍属于缺少观测，不能以加点或增加权重代替真实约束。动态路线已经废弃。</p>
<div id="clothing-delivery"><h3>本轮点云与加载方式</h3><p><a href="./?model=whole-body-static-boundary">查看069边界对照</a> · <a href="models/whole-body-static-boundary/shuolee_069_s18000_f0100.ply?download=1">下载069标准PLY</a> · <a href="body-boundary-current/analysis.json">诊断依据</a></p>
<p>在00的shuolee查看器点击“打开文件”，采用默认+Y向上，无需人体模板或再次蒙皮。本地已实际验证高斯与点中心两种模式；本轮没有在00部署。格式兼容通过，质量未通过。</p>
<details><summary>完整路径与其他静态对照</summary><p style="overflow-wrap:anywhere">C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\viewer\\models\\whole-body-static-boundary\\shuolee_069_s18000_f0100.ply</p><p><a href="models/whole-body-static-no-scale/shuolee_065_s18000_f0100.ply?download=1">065：取消尺度压缩</a> · <a href="models/whole-body-static-no-multiview/shuolee_066_s18000_f0100.ply?download=1">066：取消多视图约束</a>。两者亦未获得整体质量改善。每轮独立从头训练。</p><p>标准字段、位置/完整协方差/球谐旋转及本地重载均已核验；+Y旋转后的少量浮点深度排序例外单独记录，不宣称逐像素完全一致。</p><a href="models/whole-body-static-boundary/delivery.json">完整导出核验</a></details></div>
</section><details><summary>此前静态表面约束的原理与效果</summary>'''+old+'''</details>
<!-- CLOTHING_PIPELINE_CURRENT_END -->'''
t=t[:s]+new+t[e:];t=t.replace('>此前动态实验</a>','>动态历史（已停止）</a>')
page.write_text(t,encoding='utf8')
index=V/'index.html';t=index.read_text(encoding='utf8');t=t.replace('<a href="body-dynamic.html" target="_blank">动态人体＋背景 · 1.8万步效果 ↗</a>','<a href="body-smplx.html#boundary-analysis" target="_blank">静态边界 · 问题分析与对照 ↗</a>');index.write_text(t,encoding='utf8')
meta=read(V/'model-whole-body-static-boundary.json');meta['status']='069边界组合对照：白壳减轻，但碎条及局部缺失仍在，未通过质量验收。';meta['report']='body-smplx.html#boundary-analysis';save(V/'model-whole-body-static-boundary.json',meta)
print(json.dumps(dict(coverage_means=means,gradient_ratio_range=[min(ratios),max(ratios)],page='http://127.0.0.1:8870/body-smplx.html#boundary-analysis',quality_accepted=False),ensure_ascii=False))
