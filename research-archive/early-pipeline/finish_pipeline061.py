"""Publish the reviewed fresh clothing-motion round and its verified PLY."""
import datetime,html,json,shutil
from pathlib import Path
from publish_round_delivery import main as publish

B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
R=B/'current/person-clothing-motion-fresh-061';V=B.parent/'viewer';P=V/'body-clothing-motion-current'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,data):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8');temp.replace(p)

result=read(R/'gaussian-run/result.json');audit=read(R/'gaussian-run/export-audit/audit.json')
delivery=read(R/'gaussian-run/delivery.json');control=read(R/'clothing-init/control-fit.json')
review=read(R/'visual-review.json')
assert result['steps']==18000 and result['from_scratch'] and audit['state']=='passed' and audit['sources_unchanged']
assert delivery['compatibility_passed'] and review['actually_inspected']
publish(R,'061','whole-body-clothing-motion','061 · 观测约束衣物运动 · 1.8万步候选')
public=read(V/'models/whole-body-clothing-motion/delivery.json')
conclusion=dict(created=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
    state='fresh_training_and_compatible_export_completed_quality_unaccepted',
    from_scratch=True,old_subject_fit_loaded=False,old_gaussian_loaded=False,old_optimizer_loaded=False,
    real_frames=175,calibrated_cameras=59,interpolated_cameras=116,geometry_steps=1200,
    initial_gaussians=200000,gaussian_steps=result['steps'],final_gaussians=result['points'],
    frame_update_min=min(result['per_frame_updates']),frame_update_max=max(result['per_frame_updates']),
    controls=dict(nodes=control['nodes'],knots=control['time_knots'],held_tracks=control['held_tracks'],
        before=control['before']['held_tracks'],after=control['after']['held_tracks'],
        max_model_m=control['maximum_node_displacement_model_m']),
    visual_review=review,compatibility_passed=True,geometry_accepted=False,quality_accepted=False,
    pc00_opened_this_round=False,download_url=public['download_url'],local_ply=public['download_path'],
    sha256=public['sha256'],source_world_ply_sha256=delivery['sha256'],comparison='Same exported pose99 and cameras as060; fresh hull changes with motion, not a pure single-loss ablation.',
    next_experiment=review['next_experiment'])
save(R/'phase-conclusion.json',conclusion)
P.mkdir(exist_ok=True)
for src,dst in {
    'clothing-init/initialization-review.jpg':'initialization.jpg',
    'gaussian-run/review-current.jpg':'review-current.jpg','gaussian-run/novel-current.jpg':'novel-current.jpg',
    'gaussian-run/progress.json':'progress.json','gaussian-run/export-audit/audit.json':'export-audit.json',
    'gaussian-run/export-audit/center-distribution.jpg':'center-distribution.jpg',
    'gaussian-run/export-audit/constant-color-views.jpg':'constant-color-views.jpg',
    'gaussian-run/export-audit/same-pose-060-061.jpg':'same-pose-060-061.jpg',
    'phase-conclusion.json':'conclusion.json'
}.items():shutil.copy2(R/src,P/dst)

count=f'{result["points"]:,}';notes=html.escape(review['summary_zh']);local=html.escape(public['download_path'])
filename=Path(public['download_path']).name
block=f'''<!-- CLOTHING_PIPELINE_CURRENT_BEGIN -->
<section id="pose-clothing-current">
<h2>人体姿态与穿衣表面：分别优化，分别验收</h2>
<p>SMPL-X提供身体骨架和运动参照；外套、裤子、头发和鞋是最终可见表面。高斯位置可以离开裸人体，不再要求每个身体三角面保留点，也不使用将所有高斯拉回皮肤的约束。</p>
<h3 id="clothing-motion-check">衣物运动：由真实观测补充身体驱动</h3>
<p>身体动作预测不一致时，直接让衣服服从身体运动，会把同一纹理移动到错误位置。本轮用原图三角化的衣物纹理支持64个空间控制节点、16个时间结点；只在观测支持附近补充位移。人体姿态仍来自重新进行的175帧联合拟合，没有加载060的个体参数。</p>
<math display="block"><mrow><msub><mi>μ</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub><mo>=</mo><msub><mi>A</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub><msub><mi>μ</mi><mi>i</mi></msub><mo>+</mo><msub><mi>b</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub><mo>+</mo><munder><mo>∑</mo><mi>k</mi></munder><msub><mi>w</mi><mrow><mi>i</mi><mi>k</mi></mrow></msub><msub><mi>δ</mi><mrow><mi>k</mi><mo>,</mo><mi>t</mi></mrow></msub></mrow></math>
<p><var>μ<sub>i</sub></var>是第<var>i</var>个高斯的参考空间中心；<var>A</var>、<var>b</var>来自身体运动；<var>δ</var>是控制节点位移，<var>w</var>由空间距离和观测覆盖确定。在没有衣物观测的位置，位移权重衰减。每个高斯的权重在显式更新之间固定，因此局部仿射近似的协方差为：</p>
<math display="block"><mrow><msub><mi>Σ</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub><mo>=</mo><msub><mi>A</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub><msub><mi>Σ</mi><mi>i</mi></msub><msup><msub><mi>A</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub><mi>T</mi></msup></mrow></math>
<p>这项本地适配参考自由参考空间高斯与人体驱动的表示方式；<a href="https://github.com/skhu101/GauHuman">GauHuman</a>、<a href="https://github.com/JiahuiLei/GART">GART</a>是相关开源参考，不是本轮已完整复现的方法。45条未参与控制拟合的纹理，中位偏差54.44→7.82像素，但90分位仍为43.97像素。这些点按静态一致性筛选，不能代表全部衣物运动。</p>
<h3>从衣物轮廓初始化20万点，再训练外观与空间分布</h3>
<figure><img src="body-clothing-motion-current/initialization.jpg" alt="061轮六个相机下的新衣物高斯初始化"><figcaption>实际初始化渲染，0次高斯更新；不是最终衣物表面真值。</figcaption></figure>
<p class="notice">061轮已完成18,000步，从20万个新高斯训练到{count}个高斯。175个有效实拍帧均参与；59个相机原始标定，116个按实际时间插值。兼容性核验通过，人体姿态与新视角质量尚未通过。</p>
<figure><img src="body-clothing-motion-current/review-current.jpg" alt="061轮六个真实相机的实拍与18000步渲染"><figcaption>每对左实拍、右渲染。这是参与训练的图像，不能作为未知视角准确率。</figcaption></figure>
<figure><img src="body-clothing-motion-current/novel-current.jpg" alt="061轮固定姿态的七个新视角"><figcaption>固定同一时刻，检查侧面、俯视与仰视。没有这些角度的实拍真值。</figcaption></figure>
<p>{notes}</p>
<div id="clothing-centers"><h3>检查高斯中心，而不是只看颜色</h3>
<figure><img src="body-clothing-motion-current/center-distribution.jpg" alt="061轮实际高斯中心在八个固定相机中的分布"><figcaption>仅显示透明度大于0.05的中心，每点一个像素；不使用椭球覆盖或球谐纹理。</figcaption></figure>
<details><summary>统一灰色与060同姿态、同相机对照</summary><figure><img src="body-clothing-motion-current/constant-color-views.jpg" alt="061统一灰色高斯渲染"></figure><figure><img src="body-clothing-motion-current/same-pose-060-061.jpg" alt="060与061同一人物姿态和八个相机的成对结果"><figcaption>每对左060、右061。位移场也改变了从头构建的轮廓体初始化，因此不是只修改一个损失项的消融。</figcaption></figure></details></div>
<div id="clothing-delivery"><h3>每轮点云文件与00查看器</h3>
<p><a href="./?model=whole-body-clothing-motion">打开061高斯与点云</a> · <a href="models/whole-body-clothing-motion/{filename}?download=1">下载061兼容版PLY</a> · <a href="models/whole-body-clothing-fresh/shuolee_060_s18000_f0100.ply?download=1">下载060对照PLY</a></p>
<p>将PLY传到00后，在桌面“工作／shuolee”的查看器点击“打开文件”。该文件已经包含原视频第100帧的完整世界坐标形态，统一为+Y向上，位置、协方差与球谐颜色同时转换。竖直方向选择“+Y向上”或默认，不需要加载SMPL-X或再次进行骨架变换。可切换高斯渲染与点云模式。单个PLY只表示一个时刻。</p>
<details><summary>完整本地路径和导出验证</summary><p style="overflow-wrap:anywhere">{local}</p><p>标准3DGS字段：对数标准差、wxyz四元数、logit不透明度、标准SH顺序。八个相机下原生渲染与重载PLY一致性检查通过；格式兼容不代表三维人体姿态正确。本轮没有在00远端重新打开验证。</p><a href="models/whole-body-clothing-motion/delivery.json">导出记录</a></details></div>
</section>
<!-- CLOTHING_PIPELINE_CURRENT_END -->'''
p=V/'body-smplx.html';text=p.read_text(encoding='utf8')
start=text.index('<!-- CLOTHING_PIPELINE_CURRENT_BEGIN -->');end=text.index('<!-- CLOTHING_PIPELINE_CURRENT_END -->')+len('<!-- CLOTHING_PIPELINE_CURRENT_END -->')
p.write_text(text[:start]+block+text[end:],encoding='utf8')
status=read(B/'work-status.json');status.update(updated=conclusion['created'],local_training_running=False,
    current_phase='clothing_motion_061_reviewed_compatible_ply_available',current_decision=review['summary_zh'])
status['latest_local_execution']=dict(kind='person_clothing_motion_fresh_061',state=conclusion['state'],running=False,directory=str(R),
    steps=18000,initial_points=200000,final_points=result['points'],from_scratch=True,geometry_accepted=False,quality_accepted=False,
    review_url='http://127.0.0.1:8870/body-smplx.html#pose-clothing-current',candidate_url=public['viewer_url'],download_url=public['download_url'])
save(B/'work-status.json',status)
plan=B/'long-term-plan.md';text=plan.read_text(encoding='utf8')
start=text.index('061当前：');end=text.index('\n\n每轮高斯结果',start)
text=text[:start]+f'061已完成：175帧重新拟合、20万新点、18,000步，最终{count}点。观测约束衣物运动以64个空间节点、16个时间结点表示；45条留出纹理中位残差54.44→7.82px，90分位仍43.97px。{review["summary_zh"]}世界坐标PLY导出与八相机重载检查通过；本轮未在00远端打开。下一项：{review["next_experiment"]}'+text[end:]
text=text.replace('061等待训练和导出检查完成。',f'061已提供{filename}，两轮独立文件不覆盖。')
plan.write_text(text,encoding='utf8')
print(json.dumps({'points':result['points'],'download':public['download_url'],'quality_accepted':False},ensure_ascii=False))
