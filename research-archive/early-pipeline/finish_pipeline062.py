"""Publish the reviewed static whole-person round without altering older models."""
import datetime,html,json,shutil,hashlib
from pathlib import Path
from publish_round_delivery import main as publish

B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
R=B/'current/person-static-world-fresh-062';V=B.parent/'viewer';P=V/'body-static-current'
REPO=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/repos/GaussianAvatars_SMPLX-shuolee')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))

def save(p,d):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8');temp.replace(p)

result=read(R/'gaussian-run/result.json');audit=read(R/'gaussian-run/export-audit/audit.json')
config=read(R/'gaussian-run/config.json');review=read(R/'visual-review.json')
assert result['steps']==18000 and result['from_scratch'] and result['frames_used']==175
assert config['representation']=='shared_static_world' and not config['body_deformation_used']
assert audit['state']=='passed' and audit['sources_unchanged'] and audit['static_time_invariance']['frames_checked']==175
assert review['actually_inspected']
publish(R,'062','whole-body-static-fresh','062 · 全身静态 · 1.8万步')
public=read(V/'models/whole-body-static-fresh/delivery.json')
conclusion=dict(created=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
    state='static_training_and_compatible_export_completed_quality_under_review',
    representation='shared_static_world',all_frames_share_xyz_and_covariance=True,
    body_deformation_used=False,clothing_temporal_deformation_used=False,
    from_scratch=True,old_subject_fit_loaded=False,old_gaussian_loaded=False,old_optimizer_loaded=False,
    real_frames=175,calibrated_cameras=59,interpolated_cameras=116,
    fresh_body_fit_use='Only coarse initialization bounds and scale; no skinning or attraction to the naked template.',
    initial_gaussians=200000,gaussian_steps=result['steps'],final_gaussians=result['points'],
    frame_update_min=min(result['per_frame_updates']),frame_update_max=max(result['per_frame_updates']),
    static_time_invariance=audit['static_time_invariance'],visual_review=review,
    compatibility_passed=True,geometry_accepted=False,quality_accepted=False,pc00_opened_this_round=False,
    download_url=public['download_url'],local_ply=public['download_path'],sha256=public['sha256'],
    comparison='Same cameras and common world as060. Static shared reconstruction is not a matched dynamic pose; representation and fresh initialization both change.',
    next_experiment=review['next_experiment'])
save(R/'phase-conclusion.json',conclusion)
P.mkdir(exist_ok=True)
for src,dst in {
    'clothing-init/initialization-review.jpg':'initialization.jpg',
    'gaussian-run/review-current.jpg':'review-current.jpg','gaussian-run/novel-current.jpg':'novel-current.jpg',
    'gaussian-run/progress.json':'progress.json','gaussian-run/export-audit/audit.json':'export-audit.json',
    'gaussian-run/export-audit/center-distribution.jpg':'center-distribution.jpg',
    'gaussian-run/export-audit/constant-color-views.jpg':'constant-color-views.jpg',
    'gaussian-run/export-audit/same-camera-060-062.jpg':'same-camera-060-062.jpg',
    'phase-conclusion.json':'conclusion.json'
}.items():shutil.copy2(R/src,P/dst)

count=f'{result["points"]:,}';notes=html.escape(review['summary_zh']);local=html.escape(public['download_path'])
filename=Path(public['download_path']).name
block=f'''<!-- CLOTHING_PIPELINE_CURRENT_BEGIN -->
<section id="pose-clothing-current">
<h2>全身静态重建：所有实拍视角共同约束一个穿衣人物</h2>
<p>按当前要求，先将整个穿衣人物建成一个静态三维模型。相机随拍摄位置变化，人物高斯的位置、尺度、旋转和不透明度在所有帧间共享；颜色仍可通过球谐函数随观察方向变化。没有按帧身体驱动或衣物位移场。</p>
<math display="block"><mrow><msub><mi>μ</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub><mo>=</mo><msub><mi>μ</mi><mi>i</mi></msub><mo>,</mo><mspace width="1em"/><msub><mi>Σ</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub><mo>=</mo><msub><mi>Σ</mi><mi>i</mi></msub><mo>,</mo><mspace width="1em"/><msub><mi>α</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub><mo>=</mo><msub><mi>α</mi><mi>i</mi></msub></mrow></math>
<p><var>i</var>为高斯索引，<var>t</var>为实拍帧索引；<var>μ</var>是世界坐标中心，<var>Σ</var>是协方差，<var>α</var>是不透明度。每帧用各自标定或插值的相机，将同一组三维参数投影到对应图像。新增视角用于约束共同几何，不再拟合每帧不同的人体形状。</p>
<h3 id="clothing-motion-check">初始化也使用全部有效观察</h3>
<p>从原图和相机重新开始，用175帧人物轮廓构造静态轮廓体，在其边界及原图三角化特征附近放置20万个新高斯。新拟合的人体模板仅给出粗范围与尺度，没有裸模吸附、每三角面保点或蒙皮。最终点可移动到外套、裤子、头发和鞋附近。</p>
<figure><img src="body-static-current/initialization.jpg" alt="062全身静态高斯初始化，六个原相机"><figcaption>本轮实际初始化渲染，尚未进行高斯训练。轮廓体是初始化依据，不是真实衣物表面测量。</figcaption></figure>
<p class="notice">062轮从20万个新点完成18,000步，最终{count}个高斯。175帧均参与，每帧更新{min(result['per_frame_updates'])}—{max(result['per_frame_updates'])}次。59个相机来自原始标定，116个按时间插值；插值相机和遮罩仍有误差。没有加载旧高斯或旧优化器。</p>
<h3>训练图像与未拍摄方向分别检查</h3>
<figure><img src="body-static-current/review-current.jpg" alt="062六个实际相机中的输入与18000步渲染"><figcaption>每对左侧为实拍，右侧为渲染。这些帧参与了训练。</figcaption></figure>
<figure><img src="body-static-current/novel-current.jpg" alt="同一个062静态模型的正面侧面俯仰渲染"><figcaption>切换相机不改变模型。侧面、俯视和仰视用于查找重影与破碎，未拍摄方向没有实拍真值。</figcaption></figure>
<p>{notes}</p>
<div id="clothing-centers"><h3>同相机比较与实际点中心</h3>
<figure><img src="body-static-current/same-camera-060-062.jpg" alt="060与062在同一世界坐标相机下的成对比较"><figcaption>每对左060身体驱动的固定时刻，右062共享静态模型。相机和世界坐标相同；两者表示方式和初始化不同，不能称为同姿态的单一损失消融。</figcaption></figure>
<details><summary>查看实际高斯中心与统一灰色渲染</summary><figure><img src="body-static-current/center-distribution.jpg" alt="062真实高斯中心分布"><figcaption>透明度大于0.05的中心，每点一个像素，不使用高斯覆盖和颜色纹理。</figcaption></figure><figure><img src="body-static-current/constant-color-views.jpg" alt="062统一灰色高斯渲染"></figure></details></div>
<div id="clothing-delivery"><h3>062静态点云与00查看器</h3>
<p><a href="./?model=whole-body-static-fresh">打开062静态模型</a> · <a href="models/whole-body-static-fresh/{filename}?download=1">下载062兼容版PLY</a> · <a href="./?model=whole-body-clothing-fresh">060对照</a></p>
<p>文件已统一为+Y向上，中心、完整协方差和全部球谐系数一起转换。在00桌面“工作／shuolee”查看器点击“打开文件”，选择本PLY；竖直方向使用默认或+Y，不加载人体模板、骨架或额外形变。文件名中的f0100仅标识查看相机参考，整个文件是全部175帧共同拟合的静态模型。</p>
<details><summary>完整本地路径和核验范围</summary><p style="overflow-wrap:anywhere">{local}</p><p>175个帧索引下的高斯位置与协方差逐位相同；八相机的原生、重载PLY以及+Y坐标转换核验通过。格式与重建质量分开验收。本轮尚未在00远端打开验证。</p><a href="models/whole-body-static-fresh/delivery.json">导出记录</a></details></div>
<h3>下一步仍以静态整体为主</h3><p>先定位同一纹理在不同帧中的相机重投影偏差、真实微动与遮罩误差，使用可信实拍约束静态几何。静态表示减少错误形变自由度，但不能凭空恢复未观察的表面；未拍摄的俯仰方向继续只作诊断。静态整体验收前不恢复动态分支。</p>
</section>
<!-- CLOTHING_PIPELINE_CURRENT_END -->'''
p=V/'body-smplx.html';text=p.read_text(encoding='utf8');a=text.index('<!-- CLOTHING_PIPELINE_CURRENT_BEGIN -->');b=text.index('<!-- CLOTHING_PIPELINE_CURRENT_END -->')+len('<!-- CLOTHING_PIPELINE_CURRENT_END -->');p.write_text(text[:a]+block+text[b:],encoding='utf8')
status=read(B/'work-status.json');status.update(updated=conclusion['created'],local_training_running=False,
    current_phase='static_whole_person_062_trained_and_exported',current_decision=review['summary_zh'],
    user_priority='Static clothed whole person first; shared geometry across all views; dynamic modeling deferred.')
status['latest_local_execution']=dict(kind='person_static_world_fresh_062',state=conclusion['state'],running=False,directory=str(R),
    representation='shared_static_world',steps=18000,initial_points=200000,final_points=result['points'],from_scratch=True,
    geometry_accepted=False,quality_accepted=False,review_url='http://127.0.0.1:8870/body-smplx.html#pose-clothing-current',
    candidate_url=public['viewer_url'],download_url=public['download_url'])
save(B/'work-status.json',status)
plan=B/'long-term-plan.md';text=plan.read_text(encoding='utf8');a=text.index('062正在从头运行当前优先的全身静态实验：');b=text.index('\n<!-- END_GVHMR_059_ACTIVE -->',a)
text=text[:a]+f'062已完成当前优先的全身静态实验：175帧共同拟合，20万新点训练18,000步，最终{count}点。175个帧索引下位置与协方差逐位相同。+Y向上独立PLY通过原生/重载和坐标转换一致性检查，已提供{filename}。{review["summary_zh"]}下一项：{review["next_experiment"]}静态整体未验收前不恢复动态；未拍摄视角不作为真值。'+text[b:];plan.write_text(text,encoding='utf8')
freeze=R/'source-final';freeze.mkdir(exist_ok=True);manifest={}
for name in ['train_clothing_cloud.py','initialize_static_clothing_cloud.py','scene/static_world_gaussian_model.py','compatible_gaussian_export.py','export_viewer_upright.py','audit_clothing_export.py']:
    dst=freeze/Path(name).name;shutil.copy2(REPO/name,dst);manifest[name]=hashlib.sha256(dst.read_bytes()).hexdigest()
assert manifest['train_clothing_cloud.py']==config['source_code_sha256']
assert manifest['scene/static_world_gaussian_model.py']==config['model_code_sha256']
save(freeze/'manifest.json',dict(created=conclusion['created'],files=manifest,training_hashes_verified=True))
print(json.dumps({'points':result['points'],'download':public['download_url'],'quality_accepted':False},ensure_ascii=False))
