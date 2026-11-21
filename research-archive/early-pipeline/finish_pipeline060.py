"""Publish verified experiment evidence without promoting unaccepted geometry."""
from pathlib import Path
import json,shutil,hashlib,re,datetime
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
R=B/'current/person-clothing-pipeline-fresh-060';V=B.parent/'viewer';P=V/'body-clothing-current'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
result=read(R/'gaussian-run/result.json');audit=read(R/'gaussian-run/export-audit/audit.json')
assert result['steps']==18000 and audit['state']=='passed' and audit['sources_unchanged']
conclusion=dict(created=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
    state='clothing_distribution_changed_pose_and_novel_views_unaccepted',from_scratch=True,
    real_frames=175,raw_decoded_frames=177,calibrated_cameras=59,interpolated_cameras=116,
    geometry_steps=1200,initial_gaussians=200000,gaussian_steps=18000,final_gaussians=result['points'],
    minimum_frame_updates=min(result['per_frame_updates']),maximum_frame_updates=max(result['per_frame_updates']),
    body_pose_accepted=False,clothing_surface_accuracy_accepted=False,overall_quality_accepted=False,
    findings=[
        'All 175 valid decoded frames participated; no old subject fit or Gaussian checkpoint initialized this experiment.',
        'Free world-coordinate Gaussian centers form coat/sleeve/trouser outlines, confirmed without SH or splat footprints.',
        'Native rendering and reloaded world PLY agree within one 8-bit code value in eight tested cameras.',
        'Fixed-side knee pose remains questionable; lower 2D fitting error is not anatomical validation.',
        'Novel side and low views retain sleeve/hem/trouser artifacts; this is not a best-model promotion.'
    ],
    next_experiment='Audit observed landmark-to-joint semantics and final world-motion consistency, then compare a fresh body-motion/control run with cloth-aware deformation. Do not retune the old checkpoint.',
    preservation='Keep current candidate and previous 059 control, source inputs and licensed weights. No deployment to00.',
    training_ply_sha256=result['world_ply_sha256'])
motion=read(R/'motion-audit/audit.json') if (R/'motion-audit/audit.json').exists() else None
if motion:
    conclusion['motion_field_check']=dict(tracks=motion['input_tracks'],observations=motion['observations'],
        static_median_px=motion['static_reprojection_px']['median'],body_driven_median_px=motion['body_driven_reprojection_px']['median'],
        selection_bias='Preselected static-consistent tracks; not evidence that every body region is static.',
        result='Body-derived deformation contradicts these observed garment features.')
    conclusion['next_experiment']='Fresh body/clothing motion separation with observed-track support. Do not force every garment point to follow skin motion or use old individualized states as initialization.'
save(R/'phase-conclusion.json',conclusion)
for src,dst in {
    'gaussian-run/review-current.jpg':'review-current.jpg','gaussian-run/novel-current.jpg':'novel-current.jpg',
    'gaussian-run/progress.json':'progress.json','gaussian-run/export-audit/audit.json':'export-audit.json',
    'gaussian-run/export-audit/center-distribution.jpg':'center-distribution.jpg',
    'gaussian-run/export-audit/constant-color-views.jpg':'constant-color-views.jpg',
    'gaussian-run/export-audit/world-ply-views.jpg':'world-ply-views.jpg','phase-conclusion.json':'conclusion.json'
}.items():shutil.copy2(R/src,P/dst)
model_id='whole-body-clothing-fresh';dest=V/'models'/model_id;dest.mkdir(exist_ok=True)
shutil.copy2(R/'gaussian-run/point_cloud.ply',dest/'point_cloud.ply')
assert sha(dest/'point_cloud.ply')==result['world_ply_sha256']
old=read(V/'model-whole-body-gvhmr-current.json')
meta={k:old[k] for k in ['focus','initialView','cameraSource','cameras','filter']}
for cam in meta['cameras']:
    cam['label']=cam['img_name']+' · 原始标定相机 · 人物固定原视频第100帧'
    cam['split']='original59_calibrated_camera_positions_not_independent_test'
meta.update(label='衣物自由高斯 · 全帧从头训练候选',model=f'models/{model_id}/point_cloud.ply',
    status='20万点初始化，1.8万步完成；已脱离裸模表面分布。姿态与新视角仍未通过。',
    sha256=result['world_ply_sha256'],gaussianCount=result['points'],initialGaussianCount=200000,
    report='body-smplx.html#pose-clothing-current',qualityAccepted=False,deliveryAccepted=False,
    trained=True,steps=18000,fromScratch=True,oldCheckpointLoaded=False,independentTest=False,
    realFramesUsed=175,calibratedCameras=59,interpolatedCameras=116,timeSnapshot=99,rawVideoFrame=100,
    singlePlyIsNotDynamic=True,localAdaptation=True,novelViewReview='body-clothing-current/novel-current.jpg',
    exportAudit='body-clothing-current/export-audit.json')
save(V/f'model-{model_id}.json',meta)
js=V/'viewer.js';s=js.read_text(encoding='utf-8')
if '"'+model_id+'"' not in s:s=s.replace('const BUILTINS = [','const BUILTINS = ["'+model_id+'", ',1)
js.write_text(s,encoding='utf-8')
# Keep the old result available; the new label explicitly says candidate.
html_path=V/'body-smplx.html';s=html_path.read_text(encoding='utf-8')
s=s.replace('训练与审阅进行中。点云分布、姿态准确性、未知视角质量分别验收。',
    '已完成20万点初始化、18000步从头训练，最终132663个高斯；175帧各参与102—103次。衣物外形已进入点中心分布，但固定侧视仍有膝部姿态疑点，衣袖与裤脚仍有新视角拖影。两项质量问题尚未全部解决。')
s=s.replace('可能晚于实时进度。','可能滞后于实时进度。')
insertion='''<div id="clothing-centers">
<h3>去掉颜色后，检查高斯中心是否覆盖衣物</h3>
<figure><img src="body-clothing-current/center-distribution.jpg" alt="八个固定视角下的实际高斯中心分布"><figcaption>实际高斯中心，透明度大于0.05；每个中心仅绘制一个像素，没有球谐颜色或椭球覆盖。衣袖、衣摆与裤腿已有空间分布，仍有散点，不能称为真实衣物表面。</figcaption></figure>
<details><summary>查看统一灰色渲染，排除颜色纹理的影响</summary><figure><img src="body-clothing-current/constant-color-views.jpg" alt="相同高斯形状与透明度的统一灰色渲染"><figcaption>保留椭球与透明度，仅替换颜色。新视角下的拖影依然存在。</figcaption></figure></details>
<p><a href="./?model=whole-body-clothing-fresh">打开本轮候选 · 可切换点云模式</a> · <a href="models/whole-body-clothing-fresh/point_cloud.ply?download=1">下载本轮世界坐标PLY</a> · <a href="./?model=whole-body-gvhmr-current">上一轮059对照</a></p>
<p>这份PLY固定为原视频第100帧的姿态，不包含动态骨架。八个相机下重载核验通过：位置与颜色参数一致，显示差异至多一个8位色阶。这只验证导出完整性，不能证明姿态正确。</p>
</div>'''
if 'id="clothing-centers"' not in s:s=s.replace('<p id="cloth-conclusion">',insertion+'\n<p id="cloth-conclusion">',1)
if motion:
    shutil.copy2(R/'motion-audit/source-000033.jpg',P/'motion-features.jpg')
    shutil.copy2(R/'motion-audit/audit.json',P/'motion-audit.json')
    motion_html='''<div id="clothing-motion-check"><h3>衣服为什么不能直接服从当前身体运动</h3>
<p>对225条预先筛出的稳定衣物纹理、3624次真实观测，保持原空间位置的投影偏差中位数为0.89像素；套用当前身体驱动后为48.66像素。这批特征与身体驱动明显不一致，支持继续区分姿态误差、身体运动和衣物运动。</p>
<figure><img src="body-clothing-current/motion-features.jpg" alt="同一原图中，真实纹理点、静态空间支持与身体驱动后的特征投影"><figcaption>左：原图白圈观测；中：原空间支持的投影；右：身体驱动后的投影，红线连接偏差。全部为实际计算。</figcaption></figure>
<p>这里存在选择偏差：这些点本来按静态一致性筛选，使用的也是参与训练的相机与图像。它只说明当前运动场不适合这批衣物点，不能证明整个人静止，也不能恢复被衣物遮住的骨关节真值。下一轮从头比较观测约束的衣物运动与身体驱动，不继续沿用旧模型修补。</p></div>'''
    if 'id="clothing-motion-check"' not in s:s=s.replace('<p id="cloth-conclusion">',motion_html+'\n<p id="cloth-conclusion">',1)
html_path.write_text(s,encoding='utf-8')
st=read(B/'work-status.json');st.update(updated=conclusion['created'],local_training_running=False,
    current_phase='clothing_free_gaussian_060_reviewed_pose_and_novel_views_pending',
    current_decision='060全175帧重提特征并从头拟合，20万新高斯18k完成。衣物进入自由点中心分布，PLY重载通过；姿态及新视角破碎未通过。继续以原始输入做姿态—衣物运动解耦对照，禁止旧个体断点续训。')
st['latest_local_execution']=dict(kind='person_clothing_pipeline_fresh_060',state=conclusion['state'],running=False,
    directory=str(R),real_frames=175,calibrated_cameras=59,interpolated_cameras=116,steps=18000,initial_points=200000,final_points=result['points'],
    old_person_fit_loaded=False,old_gaussian_loaded=False,old_optimizer_loaded=False,geometry_accepted=False,quality_accepted=False,
    review_url='http://127.0.0.1:8870/body-smplx.html#pose-clothing-current',candidate_url='http://127.0.0.1:8870/?model='+model_id)
save(B/'work-status.json',st)
plan=B/'long-term-plan.md';text=plan.read_text(encoding='utf-8')
start=text.index('<!-- GVHMR_059_ACTIVE -->');end=text.index('<!-- END_GVHMR_059_ACTIVE -->')+len('<!-- END_GVHMR_059_ACTIVE -->')
current='''<!-- GVHMR_059_ACTIVE -->
## 当前执行：全帧姿态与衣物表面分开建模，从头实验

用户指出的两个问题分别验收：人体位姿应正确；最终高斯应形成穿衣外表面。SMPL-X承担运动和体形先验，不是衣物表面，不通过整体膨胀假造衣服。

060已从原视频重新处理175个独立帧（177帧中首尾两帧不在原标定时间范围内）。59帧有原标定，116帧相机按实际时间插值，不是真值。特征、预训练GVHMR预测及全帧1200步身体拟合均重新运行；取消按个体帧号硬写的可见性。姿态侧视仍有膝部疑点，未通过。

060使用真实衣物特征和全帧轮廓体初始化200000个新高斯，18,000步从头训练已完成，最终132663点。175帧各使用102—103次。自由XYZ与身体运动参照分离，取消裸模每个三角面强制保点；位置和协方差使用一致变换。去颜色、仅中心图确认外套/袖子/裤腿已进入点中心分布，但这不是衣物三维真值。新视角衣袖、衣摆和裤脚仍有拖影及破碎，整体未通过。

世界PLY按原视频第100帧导出，八相机原生/重载检查通过；保留为本轮候选，不冒充最佳结果。059保留为旧对照。当前页：http://127.0.0.1:8870/body-smplx.html#pose-clothing-current 。候选：http://127.0.0.1:8870/?model=whole-body-clothing-fresh 。事实记录：current/person-clothing-pipeline-fresh-060/phase-conclusion.json。

下一项先定位误差来源：在原图上核验可见关节的语义映射、遮挡权重与最终世界运动，区分身体姿态误差、衣物非刚性运动和未观察区域。比较身体驱动与衣物独立变形的原图支持，参考GauHuman/GART源码；不得只靠二维重投影误差、训练颜色或者更多点数宣布成功。若现有全身拟合无法支持衣物随时间一致，再从官方初始预测做联合姿态与衣物的受约束对照，不接着060模型修补。

060只读运动核验：225条预选稳定衣物纹理共3624次原图观测，静态空间支持残差中位0.885px，当前身体驱动后48.663px。该比较有静态筛选偏差，不能推出整个人静止，但表明不能将当前身体运动直接施加于全部衣物。下一项优先做观测约束的人体/衣物运动分离，所有新拟合和新高斯仍从头；不要重复仅增加点数或仅优化二维关节的小项扫描。

所有新实验从原始图像/相机/许可模板及官方预训练输出开始；旧个体拟合、高斯、残差场、优化器只能只读对比。原始观测可复用，禁止旧学习结果初始化。一次只保留必要对照和当前候选。先本地，不部署00；眼镜专项停止，不同步E盘。实验通过后更新同一个方法PPT，不另建重复报告。
<!-- END_GVHMR_059_ACTIVE -->'''
text=text[:start]+current+text[end:];plan.write_text(text,encoding='utf-8')
print(json.dumps(dict(candidate=model_id,points=result['points'],sha256=result['world_ply_sha256'],quality_accepted=False),ensure_ascii=False))
