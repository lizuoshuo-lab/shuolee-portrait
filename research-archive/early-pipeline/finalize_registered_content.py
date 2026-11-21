import json,hashlib,shutil,datetime
from pathlib import Path
import numpy as np
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee');V=B.parent/'viewer'
R=B/'current/person-static-registered-camera-fresh-072';O=B/'current/pixel-mapping-audit-073'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8');t.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
now=datetime.datetime.now().astimezone().isoformat(timespec='seconds')
n=read(R/'review/same-camera-result.json');a=read(O/'audit.json');g=read(O/'ghost-contribution.json')
assert not n['numeric_screening_passed'] and a['raw_video_pixel_parity']==0
decision=dict(created=now,actually_inspected=True,quality_accepted=False,numeric_screening_passed=False,
    inspected_images=['head-000.jpg','head-023.jpg','head-069.jpg','head-099.jpg','head-135.jpg','head-174.jpg',
        'contour-069.jpg','novel-below25-bg0.25.jpg','continuous-060.jpg'],
    findings=['Main head texture and body remain recognizable; original frame100 still has nose/glasses ghosting.',
        'All175 same-camera contour distance improves only2.13 percent; neither10percent primary gate passes.',
        'Unseen below25 still has face/torso/foot duplication and translucency; no promotion to best or00deployment.'],
    viewer_status='072已完成18,000步；格式通过，脸部重影与未知视角质量未通过。',
    report='body-smplx.html#pixel-mapping',
    ppt_effect='全175帧同相机评测：轮廓距离2.192降至2.145像素，内部颜色误差降低11.0%。',
    ppt_summary='逐帧配准与训练已完成，边界和未知视角仍未通过。',
    ppt_limit='175帧像素映射核验通过，局部重影仍在。\n继续追踪错误投影贡献，暂不追加训练或部署00。',
    next_step='Trace high-error Gaussian depth, footprint and cross-view visibility before freezing a new single-variable training protocol.',
    all_gt_claims=False,continuous_video_generated=True,continuous_playback_reviewed=False)
save(R/'review/visual-decision.json',decision)
causal=dict(created=now,diagnosis_completed=True,new_training_ready=False,quality_accepted=False,
    confirmed=['175 original decoded frames reproduce every rectified training pixel exactly.',
        '175 training matrix records equal the per-frame manifest after resize; original59world transforms retained.',
        'Frame100 stable head texture matches have median0.41849px residual at1536H; the ghost is not explained by one global shift.',
        '2770 selected Gaussians contribute to the exterior head region; temporary exclusion reduces its alpha0.46093to0.26550.'],
    not_proven=['Camera/intrinsic estimates are perfectly accurate.', 'A single global translation solves the error.',
        'Real motion is the sole cause.', 'Deleting the diagnostic set is a valid multiview fix.'],
    hypotheses=[dict(name='Frame/crop/resize association error',status='Not observed in full175-frame pixel and matrix checks'),
        dict(name='Residual local camera/intrinsic or observation inconsistency',status='Not excluded; SfM and masks are not external ground truth'),
        dict(name='Erroneous Gaussian geometry, footprint or visibility support',status='Directly implicated by frozen exterior-alpha contribution probe; learning-time cause still unresolved')],
    next_required=['Trace selected exterior contributors through all source views with occlusion-aware depth and silhouette constraints.',
        'Separate center outside from footprint overreach and masked-out RGB support.',
        'Define one bounded from-scratch test only after the diagnosis distinguishes the hypotheses.'],
    model_modified=False,optimizer_updates=0,evidence=[str(O/x) for x in ['audit.json','render-pixel-displacements.json','observed-projection.json','epipolar-real-images.json','ghost-contribution.json']])
save(O/'causal-decision.json',causal)
asset=V/'body-camera-current';asset.mkdir(exist_ok=True)
for src,dst in [(O/'overlay-099.jpg','overlay-100.jpg'),(O/'ghost-contributors-099.jpg','contributors-100.jpg'),
    (R/'review/head-099.jpg','camera-training-head.jpg'),(R/'review/body-069.jpg','camera-training-body.jpg'),
    (R/'review/novel-below25-bg0.25.jpg','camera-training-below25.jpg'),(R/'review/continuous-same-camera.mp4','continuous.mp4')]:shutil.copy2(src,asset/dst)
page=V/'body-smplx.html';html=page.read_text(encoding='utf8')
start=html.index('<p class="notice"><strong>先分析，再迭代。</strong>');end=html.index('<h3 id="camera-registration">',start)
new='''<p class="notice"><strong>072已从头完成18,000步，质量仍未通过。</strong>175帧均使用图像配准相机，最终323,174个高斯。脸部重影持续存在，现阶段先核对映射与错误投影贡献，不追加训练。</p>
<h3 id="pixel-mapping">同一像素位置：实拍、训练渲染与叠加</h3>
<p>把整幅图平移错位与局部多余投影分开检查。以下全部使用原视频第100帧、同一相机和同一个裁片，没有分别居中，也没有为展示平移渲染。主面部纹理已经接近实拍，鼻子、眼镜外侧仍有多余覆盖。</p>
<figure><a href="body-camera-current/overlay-100.jpg"><img src="body-camera-current/overlay-100.jpg" alt="原视频第100帧、072同相机渲染、50%叠加，同像素裁剪且没有额外配准"></a><figcaption>左：真实图像的人物区域。中：072原生渲染。右：50%叠加。灰背景用于看清边界，没有改变模型、相机或像素位置。</figcaption></figure>
<p>重新顺序解码原视频后，175帧分别按原相机参数去畸变，结果与实际训练图逐像素一致，最大8位像素差为0；对应mask尺寸和哈希、训练帧号、全图缩放及175组训练矩阵也逐项核验。第100帧可靠头部纹理匹配的位移中位数为0.42个训练像素，不能用统一平移解释外围重影。这些核验排查输入与投影实现，不证明估计相机或真实三维几何完全正确。</p>
<h3>多余高斯确实参与了脸外重影</h3>
<p>冻结模型后，对头部轮廓外3—30像素内的覆盖计算不透明度梯度，定位到2,770个主要贡献点。仅在内存中临时隐藏这组点，区域平均不透明度从0.461降到0.265。重影有所减轻，仍没有完全消失。这是一次定位诊断，未修改交付点云。</p>
<figure><a href="body-camera-current/contributors-100.jpg"><img src="body-camera-current/contributors-100.jpg" alt="原始渲染、重影贡献点单独渲染、临时屏蔽后的诊断图"></a><figcaption>这组点在其他方向可能参与有效人物覆盖。不能把单视角隐藏当成修复，也不能仅由末态梯度断定训练早期原因。</figcaption></figure>
<p>下一步逐个追踪这些点的深度、投影范围和跨视图遮挡，区分点中心位置错误、过大的覆盖范围与观测冲突。相机局部误差、mask边缘误差和实拍微动仍是待区分的因素。继续共享静态模型，不恢复动态网络。</p>
<p><a href="./?model=latest">打开最新一轮（自动更新）</a> · <a href="models/whole-body-static-registered-camera/shuolee_072_s18000_f0100.ply?download=1">下载072标准PLY</a>。最新一轮不等于最佳结果；本轮未部署00。</p>
<details><summary>072训练效果、同相机比较与连续视角</summary>
<p>两模型均使用071的同一套175个相机评测。070到072的平均轮廓距离2.192降至2.145像素，边界不透明度误差0.19695降至0.19238，均未达到预先固定的10%改善要求。内部颜色误差0.00767降至0.00683。全部为训练观测上的诊断，没有未知方向的真实图像作GT。</p>
<figure><img src="body-camera-current/camera-training-head.jpg" alt="实拍、070、072，同相机头部比较"></figure>
<figure><img src="body-camera-current/camera-training-body.jpg" alt="实拍、070、072，同相机全身比较"></figure>
<figure><img src="body-camera-current/camera-training-below25.jpg" alt="070与072在同一未知仰视方向的实际效果"></figure>
<video id="registered-continuous" controls preload="metadata" src="body-camera-current/continuous.mp4"></video>
</details>
'''
html=html[:start]+new+html[end:]
html=html.replace('训练入口现在拒绝插值相机、错配帧、无效旋转及未审阅的配准数据。下一轮先冻结相机单项对照协议，再重算人物粗范围、初始化20万个新高斯和优化器；不会接着旧模型训练，也不会同时改变轮廓权重。','训练入口拒绝插值相机、错配帧、无效旋转及未审阅配准数据。072已按预先固定协议重新拟合粗范围、初始化20万个高斯与优化器并完成18,000步，保持070损失、采样顺序和初始化投票权重。相机改变所引起的新粗范围、点采样和邻居选择属于下游变化，不能称为固定初始点的位姿消融。')
page.write_text(html,encoding='utf8')
status=read(B/'work-status.json');status['updated']=now
status['current_phase']='072 complete;073 pixel/geometry contribution diagnosis, no new training'
status['latest_local_execution']=dict(running=False,state='diagnosis_complete_followup_required',directory=str(O),training_steps=0)
status['camera_registration']['gaussian_retrained']=True;status['camera_registration']['retraining_round']='072'
status['latest_completed_static_delivery']=dict(round='072',steps=18000,points=323174,quality_accepted=False,
    compatibility_passed=True,local_viewer_passed=False,pc00_opened_this_round=False,
    viewer_url='http://127.0.0.1:8870/?model=latest',
    download_url='http://127.0.0.1:8870/models/whole-body-static-registered-camera/shuolee_072_s18000_f0100.ply?download=1',
    local_ply=str(V/'models/whole-body-static-registered-camera/shuolee_072_s18000_f0100.ply'),training_cameras='59 SfM +116 registered,0interpolated',review=str(R/'review/visual-decision.json'))
status['iteration_policy'].update(new_training_ready=False,read_only_diagnosis=str(O/'causal-decision.json'),required_next_protocol='occlusion-aware attribution before choosing a single training change')
status['viewer_update_policy']=dict(automatic=True,url='http://127.0.0.1:8870/?model=latest',poll_seconds=15,after='verified complete per-round export; publish model/cameras/download/review together',latest_is_best=False)
save(B/'work-status.json',status)
plan=B/'long-term-plan.md';text=plan.read_text(encoding='utf8');p0=text.index('<!-- CAMERA_REGISTRATION_CURRENT_BEGIN -->');p1=text.index('<!-- CAMERA_REGISTRATION_CURRENT_END -->')+len('<!-- CAMERA_REGISTRATION_CURRENT_END -->')
text=text[:p0]+'''<!-- CAMERA_REGISTRATION_CURRENT_BEGIN -->
071已补齐全部175帧观测相机（59原SfM、116背景EPnP-RANSAC及LM，0插值），072从头重新拟合粗范围、初始化20万点与新优化器，完成18k，323174点。全部175相机一致评测：070/072轮廓2.191883/2.145189训练像素，边界alpha0.196945/0.192376，未通过预先10%改善要求；未知仰视、鞋底与脸部重影仍未通过，不晋升、不部署00。
073只读映射核查：175帧按原视频实际帧号独立解码、去畸变后，与训练图逐像素一致；训练K/E与清单和缩放一致。原第100帧可靠面部纹理残差中位0.41849训练像素，重影不能只用整幅图统一偏移解释。冻结模型中2770个主要脸外贡献点临时屏蔽后，区域alpha0.46093降到0.26550，但仍有重影；不保存裁掉点的新模型，不称修复。相机局部精度、mask语义边界、真实微动与高斯深度/覆盖/遮挡须继续区分。
下一步先追踪错误贡献点跨视图的深度与实际遮挡，区分中心位置、覆盖范围和观测冲突，获得证据后才冻结新单变量从头训练协议。禁止自动追加步数或扫权重。证据current/pixel-mapping-audit-073/causal-decision.json。右侧?model=latest每15秒自动检查每轮已通过导出核验的模型、实际相机、下载与评测链接，保留明确质量状态与手动历史比较。
<!-- CAMERA_REGISTRATION_CURRENT_END -->'''+text[p1:];plan.write_text(text,encoding='utf8')
print(json.dumps(dict(status='072 content updated;await viewer and presentation verification',diagnostic=str(O)),ensure_ascii=False))
