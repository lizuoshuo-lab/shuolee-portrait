"""Publish the measured full-image SfM diagnostic, without adopting its cameras."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shutil

B = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
V = B.parent / 'viewer'
O = B / 'current/full-scene-localization-074'
now = datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def save(p, obj):
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    os.replace(tmp, p)


e = read(O / 'evaluation.json')
s = read(O / 'state.json')
assert s['registered'] == 175 and s['points'] == 102495
assert not e['numeric_screen_passed'] and not e['new_gaussian_training']
metrics = e['results']['colmap_pixel_center']
diagnostic = V / 'body-full-scene-current'
diagnostic.mkdir(exist_ok=True)
for name in ('viewer-full-scene.png', 'face-reprojection-000082.jpg',
             'face-reprojection-000100.jpg', 'evaluation.json',
             'decoder-coordinate-audit.json', 'actual-sparse-reprojection.json',
             'cameras.json'):
    shutil.copy2(O / name, diagnostic / name)

review = dict(updated=now, scope='Sampled diagnostic images and actual preset viewer loading',
    inspected_images=['face-reprojection-000082.jpg', 'face-reprojection-000100.jpg',
                      'body-tracks-025.jpg', 'body-tracks-073.jpg', 'viewer-full-scene.png'],
    conclusion='Full-image reconstruction completed; independent held-out face residuals did not improve. Do not adopt for fresh Gaussian training yet.',
    all_correspondences_manually_verified=False, physical_motion_established=False,
    camera_ground_truth_available=False, camera_estimate_accepted_for_training=False,
    viewer_preset_opened=True, viewer_manual_open_file_tested=False,
    point_cloud_type='Colored sparse SfM, not Gaussian', points_visible=102495,
    old_gaussian_modified=False, new_gaussian_training=False, remote_00_deployed=False)
save(O / 'visual-review.json', review)
e.update(visual_review_pending=False, visual_review_scope='See visual-review.json; sampled only',
         final_decision='Full-scene SfM completed; static-face compatibility failed; no new Gaussian training',
         updated=now)
save(O / 'evaluation.json', e)
shutil.copy2(O / 'evaluation.json', diagnostic / 'evaluation.json')
s.update(updated=now, running=False,
    status='reconstruction_and_independent_evaluation_complete_face_gate_failed',
    camera_estimate_accepted=False, root_cause_established=False,
    evaluation=str(O / 'evaluation.json'), visual_review=str(O / 'visual-review.json'))
save(O / 'state.json', s)
receipt = read(O / 'viewer-delivery.json')
receipt.update(updated=now, viewer_open_test_pending=False, viewer_preset_open_test_passed=True,
    viewer_manual_file_open_tested=False, viewer_test_image=str(O / 'viewer-full-scene.png'),
    viewer_projection_math_check=str(O / 'viewer-projection-math-check.json'),
    all_point_extent_note='Fit all includes every sparse outlier; source points were not trimmed.')
save(O / 'viewer-delivery.json', receipt)
save(V / 'models/full-scene-localization-074/delivery.json', receipt)

block = '''<!-- FULL_SCENE_074_BEGIN -->
<section id="full-scene-localization"><h2>整幅图重新定位：相机与场景的联合重建</h2>
<p class="notice"><strong>175帧全部重新定位，得到102,495个稀疏点；鼻口重影的原因仍未查清。</strong>这轮完成了整场景SfM与核验，没有生成新高斯。独立观测中，背景误差有所下降，面部误差没有改善，因此暂不采用这套相机开启下一轮训练。</p>
<p>按最新要求直接使用整张原图：人物和场景中的纹理都参与匹配。人的局部可能稳定，背景也可能运动或反光；通过匹配检验、两视图几何、多视图三角化和联合束调整判断支持关系，不按“人体／背景”标签直接删点。区域标签只用于后面的误差分析。</p>
<p>本轮从175张完整视频帧重新提取特征，匹配全部15,225个图像对，再求解相机与三维点。旧位姿、旧地图及旧高斯未作为初始值；原焦距只作为可优化的相机初值。最后的全局束调整同时优化位姿、点位置、共享焦距、主点和径向畸变，并报告收敛。全部帧注册成功说明重建连通，不等于真实几何已得到验证。</p>
<p><a href="./?model=full-scene-localization-074">打开本轮整场景点云与175个相机</a> · <a href="models/full-scene-localization-074/shuolee_074_scene_sparse.ply?download=1">下载场景稀疏PLY</a> · <a href="body-full-scene-current/cameras.json">完整相机参数</a></p>
<figure><a href="body-full-scene-current/viewer-full-scene.png"><img src="body-full-scene-current/viewer-full-scene.png" style="max-width:901px" alt="实际查看器加载本轮整场景稀疏点云，包含人物、地面与室内环境"></a><figcaption>本机查看器实际截图。SfM稀疏点只表达被匹配到的纹理位置，不是完整表面或训练后的高斯；原始点没有因展示而删除。相机预设显示去畸变后的针孔投影。</figcaption></figure>
<h3>整图定位改善了哪里</h3>
<p>用967条原图纹理轨迹作独立检查：每条轨迹留出中间连续三分之一观测，用其余观测估计一个静态三维点，再检查留出位置。两套相机使用相同观测与划分，图像跟踪筛选不使用相机重投影残差。这里“留出”指独立纹理轨迹的观测，图像本身仍参加过SfM；它不是未见图像测试或真实相机标定。</p>
<p><strong>误差中位数，单位为原图像素：</strong>背景0.46→0.38；衣服／躯干1.84→1.73；鼻口2.43→2.68；上半脸1.24→1.42；腿鞋1.21→1.22。鼻口90分位也由7.20升至7.76，没有达到预先要求的20%改善门槛。</p>
<figure><a href="body-full-scene-current/face-reprojection-000082.jpg"><img src="body-full-scene-current/face-reprojection-000082.jpg" alt="同一原图82帧鼻口观测与留出重投影，左072相机右074整图相机"></a><figcaption>青色：原图跟踪观测；红色：不用当前连续帧段求出的三维点投影。左为原072相机，右为本轮074相机。两边是同一实拍裁片，未为展示重新配准；不是高斯渲染对照。</figcaption></figure>
<p>第82帧仍有鼻口投影与真实纹理的偏差，第100帧两套相机的投影非常接近。整图重建改善了部分场景定位，却没有直接消除人脸的跨视角冲突。自动纹理跟踪可能受遮挡、反光或漂移影响；这些数值不能证明人物一定运动，也不能单凭它们排除外参误差。</p>
<details><summary>另一观察方向与核验范围</summary>
<figure><img src="body-full-scene-current/face-reprojection-000100.jpg" alt="原视频100帧的相同纹理观测与两套相机留出重投影"><figcaption>同一尺度、同一像素位置，未改变脸部几何来迎合相机。</figcaption></figure>
<p>完整分辨率1520×2704；本轮共享焦距1459.3514像素，主点(762.0972,1351.1146)，径向系数0.0011426。新的相机与三维点由全场景联合求解；仅为了在已有查看器中比较，交付点云整体变换到统一的+Y向上坐标。该显示变换没有参与相机拟合。</p>
<p>对最终相机与102,495个点重新计算全部1,508,077个拟合观测，残差中位数0.837像素、均值1.109像素；这是参与重建的观测，不能代替上面的独立检查。COLMAP保存的旧点误差未随单独BA更新，因此没有把旧字段当最终结果。</p>
<p>FFmpeg与原训练解码路径的RGB值并非完全一致。对六个分布在序列中的时刻核验，匹配位置差中位数均低于0.05像素，且各自正确帧优于相邻帧，未发现帧号错配；不能把两种解码结果说成逐像素相同。导出点云的同相机回读投影最大差0.000805像素；查看器内参投影公式也通过数值检查。以上验证实现与交付，不代表几何精度通过。</p>
<p><a href="body-full-scene-current/evaluation.json">区域留出统计</a> · <a href="body-full-scene-current/decoder-coordinate-audit.json">解码坐标检查</a> · <a href="body-full-scene-current/actual-sparse-reprojection.json">最终稀疏点实际残差</a></p></details>
<h3>下一步如何继续定位原因</h3>
<p>保留整图输入和共享静态模型。先审阅鼻口与眼镜附近的真实对应，比较误差在时间上、局部空间上的一致性，以及实际纹理推断的深度与现有高斯深度之间的偏差，区分相机误差、错误对应、小幅运动和已学几何的补偿。只有证据支持一项明确改动，才从头开始新的高斯训练；不把重影简单归结为点数或训练步数。</p>
<p><a href="./?model=latest">当前高斯对照仍为072（质量未通过）</a>。本轮场景点云已接入本地查看器，未部署00。</p>
</section>
<!-- FULL_SCENE_074_END -->'''
page = V / 'body-smplx.html'
html = page.read_text(encoding='utf8')
a = html.index('<!-- FULL_SCENE_074_BEGIN -->')
b = html.index('<!-- FULL_SCENE_074_END -->', a) + len('<!-- FULL_SCENE_074_END -->')
page.write_text(html[:a] + block + html[b:], encoding='utf8')

note = ('074已完成整图从头SfM：175/175相机、102495个稀疏点，全局BA收敛。人物与背景均参与匹配，'
        '区域标签只作诊断。967条独立纹理轨迹的留出误差中位数：背景0.460→0.380像素，'
        '躯干1.842→1.726，鼻口2.431→2.680，上半脸1.240→1.415，腿鞋1.212→1.218。'
        '面部改善门槛未通过，未采用074相机追加高斯训练；仍不能认定外参或微动是唯一原因。'
        '整场景稀疏PLY及完整相机已接入本地查看器，最新高斯仍为质量未通过的072。'
        '下一步先审阅真实对应的时空残差及三维深度冲突，再提出单一主变量的从头训练协议。')
plan = B / 'long-term-plan.md'
txt = plan.read_text(encoding='utf8')
start = txt.find('074正在从175张完整原始画面重新做SfM：')
assert start >= 0 or note in txt
if start >= 0:
    end = txt.index('\n', start)
    txt = txt[:start] + note + txt[end:]
plan.write_text(txt, encoding='utf8')

status = read(B / 'work-status.json')
status.update(updated=now, current_phase='074 full-image SfM completed; face consistency gate failed; no new Gaussian training',
    current_decision=note)
status['latest_local_execution'] = dict(running=False, state=s['status'], directory=str(O),
    training_steps=0, scope='fresh SfM on all 175 complete raw frames')
status['iteration_policy'].update(new_training_ready=False,
    required_next_protocol='Audit temporal/local consistency of image correspondences and learned geometry versus texture depth before proposing one controlled fresh Gaussian trial')
status['full_scene_localization'].update(registered=175, points=102495, evaluation=str(O / 'evaluation.json'),
    visual_review=str(O / 'visual-review.json'), complete=True, accepted_for_training=False,
    metrics=metrics, delivered_sparse_ply=receipt['delivered_path'],
    viewer='http://127.0.0.1:8870/?model=full-scene-localization-074',
    new_gaussian_training=False, root_cause_established=False)
save(B / 'work-status.json', status)
meta_path = V / 'model-whole-body-static-registered-camera.json'
meta = read(meta_path)
meta.update(status='072鼻口重影未通过；074整图定位已完成，但鼻口误差未改善，尚未替换相机或高斯。',
            report='body-smplx.html#full-scene-localization')
save(meta_path, meta)
pointer = read(V / 'latest-result.json')
assert pointer['round'] == '072'
pointer.update(updated=now, revision=hashlib.sha256(meta_path.read_bytes()).hexdigest(), report=meta['report'])
save(V / 'latest-result.json', pointer)
mp = V / 'model-full-scene-localization-074.json'
m = read(mp)
m.update(status='175帧整图定位完成，102,495个稀疏点。鼻口误差未改善；未采用这套相机训练新高斯。',
         cameraEstimateAcceptedForTraining=False, qualityAccepted=False,
         deliveryAccepted=True, viewerPresetVerified=True)
save(mp, m)
print(json.dumps(dict(published=True, registered=175, points=102495, face_gate=False,
                     training_steps=0, report=str(page)), ensure_ascii=False))
