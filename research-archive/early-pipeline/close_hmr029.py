from pathlib import Path
import datetime
import hashlib
import json
import shutil

LAB = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab')
BASE = LAB / 'outputs/mini-splatting2-mip-depth-001/shuolee'
R28 = BASE / 'current/sam3dbody-calibrated-input-028'
R29 = BASE / 'current/hmr2-single-frame-prior-029'
VIEWER = BASE.parent / 'viewer'
NOW = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def save(p, d):
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


assert not (R29 / 'phase-conclusion.json').exists(), 'Do not repeat closeout'
manifest = read(R28 / 'input-manifest.json')
assert manifest['selected'] == [0, 11, 23, 33, 45, 58]
assert not manifest['inference_executed']
save(R28 / 'visual-review.json', dict(reviewed=NOW, artifact='input-crops.jpg',
     observations=['All six crops show the intended red-jacket subject with head and feet inside the frame.',
                   'Foreground mask follows the intended subject and excludes the visible background people at this review scale.',
                   '512-square preprocessing is a probe; actual authorized checkpoint configuration must be checked again.'],
     preprocessing_accepted=True, neural_inference_executed=False, geometry_accepted=False))
preflight = dict(state='preprocessing_verified_weights_pending', source_frames=manifest['selected'],
                 max_ray_error=max(r['preflight']['ray_max_abs_error'] for r in manifest['records']),
                 max_crop_uint8_error=max(r['preflight']['pixel_max_error_uint8'] for r in manifest['records']),
                 repo_commit=manifest['repo_commit'], dinov3_commit=manifest['dinov3_commit'],
                 neural_inference_executed=False, actual_checkpoint_config_available=False,
                 coordinate_requirement='Predict in original camera meters first. SfM world translations have an unknown metric scale.',
                 gaussian_updates=0, accepted=False)
save(R28 / 'phase-conclusion.json', preflight)
protocol = read(R29 / 'protocol.json')
review = read(R29 / 'review/review-manifest.json')
assert protocol['selected'] == review['source_frames'] == manifest['selected']
assert protocol['previous_subject_state_loaded'] is False
assert protocol['body_fit_updates'] == protocol['gaussian_updates'] == 0
assert sha(R29 / 'prediction.npz') == protocol['prediction_sha256']
j = review['source_frames'].index(23)
knees = {k: {name: values[j] for name, values in d.items() if 'knee' in name}
         for k, d in review['model_joint_bends_degrees'].items()}
summary = dict(state='single_frame_prior_not_adopted', completed=NOW,
     source_frames=protocol['selected'], model='HMR2.0a full SMPL head in GVHMR adaptation',
     new_independent_image_predictions=6, body_fit_updates=0, gaussian_updates=0,
     native_template='SMPL neutral, not SMPL-X; no cross-template parameter substitution',
     raw_video_reference='GVHMR raw output from 019; comparison only, never initialization',
     frame23_model_knee_bend_deg=knees,
     observation='Source overlays retain head, wrist and foot offsets. At frame23 the independent single-image prediction has pronounced knee flexion in the fixed relative side view; it is not selected as a better subject initializer.',
     limitation='No joint GT or simultaneous cameras; knee angles describe model geometry, not subject accuracy. This is the GVHMR adaptation of HMR2, not a reproduction of every 4D-Humans setting.',
     evidence=['review/six-source-comparison.jpg', 'review/frame23-pelvis-centered-geometry.jpg'],
     visual_review='Six-frame contact sheet, full source23/source45 overlays and pelvis-centered frame23 at two matched relative rotations inspected.',
     next='Run authorized SAM3DBody on the six original images when weights are available. Preprocessing and public-example conversion are verified; actual subject prediction is not. Do not increase fitting iterations or start Gaussians from this rejected prior.',
     current_gaussian_replaced=False, remote_deployment=False, quality_accepted=False)
save(R29 / 'phase-conclusion.json', summary)
save(R29 / 'visual-review.json', dict(reviewed=NOW, state=summary['state'], scope=summary['visual_review'],
      observation=summary['observation'], quality_accepted=False))
review['state'] = 'visually_reviewed_not_adopted'
review['visual_review_pending'] = False
save(R29 / 'review/review-manifest.json', review)
save(R29 / 'run-state.json', dict(state=summary['state'], running=False, gaussian_updates=0, body_fit_updates=0))

vposer = Path('D:/GS-SLAM-Data/models/vposer/V02_05')
vr = read(vposer / 'verification.json')
assert sha(vposer / 'snapshots' / vr['checkpoint_name']) == vr['checkpoint_sha256']
status = read(BASE / 'work-status.json')
status.update(updated=NOW, current_phase='hmr2_single_frame_prior_029', local_training_running=False,
              current_decision=summary['observation'], next_action=summary['next'])
status['body_sam_input_028'] = {**preflight, 'report': str(R28 / 'phase-conclusion.json')}
status['body_single_frame_prior_029'] = {**summary, 'report': str(R29 / 'phase-conclusion.json')}
status['latest_local_execution'] = dict(kind='hmr2_single_frame_prior_029', state=summary['state'], running=False,
      directory=str(R29), image_predictions=6, body_fit_updates=0, gaussian_updates=0, accepted=False)
status['body_prior_download'] = dict(model='VPoser V02_05', directory=str(vposer), verified_sha256=vr['checkpoint_sha256'],
      loading='Official source, strict state load, CUDA encoding/decoding, gradients and SMPL-X forward previously verified',
      download_origin='Pinned public mirror; hash verified against mirror LFS, not an official hash')
save(BASE / 'work-status.json', status)

plan = BASE / 'long-term-plan.md'
text = plan.read_text(encoding='utf-8-sig')
old = '下一步准备六视角原图、真实内参与遮罩的SAM3DBody输入，等授权权重到位后先检查原始MHR预测。'
assert text.count(old) == 1
text = text.replace(old, '028已完成六张原图、真实内参与遮罩的官方预处理、射线条件与投影核验；源码固定版本，未改动原有训练环境。真实主体、头脚范围及遮罩已检查。512方形输入仅用于预检，授权权重及其真实配置仍未到位；未运行主体SAM预测。')
anchor = '原始几何通过已知相机的叠图后，才考虑共享世界注册和新的高斯训练。'
assert text.count(anchor) == 1
text = text.replace(anchor, anchor + '\n\n029用已经下载的HMR2.0a权重，在GVHMR适配代码中启用完整SMPL预测头，独立读取0/11/23/33/45/58六张原图和标定内参。严格加载官方网络权重，使用许可SMPL中性模板，未复用旧人物参数、未做SMPL到SMPL-X参数复制。已检查六帧叠图及23帧同相机相对侧视。HMR2在23帧的左右膝弯曲为45.22/49.34度，GVHMR原始预测为13.69/10.53度；仅为模型角度，没有关节GT。单帧预测仍有头、手腕和足部偏移，并出现明显屈膝，不作为更好的初始化候选。这次0次身体拟合、0次高斯更新。保持原查看器，等待已申请的SAM3DBody权重后先核验其原始几何；不重复增加步数或把未验收姿态用于高斯训练。\n')
plan.write_text(text, encoding='utf-8')

# Add compact evidence to the existing collapsed appendix, not another report page.
asset = VIEWER / 'body-fitting-method/body-prior-reference'
asset.mkdir(exist_ok=True)
for name in summary['evidence']:
    src, dst = R29 / name, asset / Path(name).name
    assert not dst.exists()
    shutil.copy2(src, dst)
    assert sha(src) == sha(dst)
save(asset / 'hmr2-conclusion.json', summary)
page = VIEWER / 'body-smplx.html'
html = page.read_text(encoding='utf-8-sig')
before = sha(page)
old = 'SAM 3D Body官方权重需要账号获准访问，当前仅完成公开源码与接口核验。'
assert html.count(old) == 1
html = html.replace(old, 'SAM 3D Body官方权重需要账号获准访问。目前四个公开样例的MHR到SMPL-X转换已核验，六张本地原图的裁切、内参与射线条件也已检查；尚未运行此人物的SAM预测。')
marker = '</details>\n</section>\n<!-- GVHMR_PRIOR_019_END -->'
assert html.count(marker) == 1
appendix = '''<h4 id="body-prior-reference-029">单张图像先验与视频先验对照</h4>
<p>HMR2.0a使用已下载权重，通过GVHMR适配实现的完整预测头，独立处理六张原始照片。网络输入保留背景，人物框来自现有遮罩；采用标定内参和许可SMPL模板。没有读取旧人物拟合参数，没有转换为SMPL-X，也没有进行高斯训练。</p>
<figure><a href="body-fitting-method/body-prior-reference/six-source-comparison.jpg"><img src="body-fitting-method/body-prior-reference/six-source-comparison.jpg" alt="六张实拍、GVHMR视频原始预测和HMR2单张原始预测的同相机叠图"></a><figcaption>本地实际原图与网格渲染，每组依次为原图、GVHMR、HMR2。裸人体与衣物不同；头、腕和足部的位置仍需检查。</figcaption></figure>
<figure><a href="body-fitting-method/body-prior-reference/frame23-pelvis-centered-geometry.jpg"><img src="body-fitting-method/body-prior-reference/frame23-pelvis-centered-geometry.jpg" alt="23帧两种原始人体预测，在相同相对旋转下的正侧面对照"></a><figcaption>固定23帧，分别平移骨盆至相同位置，保留各自姿态与模板米制尺度，再施加相同的相机相对旋转。用于检查形态，不代表世界坐标配准通过。</figcaption></figure>
<p>HMR2单帧参考没有改善本素材的整体姿态，侧面可见明显屈膝；23帧左右膝模型弯曲角为45.22°、49.34°，GVHMR原始预测为13.69°、10.53°。这里没有关节真值，不能据此计算准确率。当前不采用该单帧结果作新初始化，保留原查看器。<a href="body-fitting-method/body-prior-reference/hmr2-conclusion.json">精简判定</a></p>
'''
html = html.replace(marker, appendix + marker)
page.write_text(html, encoding='utf-8')
save(R29 / 'publication-receipt.json', dict(page=str(page), before_sha256=before, after_sha256=sha(page),
     url='http://127.0.0.1:8870/body-smplx.html#body-prior-reference-029',
     scope='One subsection in the existing collapsed appendix',
     ppt_changed=False, gaussian_viewer_model_changed=False, remote_deployment=False))
print(json.dumps(dict(state=summary['state'], vposer_sha256_verified=True, frame23_knees=knees,
                     page='http://127.0.0.1:8870/body-smplx.html#body-prior-reference-029'), ensure_ascii=False))
