from pathlib import Path
from datetime import datetime
import hashlib
import json
import re
import shutil

LAB = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab')
BASE = LAB / 'outputs/mini-splatting2-mip-depth-001/shuolee'
VIEWER = BASE.parent / 'viewer'
REPO = LAB / 'repos/GaussianAvatars_SMPLX-shuolee'
R14 = BASE / 'current/smplx-observation-support-014'
R15 = BASE / 'current/smplx-leg-visibility-fresh-015'
R16 = BASE / 'current/smplx-floor-contact-fresh-016'
stamp = datetime.now().astimezone().isoformat(timespec='seconds')
read = lambda p: json.loads(p.read_text('utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def save(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2), 'utf-8')

reviews = {
    R15: ['fixed-left90.jpg', 'fixed-right90.jpg', 'fixed-front.jpg', 'source-000023.jpg', 'source-000045.jpg'],
    R16: ['fixed-left90.jpg', 'source-000045.jpg', 'source-000023.jpg', 'fixed-above25.jpg'],
}
conclusions = {
    R15: 'Excluding occluded far-leg predictions modestly changes knee observation residuals but does not correct knee flexion or raised toes. VPoser variant also fails whole-body review.',
    R16: 'Sole orientation improves under the floor-contact loss, but knee geometry is almost unchanged and body/ankle observation residuals increase. Not accepted as a whole-body fit. Retain only as the current diagnostic comparison; no Gaussian training or remote deployment.',
}
for root, images in reviews.items():
    manifest = read(root / 'review/review-manifest.json')
    manifest.update(state='reviewed_not_accepted', quality_accepted=False,
                    visual_review_pending=False, reviewed_at=stamp,
                    manually_reviewed_images=images,
                    review_scope='Named images only. Other generated images are not claimed to have been manually inspected.',
                    conclusion=conclusions[root])
    save(root / 'review/review-manifest.json', manifest)
    state = read(root / 'run-state.json')
    state.update(state='complete_reviewed_not_accepted', quality_accepted=False,
                 reviewed_at=stamp, conclusion=conclusions[root])
    save(root / 'run-state.json', state)

audit = read(R14 / 'track-audit.json')
audit.update(state='diagnostic_reviewed', reviewed_at=stamp,
             reviewed_corrected_track_images=['track-review-000000.jpg', 'track-review-000058.jpg'],
             reviewed_lower_limb_contact_sheets=['lower-limb-contact-00.jpg', 'lower-limb-contact-12.jpg', 'lower-limb-contact-24.jpg', 'lower-limb-contact-36.jpg', 'lower-limb-contact-48.jpg'],
             interpretation='Source texture tracks support a partly stable clothed surface. They do not measure hidden joints, bare skin or independent camera accuracy.')
save(R14 / 'track-audit.json', audit)
state = read(R14 / 'run-state.json')
state.update(state='diagnostic_reviewed', reviewed_at=stamp)
save(R14 / 'run-state.json', state)

# Correct only copied explanatory metadata. Keep the original executed source hash.
for arm in ['control', 'floor_contact']:
    file = R16 / arm / 'fit-training-protocol.json'
    protocol = read(file)
    before = {'body_initialization_difference': protocol['body_initialization_difference'],
              'lower_limb_visibility': protocol['lower_limb_visibility'].copy()}
    protocol['body_initialization_difference'] = 'Both axis-angle arms have identical initial arrays. Neither arm uses VPoser.'
    protocol['lower_limb_visibility'].pop('only_A_vs_B_difference', None)
    protocol['lower_limb_visibility']['between_arm_difference'] = 'None: both arms use identical lower-limb visibility weights.'
    protocol['metadata_correction'] = {
        'created': stamp, 'previous_explanatory_fields': before,
        'reason': 'Copied descriptions referred to the three-arm 015 experiment. Corrected to the executed two-arm contact-only contrast.',
        'executed_source_sha256': protocol['source_code_sha256'],
        'source_after_metadata_only_correction_sha256': sha(REPO / 'fit_floor_contact_from_scratch.py'),
        'training_repeated': False, 'numerical_training_changed': False,
    }
    save(file, protocol)

next_step = ('Audit view-correlated root rotation, translation and lower-limb articulation against stable real feature tracks. '
             'Then run a fresh controlled comparison of shared lower-body placement versus bounded per-frame motion, '
             'while allowing observed head/arm motion. Do not hard-code a standing pose or use old fitted states as initialization. '
             'Treat clothing and shoes as surface evidence, not joint/skin truth.')
summary = {
    'created': stamp, 'state': 'reviewed_not_accepted', 'body_prior_download_complete': True,
    'rounds': ['014 diagnostics', '015 three fresh fits', '016 two fresh fits'],
    'new_fitting_updates': 7000, 'new_gaussian_updates': 0,
    'all_fits_from_raw_observations': True, 'old_learned_initialization': False,
    'same_016_initial_arrays': True, '016_control_reproduces_015_axis_visible': True,
    'floor_camera_up_change_degrees': 10.883062768393469,
    'feature_candidates': 943, 'stable_feature_tracks': 312,
    'sole_angles_deg_control': [31.763165213347644, 22.445466086257433],
    'sole_angles_deg_contact': [13.334949654689362, 10.838568053173178],
    'visible_body_prediction_residual_median_px': [8.118761320749135, 9.25350007490832],
    'limitations': ['Contact metric participates in the constraint; not independent accuracy.',
                    'No independent 3D joint ground truth; knee flexion remains.',
                    'Source floor normal is not IMU gravity; clearance uses model scale.',
                    'Both shoes planted and 1-7cm clearance are bounded assumptions.'],
    'adopted_as_final_model': False, 'remote_deployed': False,
    'next_step': next_step,
}
save(R16 / 'review/phase-conclusion.json', summary)

dest = VIEWER / 'body-fitting-method/floor-contact'
dest.mkdir(parents=True, exist_ok=True)
for file in reviews[R16]:
    shutil.copy2(R16 / 'review' / file, dest / file)
shutil.copy2(R14 / 'track-review-000058.jpg', dest / 'source-correspondences-000058.jpg')
for src, name in [(R16 / 'review/review-manifest.json', 'comparison-016.json'),
                  (R15 / 'review/review-manifest.json', 'comparison-015.json'),
                  (R16 / 'review/foot-floor-diagnostic.json', 'foot-floor-diagnostic.json'),
                  (R16 / 'review/phase-conclusion.json', 'phase-conclusion.json'),
                  (R14 / 'floor-reference.json', 'floor-reference.json')]:
    shutil.copy2(src, dest / name)

page = VIEWER / 'body-smplx.html'
text = page.read_text('utf-8')
old = '目前确认本地拟合流程的观测处理和约束存在问题。可见性、手部与VPoser的从头拟合对照已完成，身体姿态尚未通过；没有新增高斯训练，也没有向00部署。不能据此认定 GaussianAvatars 原方法无效。'
new = '身体先验、遮挡剔除与脚底接触的从头对照已完成。脚掌朝向有所改善，但膝部与整体姿态仍未通过；当前继续核查跨视角运动估计，没有新增高斯训练或部署00。以下均为本地拟合诊断，不能据此否定原论文完整方法。'
assert old in text
text = text.replace(old, new, 1)
text = text.replace('下一步检查真实对应、人物运动与三维表面支持，再决定拟合约束。', '真实对应与地面参考的检查结果见下方；下一步继续区分实际运动与逐帧拟合漂移。', 1)
anchor = '<details><summary>配套源码、实验范围与记录</summary>'
assert anchor in text and 'id="floor-contact-016"' not in text
section = '''<h3 id="floor-contact-016">身体先验之外，还需要场景中的接触关系</h3>
<p>人体姿态先验约束关节配置；相机和真实图像约束投影；地面接触则约束脚底的位置和朝向。三者提供的信息不同。此次从原始稀疏场景点估计地面，只调整检查相机的竖直参考，人体几何保持原值；这样可以把显示方向与局部关节错误分开检查。</p>
<p>在人物双脚看起来持续着地的这段视频中，再从头比较无接触项与增加接触项。裸脚位于鞋内，因此给脚底保留高度范围，不把裸脚强压到地面，也不要求人体填满宽松衣服。</p>
<figure><a href="body-fitting-method/floor-contact/fixed-left90.jpg"><img src="body-fitting-method/floor-contact/fixed-left90.jpg" alt="同一人物时刻和地面参考相机：无接触约束、加入脚底约束、原图特征的稀疏三维支持"></a><figcaption>固定时刻23，所有对象使用同一相机。脚掌更平，但膝部弯曲基本未变。右侧来自真实纹理对应，只描述少量可见表面，不能当作完整人体或关节真值。</figcaption></figure>
<figure style="max-width:760px"><a href="body-fitting-method/floor-contact/source-000045.jpg"><img src="body-fitting-method/floor-contact/source-000045.jpg" alt="实拍侧面与两组裸人体叠加，接触约束没有解决整体姿态"></a><figcaption>实拍45：A不含脚底约束，B加入脚底约束。应同时检查原图和新视角，不能只根据接触项降低就认定对齐正确。</figcaption></figure>
<p>这组的脚底与地面夹角减小，但身体关节图像残差增大，整体仍不采用。后续重点核查：同一人物的整体旋转、位置和腿部姿态，是否在逐帧优化中随观察角度漂移。头部和手臂可能有真实运动，不能把全身强行设成完全静止。</p>
<details><summary>实验附录：真实对应、遮挡与接触对照</summary>
<p>原始59帧的纹理特征先处理同帧重复对应，再按连续10帧分块留出三角化。943条候选中312条满足稳定性筛选；其中脚部66条、下腿区域72条。区域按人物包围框的高度粗分，并非解剖标注。所有候选的脚部留出残差中位数1.39像素，下腿1.79像素。相机参与过原始SfM，这不是独立标定准确率，也不代表衣服下的关节位置。</p>
<p>检查全部59帧后，剔除0–11帧中被遮挡的右膝、右踝和足部点，以及42–58帧中被遮挡的左侧对应点。015的三组各1400步：剔除后膝部残差略减，但网格姿态没有明显修复；VPoser组也未通过。</p>
<p>地面来自1393个原始SfM候选点、1284个平面内点。检查相机的新旧竖直方向相差10.88°，这只改变观察方向，不修改网格，也不是独立重力测量。</p>
<p>016两组初始数组逐项相同，各1400步；控制组逐字节复现015的可见性组。接触项在250步后用200步逐渐加入，权重1。模板脚底由踝/脚蒙皮权重大于0.7且距最低点8毫米内的顶点选取，每脚99个。允许1–7厘米的模型尺度高度和12毫米的脚底高度起伏；这些是考虑鞋的有限假设，不是测量结果。</p>
<p>固定时刻23的左右脚底平面夹角分别由31.76°、22.45°变为13.33°、10.84°；左右膝几何弯曲约23.37°、22.19°变为23.39°、21.63°。可见身体关节残差8.12→9.25像素，手部预测残差6.23→6.05像素。接触指标参与约束，检测残差来自训练观测，均不是独立几何精度。</p>
<p>本阶段015/016共7000次身体拟合、0次高斯更新。016人工检查侧面、俯视及正面/侧面实拍四图后判为整体未通过；没有宣称其余生成图或连续视频已完成人工验收。</p>
<div class="route-links"><a href="body-fitting-method/floor-contact/source-000023.jpg">正面叠图</a><a href="body-fitting-method/floor-contact/fixed-above25.jpg">俯视</a><a href="body-fitting-method/floor-contact/source-correspondences-000058.jpg">真实对应</a><a href="body-fitting-method/floor-contact/comparison-015.json">遮挡对照</a><a href="body-fitting-method/floor-contact/comparison-016.json">接触对照</a><a href="body-fitting-method/floor-contact/foot-floor-diagnostic.json">脚底几何量</a></div>
</details>
'''
page.write_text(text.replace(anchor, section + anchor, 1), 'utf-8')

status = read(BASE / 'work-status.json')
status.update(updated=stamp, current_phase='floor_contact_reviewed_next_motion_audit', local_training_running=False)
status['current_experiment'].update(status='fitting_reviewed_not_accepted', accepted=False,
    latest_fitting_trial='016', latest_fitting_steps=1400, latest_fitting_arms=2,
    latest_fitting_gaussian_steps=0, current_training_running=False,
    remote_deployed=False, eyewear_specialization=False)
status['surface_contact_closeout'] = summary
status['next_action'] = next_step
save(BASE / 'work-status.json', status)
plan = BASE / 'long-term-plan.md'
text = plan.read_text('utf-8')
new_top = '''<!-- CURRENT_GAUSSIANAVATARS_SMPLX -->
当前优先解决 GaussianAvatar / GaussianAvatars_SMPLX 的全身配准。所有新实验从原始观测、相机、许可模板与官方预训练输出开始；旧学习结果只读对照。眼镜专项停止，不部署00，取消E盘同步。

VPoser V02_05已下载、接入并通过RTX5080运行核验，无需重下。012/013四组各1400步没有通过整体姿态检查。014随后从原始SfM纹理对应检查三维支持：同帧对应先去重，943条候选中312条稳定；这是衣物、鞋面等可见表面的有限证据，不是关节或皮肤真值，也不是独立标定精度。

015按59帧真实遮挡剔除不可见膝/踝/足部观测，轴角控制、可见性轴角、可见性VPoser三组各1400步。局部膝残差略减，膝部和翘脚姿态仍未解决。已审阅的五张关键图与完整数值保留，失败模型和重复渲染按清单精简；016控制已逐字节复现015可见性组。

016两组各1400步，从相同原始初始化比较脚底接触约束，共2800次新拟合。地面从原始SfM点估计，相机竖直参考改正10.88度，人体几何没有旋转。固定时刻23脚底夹角31.76/22.45度降至13.33/10.84度，但膝弯曲约23/22度几乎不变，可见身体残差8.12升到9.25px；接触项与检测残差不能作为独立真值。实拍正面/侧面、固定侧面/俯视四图已检查，整体不采用，暂留016作为当前诊断对照。015/016合计7000次身体更新，新增高斯更新为0。

下一步：用真实纹理轨迹核查逐帧根旋转、平移和腿部关节是否随观察方向漂移。原始单目根朝向在世界中已存在差异；不能直接当运动真值。下一组从头比较共享下身位置与受限逐帧运动，同时允许真实头部/手臂运动；先检查全局运动和关节映射，再决定参数约束，不强行写入站姿或把衣服表面当皮肤。接触约束目前只能作为局部候选，不能当整体已正确。

通过裸网格原图叠加、固定时刻多方向与时间一致性后，再从新的20万高斯及新优化器训练18000步，禁止恢复180000步任务。保持同一55页技术PPT，更新最后一页及技术备注，不新增同题报告。

[身体先验](http://127.0.0.1:8870/body-smplx.html#vposer-ready) · [接触与几何对照](http://127.0.0.1:8870/body-smplx.html#floor-contact-016)
<!-- END_CURRENT_GAUSSIANAVATARS_SMPLX -->'''
text, count = re.subn(r'<!-- CURRENT_GAUSSIANAVATARS_SMPLX -->.*?<!-- END_CURRENT_GAUSSIANAVATARS_SMPLX -->', lambda _:new_top, text, count=1, flags=re.S)
assert count == 1
plan.write_text(text, 'utf-8')
print(json.dumps({'reviewed': ['014', '015', '016'], 'accepted': False, 'page': str(page), 'new_updates': 7000}, ensure_ascii=False))
