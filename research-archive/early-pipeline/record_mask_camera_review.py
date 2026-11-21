from pathlib import Path
import json,datetime,hashlib,shutil
import numpy as np
L=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab');B=L/'outputs/mini-splatting2-mip-depth-001/shuolee'
R=B/'current/person-static-mask-foreground-fresh-070';C=B/'current/static-camera-registration-071';V=B.parent/'viewer'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
stamp=datetime.datetime.now().astimezone().isoformat(timespec='seconds')
numeric=read(R/'review/numeric-screening.json');audit=read(C/'audit.json');frozen=read(C/'frozen-model-review/result.json')
decision=dict(created=stamp,actually_inspected=True,quality_accepted=False,whole_boundary_repair=False,
    numeric_screening_passed=False,reviewed=['head-000.jpg','head-023.jpg','head-099.jpg','coverage/boundary-023.jpg','backgrounds/below25-backgrounds.jpg'],
    observations=['Observed-view rear-head white rim is visibly reduced.','The forward face contour remains fuzzy; outside red clothing coverage persists.',
        'The unobserved below25 view still has translucent head/torso and unsupported shoe undersides.','Less visible white does not establish corrected Gaussian support.'],
    continuous_path_accepted=False,local_independent_ply_import='in_progress',pc00_opened_this_round=False,
    ppt_effect='后脑白边明显减轻。\n6个观察视角内部颜色误差下降约13%，\n支持把空白颜色排除出人物学习目标。',
    ppt_summary='前景颜色监督减轻白边；轮廓覆盖仍未通过。\n相机补齐：59个原SfM配准＋116个逐帧背景特征配准。\n相机校正后的从头高斯训练尚未执行。',
    ppt_limit='轮廓外覆盖仍过量；仰视面部、衣物和鞋底仍不完整。\n相机精度、分割与静态几何一致性需要分别验证。')
save(R/'review/visual-decision.json',decision)
summary=audit['summary'];assert summary['total_valid']==175 and summary['new_registered']==116 and not summary['failed_frames']
camera_decision=dict(created=stamp,state='all175_registered_camera_inputs_ready_for_controlled_training',
    original_calibrated=59,new_image_registered=116,remaining_interpolated=0,
    camera_correspondence_review_accepted=True,gaussian_retraining_started=False,quality_accepted=False,
    method='59 original SfM cameras retained.116 EPNP-RANSAC/LM target poses fitted from actual background tracks without interpolated pose initialization; fixed rectified intrinsics and original world frame.',
    held_background=summary,body_silhouette_used_for_pose_estimation=False,
    reviewed_files=['frozen-model-review/camera-boundary-023.jpg','frozen-model-review/camera-boundary-049.jpg'],
    frozen_result=frozen['summary'],
    conclusion='All valid frames now have observation-based camera registration. Frozen Gaussian silhouettes barely improve overall; camera correction alone has not repaired boundaries. Retraining may differ because original geometry was fitted under old cameras.',
    next='One from-scratch camera-input comparison after freezing its protocol; preserve070 mask-only photo loss and other settings. Do not jointly change mask, alpha normalization or densification.',
    caveats=['Fixed-map resection, not a new global bundle adjustment.','3D anchor points and original59 cameras share SfM fitting; not independent calibration GT.',
        'Background raw held correspondences contain outliers; median andp90 do not establish every feature is correct.','Person motion and segmentation can conflict with a shared static surface.'])
save(C/'review-decision.json',camera_decision)
doc=read(C/'training-inputs-registered.json');doc['accepted_for_training']=True;doc['reason']='Passed predeclared background resection gates; independent held statistics and frozen-model boundary review complete. This accepts camera input integrity, not human geometry.'
doc['registration_audit']=dict(path=str(C/'audit.json'),sha256=hashlib.sha256((C/'audit.json').read_bytes()).hexdigest(),review=str(C/'review-decision.json'))
save(C/'training-inputs-registered.json',doc)
st=read(B/'work-status.json');st.update(updated=stamp,local_training_running=False,current_phase='static_all_frame_camera_registration',
    current_experiment='071 camera input registration; no new Gaussian training',quality_accepted=False,
    current_decision='070人物内颜色误差改善但轮廓外覆盖增多，未晋升。已补齐116帧实际背景特征配准，175帧均有观测相机；固定模型边界改善有限，不能归结为相机单一根因。',
    next_action='固定逐帧相机后的从头对照协议；保持070颜色/轮廓/几何参数不变，只比较相机输入的全流程影响。先重新拟合和初始化，不加载旧高斯或优化器。')
st['latest_local_execution']=dict(kind='all_frame_camera_registration_071',directory=str(C),state=camera_decision['state'],running=False,from_scratch=True,new_gaussian_updates=0,quality_accepted=False)
st['latest_completed_mask_experiment']=dict(directory=str(R),steps=18000,points=325508,quality_accepted=False,review=str(R/'review/visual-decision.json'))
st['camera_registration']=dict(directory=str(C),original_sfm=59,new_background_pnp=116,interpolated=0,accepted_for_training=True,gaussian_retrained=False)
st.setdefault('iteration_policy',{}).update(new_training_ready=False,read_only_diagnosis=str(C/'review-decision.json'),required_next_protocol='camera-input-only from-scratch comparison')
save(B/'work-status.json',st)
page=V/'body-smplx.html';text=page.read_text(encoding='utf-8')
old='<p class="notice"><strong>先分析，再迭代。</strong>069未通过边界质量验收。只读检查确认，旧颜色损失包含mask外空白；070正在从原始观测重新拟合与初始化，仅对照“颜色只学人物内部”。质量结论待实际训练和审阅。</p>'
new='<p class="notice"><strong>先分析，再迭代。</strong>070已从20万个新点完成18,000步，后脑白边减轻，但整体边界仍未通过。当前已补齐逐帧相机配准，正在为相机输入的从头对照做准备；没有恢复动态路线。</p>'
assert old in text;text=text.replace(old,new,1)
asset=V/'body-mask-current';asset.mkdir(exist_ok=True)
for src,name in [(R/'review/head-023.jpg','head-comparison.jpg'),(R/'review/coverage/boundary-023.jpg','alpha-coverage.jpg'),
                 (R/'review/backgrounds/below25-backgrounds.jpg','below25-comparison.jpg'),(C/'frozen-model-review/camera-boundary-049.jpg','camera-boundary.jpg')]:shutil.copy2(src,asset/name)
addition='''
<h3 id="camera-registration">每个训练帧都需要由实际图像配准的相机</h3>
<p>原流程最初每隔约3帧选取一张图做SfM，得到59个相机。后续把训练扩展到175张真实图像时，116个新增帧只用了相邻相机插值。矩阵齐全不等于标定齐全：插值不能恢复每一帧实际的轻微转动、平移和抖动。</p>
<p>现在保留59个原SfM配准结果，对另外116帧用真实静态背景的二维—三维对应求解相机位姿。先用EPnP与RANSAC排除错误匹配，再对内点做重投影误差优化；内参及统一世界坐标保持不变，求解不使用插值位姿初值。人体mask、人体几何和高斯均不参与相机配准。</p>
<p>116帧全部通过预设配准检查。另留出的24,255个背景特征观测，重投影误差中位数由0.992降至0.647原图像素，90%分位由2.325降至1.940像素。这是已有三维地图下的逐帧定位核验，不是重新完成全局束调整或独立标定真值。原始对应仍含离群点，不能声称每个特征都已精确对齐。</p>
<figure><a href="body-mask-current/camera-boundary.jpg"><img src="body-mask-current/camera-boundary.jpg" alt="固定高斯仅改变相机，真实照片上的mask轮廓与高斯轮廓对照"></a><figcaption>青色为原始人物mask轮廓，橙色为累计不透明度0.5轮廓。064与070分别比较原相机、实际特征配准相机；同一原图、同一几何，没有为显示单独移动人物。</figcaption></figure>
<p>固定070模型时，全175帧的平均轮廓距离只从2.217降到2.192训练像素，头部覆盖误差没有同步改善。因此相机输入缺口已经补齐，但不能把白边全部解释为相机错误。旧几何已在旧相机下学成，换相机重渲染也不能代替从头训练的对照。</p>
<p>训练入口现在拒绝插值相机、错配帧、无效旋转及未审阅的配准数据。下一轮先冻结相机单项对照协议，再重算人物粗范围、初始化20万个新高斯和优化器；不会接着旧模型训练，也不会同时改变轮廓权重。</p>
<h3>人物内部颜色监督的结果与边界</h3>
<figure><a href="body-mask-current/head-comparison.jpg"><img src="body-mask-current/head-comparison.jpg" alt="实拍、064完整ROI颜色监督、070人物mask内颜色监督"></a><figcaption>实际同相机、同像素裁片。mask内颜色监督减轻后脑白壳，不能据此认定三维轮廓正确。</figcaption></figure>
<p>6个固定观察视图的内部颜色平均绝对误差由0.00888降到0.00775；但边界不透明度误差由0.12891升到0.18781，轮廓外过量覆盖比例由13.24%升到26.26%。人物内部缺口减少、轮廓外覆盖增多，表明去掉空白颜色压力后，原有全图透明度约束仍不足以保证轮廓。该现象与相机误差需要分开分析。</p>
<details><summary>查看真实覆盖及未知仰视结果</summary><figure><img src="body-mask-current/alpha-coverage.jpg" alt="070目标mask与实际不透明度覆盖"><figcaption>原图、mask、累计不透明度、覆盖误差；红色多余、蓝色缺失。</figcaption></figure><figure><img src="body-mask-current/below25-comparison.jpg" alt="064与070同相机仰视黑灰白背景"><figcaption>上064，下070。仰视头脸、衣物和鞋底仍未通过；没有该方向实拍真值。</figcaption></figure></details>
<p><a href="./?model=whole-body-static-mask-foreground">打开070结果（未通过质量验收）</a> · <a href="models/whole-body-static-mask-foreground/shuolee_070_s18000_f0100.ply?download=1">下载070标准PLY</a>。最终325,508点，+Y向上；格式与重载已核验，未部署00。</p>
'''
assert 'id="camera-registration"' not in text
text=text.replace('<h3>从现象区分外观与覆盖</h3>',addition+'\n<details><summary>此前的边界归因与方案推导</summary>\n<h3>从现象区分外观与覆盖</h3>',1)
marker='<details><summary>此前的静态表面约束与对照</summary>'
if marker in text:text=text.replace(marker,'</details>\n'+marker,1)
else:
    # End the folded prior boundary material before the existing static-history block.
    marker='<details><summary>查看此前静态表面约束与消融证据</summary>'
    if marker in text:text=text.replace(marker,'</details>\n'+marker,1)
    else:text=text.replace('<!-- CLOTHING_PIPELINE_CURRENT_END -->','</details>\n<!-- CLOTHING_PIPELINE_CURRENT_END -->',1)
page.write_text(text,encoding='utf8')
p=B/'long-term-plan.md';t=p.read_text(encoding='utf-8-sig');header='''\n<!-- CAMERA_REGISTRATION_CURRENT_BEGIN -->
当前相机输入修复：原175帧中只有59个SfM配准相机，116个为时间插值。071已在原世界坐标中以真实背景对应独立配准全部116帧（EPnP-RANSAC→LM，无插值初值），与原59个组合为175个观测相机。留出背景误差中位数0.992→0.647原图像素；不是新全局BA或绝对相机真值。入口已禁止插值与错配相机。
070仅mask内部颜色监督，18k完成、325508点。白壳减轻，但6视图外侧过覆盖13.24%→26.26%，未晋升。固定070几何换新相机，全175帧轮廓距离2.217→2.192训练像素，头部覆盖未改善，不能断言相机是唯一根因。
下一步：先写相机输入单项对照协议，从原始175帧和已验证的逐帧相机重新拟合粗范围、初始化20万点与优化器，再18k。保持070的mask内颜色、透明度和几何项不变，不能复用旧高斯，不自动叠加轮廓权重/增密调整。当前没有新Gaussian训练运行。新相机清单在current/static-camera-registration-071/training-inputs-registered.json；质量仍未通过，暂不部署00。
<!-- CAMERA_REGISTRATION_CURRENT_END -->\n'''
if '<!-- CAMERA_REGISTRATION_CURRENT_BEGIN -->' not in t:t=t.split('\n',1)[0]+'\n'+header+t.split('\n',1)[1]
p.write_text(t,encoding='utf8')
print(json.dumps(dict(mask_round_accepted=False,cameras_registered=175,new_gaussian_training=False,review_url='http://127.0.0.1:8870/body-smplx.html#camera-registration')))
