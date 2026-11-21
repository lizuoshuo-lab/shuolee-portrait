from pathlib import Path
from datetime import datetime
import hashlib
import json
import re
import shutil

BUILD=Path(__file__).parent
BASE=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
R=BASE/'current/smplx-observation-uncertainty-fresh-026'
VIEWER=BASE.parent/'viewer'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2),'utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=datetime.now().astimezone().isoformat(timespec='seconds')
report=read(R/'review/review-manifest.json')
assert all(report['control024_array_hash_equal'].values())
assert all(read(R/a/'run-state.json')['step']==1200 for a in ['rtmw','mean','uncertainty'])
assert not (R/'phase-conclusion.json').exists()
seen=['source-000023.jpg','source-000045.jpg','fixed-left90.jpg','fixed-right90.jpg','fixed-front-time-sheet.jpg','fixed-left90-time-sheet.jpg']
conclusion=dict(created=now,state='detector_weighting_not_adopted',from_scratch=True,
    identical_initial_arrays=True,control024_exact_array_replay=True,new_body_updates=3600,
    steps_per_arm=1200,new_gaussian_updates=0,body_geometry_accepted=False,remote_deployed=False,
    statistics=report['statistics'],fit_sha256=report['fit_sha256'],
    visually_reviewed=seen,visual_review_scope='Selected actual source overlays and fixed-world side views/time snapshots; no independent body GT or full video acceptance.',
    findings=['All59 calibrated image/frame mappings match. ViTPose crops are pixel-identical and RTMW affine inverse error is below1e-12px.',
        'The rtmw arm exactly replays024 body-texture, all saved arrays match hashes. Each arm independently reconstructs initialization from raw predictions.',
        'Target averaging and direction-dependent uncertainty produce small geometry changes, without resolving the reviewed side-body shape.',
        'Body texture diagnostic median2.971 ->2.815 ->2.697px, but foot texture1.312 ->1.370 ->1.439px. These reused feature tracks are not independent GT.',
        'At reference frame23 knee flexion changes13.25 ->12.87 ->13.89deg left and6.70 ->6.32 ->7.32deg right; these angles describe the model, not an anatomical truth target.'],
    decision='Do not promote any026 fit or begin Gaussian training on it. Compare a genuinely independent image-conditioned body predictor before more small weight adjustments.',
    next_route=dict(name='SAM3DBody/MHR to SMPL-X',official_source='https://github.com/facebookresearch/sam-3d-body',
        conversion_source='https://github.com/facebookresearch/MHR/tree/main/tools/mhr_smpl_conversion',
        weights_ready=False,official_download_requires_account_access=True,actual_subject_inference_done=False))
save(R/'phase-conclusion.json',conclusion)
audit=read(R/'audit/audit.json')
audit.update(visual_review_pending=False,visual_reviewed_at=now,
    visual_observations=['Loose clothing hides anatomical shoulder/elbow/knee centers. Both detector targets are predictions.',
        'Most ViTPose elbow/knee centers appear above RTMW; far-side joints are occluded and cannot be manually confirmed as GT.',
        'No obvious frame or crop displacement appears in the selected overlay photographs.'])
save(R/'audit/audit.json',audit)
for arm in ['rtmw','mean','uncertainty']:
    state=read(R/arm/'run-state.json');state.update(state='reviewed_not_accepted',running=False,accepted=False,reviewed_at=now)
    save(R/arm/'run-state.json',state)
status=read(BASE/'work-status.json')
status.update(updated=now,current_phase='body_observation_uncertainty_fresh_026',local_training_running=False,
    current_decision='026三组从头各1200步：预处理与时间映射通过，关键点均值和方向降权没有修正侧面形态，未采用。VPoser已下载可用；下一条独立预测为SAM3DBody，待官方权重访问。',
    latest_local_execution=dict(kind='body_observation_uncertainty_fresh_026',state=conclusion['state'],running=False,
        directory=str(R),arms=['rtmw','mean','uncertainty'],steps_per_arm=1200,new_body_fit_updates=3600,gaussian_updates=0,accepted=False),
    body_observation_026=dict(conclusion=str(R/'phase-conclusion.json'),accepted=False,next='Independent body predictor and calibrated camera conversion'))
save(BASE/'work-status.json',status)
save(R/'execution-state.json',status['latest_local_execution'])

page=VIEWER/'body-smplx.html'
old=page.read_text('utf-8-sig')
assert 'id="body-observation-026"' not in old
body='''<h3 id="body-observation-026">观测分歧如何进入人体拟合</h3>
<p>肩、肘和膝的二维位置往往需要检测器透过衣物和遮挡推断。两套模型给出不同位置时，直接将任一预测视为准确观测，会把这种偏差传递给三维关节。应先检查图像裁切、像素坐标和时间对应，再讨论观测权重。</p>
<figure><a href="body-fitting-method/gvhmr-motion/detector-knees.jpg"><img src="body-fitting-method/gvhmr-motion/detector-knees.jpg" alt="三个完整实拍局部中的RTMW与ViTPose膝部和脚踝观测"></a><figcaption>真实源图。红圈：RTMW；蓝叉：ViTPose；灰色：当前已剔除的观测。宽松裤装下的膝关节位置没有直接真值。</figcaption></figure>
<p>一种待验证的方法是取两套检测位置的平均值，并沿分歧方向减小约束强度。记两套检测为 <var>u<sub>A</sub></var> 和 <var>u<sub>B</sub></var>，差向量为 <var>δ</var>，相机投影为 <var>π</var>：</p>
<math display="block"><mrow><mover><mi mathvariant="bold">u</mi><mo>¯</mo></mover><mo>=</mo><mfrac><mrow><msub><mi mathvariant="bold">u</mi><mi>A</mi></msub><mo>+</mo><msub><mi mathvariant="bold">u</mi><mi>B</mi></msub></mrow><mn>2</mn></mfrac><mo>,</mo><mspace width="1em"/><mi mathvariant="bold">δ</mi><mo>=</mo><msub><mi mathvariant="bold">u</mi><mi>A</mi></msub><mo>−</mo><msub><mi mathvariant="bold">u</mi><mi>B</mi></msub></mrow></math>
<math display="block"><mrow><mi mathvariant="bold">Σ</mi><mo>=</mo><msubsup><mi>σ</mi><mn>0</mn><mn>2</mn></msubsup><msub><mi mathvariant="bold">I</mi><mn>2</mn></msub><mo>+</mo><mi mathvariant="bold">δ</mi><msup><mi mathvariant="bold">δ</mi><mi mathvariant="normal">T</mi></msup><mo>,</mo><mspace width="1em"/><mi mathvariant="bold">r</mi><mo>=</mo><msup><mi mathvariant="bold">Σ</mi><mrow><mo>−</mo><mfrac><mn>1</mn><mn>2</mn></mfrac></mrow></msup><mo>(</mo><mi mathvariant="bold">π</mi><mo>−</mo><mover><mi mathvariant="bold">u</mi><mo>¯</mo></mover><mo>)</mo></mrow></math>
<p><var>σ<sub>0</sub></var>为基础像素尺度，此处取20像素；<var>I<sub>2</sub></var>为二维单位矩阵。分歧越大，沿 <var>δ</var> 的残差受到的惩罚越弱，垂直方向保持原尺度。该矩阵是人为构造的不确定性近似，没有经过统计校准；两套检测也可能有相同偏差，平均不能消除共同错误。</p>
<figure><a href="body-fitting-method/gvhmr-motion/observation-left90.jpg"><img src="body-fitting-method/gvhmr-motion/observation-left90.jpg" alt="原观测、均值观测、均值加方向不确定性三组从头拟合的完整固定侧视对照"></a><figcaption>同一世界相机、同一人体时刻23。左：原观测；中：均值；右：均值加方向不确定性。没有逐模型转正或按身高缩放。</figcaption></figure>
<p>这组对照没有明显修正侧面形态。部分纹理残差降低，但足部残差上升，因此未采用。二维检测定义、身体先验和真实几何支持需要分别核查，不能仅凭某个残差变小就宣布人体对齐。</p>
<p>下一条独立参考是 <a href="https://github.com/facebookresearch/sam-3d-body">SAM 3D Body</a>，其预测可通过<a href="https://github.com/facebookresearch/MHR/tree/main/tools/mhr_smpl_conversion">官方MHR转换工具</a>拟合到SMPL-X。先核对原始预测在真实相机中的投影，再检查转换损失和固定侧面；目前尚未在此素材上运行。</p>
'''
start=old.index('<h3>二维关节点也需要核验</h3>')
end=old.index('<details><summary>完整视角、时间回放与实验参数</summary>',start)
new=old[:start]+body+old[end:]
section_end=new.index('</section>\n<!-- GVHMR_PRIOR_019_END -->')
details_end=new.rfind('</details>',0,section_end)
extra='''<h4>观测核验与方向不确定性对照</h4><p>026核对全部59帧：时间映射和图像散列一致，ViTPose裁切逐像素一致，RTMW仿射逆变换误差小于10<sup>−12</sup>像素。三组均重新生成原始GVHMR初始化并新建优化器，各1200步；原观测组逐数组复现024身体纹理组。手部、足部、轮廓、接触、纹理与姿态项不变，只依次改变身体关键点目标和方向权重。</p>
<p>三组的未优化身体纹理残差中位数为2.971、2.815、2.697像素；足部为1.312、1.370、1.439像素。对原RTMW点的残差为10.063、10.683、10.327像素，这些检测点不是真值。共3600次身体更新、0次高斯更新；已查看代表性实拍叠图、左右侧面与六时刻快照，没有独立身体测量或完整连续视频验收。</p>
<p><a href="body-fitting-method/gvhmr-motion/observation-source23.jpg">真实正面叠图</a> · <a href="body-fitting-method/gvhmr-motion/observation-source45.jpg">真实侧面叠图</a> · <a href="body-fitting-method/gvhmr-motion/observation-time-sheet.jpg">六时刻侧面</a> · <a href="body-fitting-method/gvhmr-motion/observation-conclusion.json">观测对照判定</a>。SAM 3D Body官方权重需要账号获准访问，当前仅完成公开源码与接口核验。</p>
'''
new=new[:details_end]+extra+new[details_end:]
# Remove a stale prospective statement from an older collapsed historical appendix.
new=new.replace('尚未运行本地GVHMR实验。接入前需验证原视频帧率、真实相机内参和旋转，分别保留原始预测与后处理结果。','后续已运行本地GVHMR预测和从头拟合，实际结论见本页前部的人体先验与运动约束部分。')
assets={'detector-knees.jpg':R/'audit/detectors-lower-0.jpg','observation-left90.jpg':R/'review/fixed-left90.jpg',
    'observation-source23.jpg':R/'review/source-000023.jpg','observation-source45.jpg':R/'review/source-000045.jpg',
    'observation-time-sheet.jpg':R/'review/fixed-left90-time-sheet.jpg','observation-conclusion.json':R/'phase-conclusion.json'}
dest=VIEWER/'body-fitting-method/gvhmr-motion'
for name,src in assets.items():shutil.copyfile(src,dest/name)
page.write_text(new,'utf-8')
save(R/'web-publication.json',dict(created=now,page=str(page),before_sha256=hashlib.sha256(old.encode()).hexdigest(),after_sha256=sha(page),anchor='body-observation-026',assets=[{'name':n,'sha256':sha(dest/n)} for n in assets],browser_verification_pending=True))
print(json.dumps({'phase':conclusion['state'],'new_body_updates':3600,'page':str(page)},ensure_ascii=False))
