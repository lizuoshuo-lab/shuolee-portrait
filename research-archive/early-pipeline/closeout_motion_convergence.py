from pathlib import Path
from datetime import datetime
import hashlib, json, re, shutil

BASE=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
VIEWER=BASE.parent/'viewer'
R17=BASE/'current/smplx-motion-consistency-fresh-017'
R18=BASE/'current/smplx-convergence-fresh-018'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
stamp=datetime.now().astimezone().isoformat(timespec='seconds')
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n','utf-8')

reviewed=['fixed-left90.jpg','source-000045.jpg','source-000023.jpg','fixed-above25.jpg']
assert read(R18/'run-state.json')['state']=='complete_needs_review'
assert all(x['exact_017_reproduction'] for x in read(R18/'run-state.json')['milestone1400'].values())
assert read(R18/'review/knee-objective-audit.json')['new_optimization_steps']==0
conclusions={
 R17:'Sole-position consistency reduces shoe-feature transfer drift under the planted-foot hypothesis. Knees and whole-body geometry remain unacceptable in the four inspected views. Not promoted.',
 R18:'Fresh5000-step fits exactly reproduce017 at1400steps. More steps reduce body prediction residuals and modestly reduce knee flexion, but whole-body geometry remains unacceptable. Stable soles reduce shoe-transfer drift. No Gaussian training or deployment.'}
for root in [R17,R18]:
    manifest=read(root/'review/review-manifest.json')
    manifest.update(state='reviewed_not_accepted',visual_review_pending=False,quality_accepted=False,
        manually_reviewed_images=reviewed,reviewed_at=stamp,conclusion=conclusions[root],
        review_scope='Only the four named images were manually inspected. Other generated frames and continuous video are not claimed accepted.')
    save(root/'review/review-manifest.json',manifest)
    state=read(root/'run-state.json')
    state.update(state='complete_reviewed_not_accepted',reviewed_at=stamp,quality_accepted=False,conclusion=conclusions[root])
    save(root/'run-state.json',state)

metrics=read(R18/'review/review-manifest.json')
motion=read(R18/'review/motion-audit.json')
perturb=read(R18/'review/knee-objective-audit.json')
probe=next(p for p in perturb['results']['stable_feet']['probes'] if p['local_knee_x_degrees']==-5)
next_step=('Check the official GVHMR sequence prior as an independent fresh alternative to framewise SMPLer-X. '
 'First inventory dependencies and existing weights, verify source frame timing, crop convention, actual camera intrinsics and rotations. '
 'Use the original time base; do not pass59 stride3 observations as a30fps clip or label the moving camera static. '
 'Compare predicted SMPL-X body geometry in the same source cameras before any personalized fit or20万/18000-step Gaussian training. '
 'Keep official raw predictions and postprocessed motion separately. No old fitted body initialization, no default large local download, no00 deployment before acceptance.')
summary={'created':stamp,'state':'reviewed_not_accepted','rounds':['017 motion comparison','018 convergence comparison'],
 'new_fitting_updates':14200,'new_gaussian_updates':0,'all_experiments_from_scratch':True,
 '017_steps':{'control':1400,'shared_lower':1400,'stable_feet':1400},
 '018_steps':{'control':5000,'stable_feet':5000},'018_milestone1400':read(R18/'run-state.json')['milestone1400'],
 'stable_shoe_tracks':motion['stable_shoe_tracks_used'],
 '018_shoe_transfer_median_px':{k:v['shoe_feature_transfer_error_px']['median'] for k,v in motion['models'].items()},
 '018_body_prediction_residual_median_px':{k:v['visible_body_median_px'] for k,v in metrics['training_observation_statistics'].items()},
 '018_frame23_pose_geometry':metrics['pose_diagnostics'],
 'knee_probe_minus5deg':probe,'accepted_as_body_model':False,'remote_deployed':False,
 'limitations':['Feature-transfer check uses a preselected stable shoe subset and an approximate rigid foot frame.',
 'Sole consistency participates in the objective; its decrease is not independent geometry accuracy.',
 'Body and hand values are errors to training predictors, not3D ground truth.',
 'The knee perturbation fixes hips, ankles and global parameters; it does not rule out a coupled better solution.',
 'Five new fits completed. No Gaussian appearance model was trained in017/018.'],
 'next_step':next_step}
save(R18/'review/phase-conclusion.json',summary)
research={'created':stamp,'status':'source_review_only_no_new_models_run',
 'preferred_next_candidate':{'name':'GVHMR','source':'https://github.com/zju3dv/GVHMR',
 'installation':'https://raw.githubusercontent.com/zju3dv/GVHMR/main/docs/INSTALL.md',
 'entry':'https://raw.githubusercontent.com/zju3dv/GVHMR/main/tools/demo/demo.py',
 'postprocess':'https://raw.githubusercontent.com/zju3dv/GVHMR/main/hmr4d/model/gvhmr/utils/postprocess.py',
 'verified':'Predicts SMPL-X parameters, estimates sequence motion, and supplies stationary-joint translation/IK postprocessing. Demo converts SMPL-X vertices to SMPL for rendering.',
 'local_test_reason':'Provides an independent temporal body prior and explicit motion/contact treatment, matching the observed framewise drift.',
 'integration_cautions':['Use actual calibrated intrinsics rather than the demo focal-length estimate.',
 'Verify camera rotation conventions and native temporal sampling; moving camera must not use static_cam.',
 'Postprocessing ground shift is a visualization/heuristic step, not recovery of the measured scene floor.',
 'Face and fingers must remain separately documented. A sequence body prior alone does not solve whole-avatar reconstruction.'],
 'weights_needed':['gvhmr_siga24_release.ckpt','hmr2/epoch=10-step=25000.ckpt','vitpose-h-multi-coco.pth','yolov8x.pt'],
 'weights_checked_locally':False,'personalized_experiment_started':False},
 'alternative':{'name':'SMPLest-X','source':'https://github.com/MotrixLab/SMPLest-X/blob/main/README.md',
 'verified':'Official code and pretrained Huge model available. README lists8.2GB weight download.',
 'reason_deferred':'A stronger framewise estimator remains useful as a second independent prior, but does not directly model the observed temporal drift. Large weight download should follow the user preference for00 when available.',
 'downloaded':False},
 'official_fitting_reference':{'source':'https://github.com/vchoutas/smplify-x/blob/master/cfg_files/fit_smplx.yaml',
 'verified':'Official configuration uses neutral SMPL-X, body prior and staged fitting with LBFGS line search. Local bounded Adam adapter is not a full reproduction.'}}
save(R18/'review/next-prior-research.json',research)

dest=VIEWER/'body-fitting-method/motion-convergence';dest.mkdir(parents=True,exist_ok=True)
for name in reviewed+['review-manifest.json','motion-audit.json','knee-objective-audit.json','phase-conclusion.json','next-prior-research.json']:
    shutil.copy2(R18/'review'/name,dest/name)

page=VIEWER/'body-smplx.html';text=page.read_text('utf-8')
old='身体先验、遮挡剔除与脚底接触的从头对照已完成。脚掌朝向有所改善，但膝部与整体姿态仍未通过；当前继续核查跨视角运动估计，没有新增高斯训练或部署00。以下均为本地拟合诊断，不能据此否定原论文完整方法。'
new='VPoser身体先验已下载并验证。脚底接触、运动一致性及更长拟合预算的从头对照已完成，脚部漂移减小，膝部与整体姿态仍未通过。下一步核验独立的时序人体先验，没有新增高斯训练或部署00。以下均为本地有界诊断。'
assert old in text;text=text.replace(old,new,1)
anchor='<details><summary>配套源码、实验范围与记录</summary>'
assert anchor in text and 'id="motion-convergence-018"' not in text
section='''<h3 id="motion-convergence-018">脚部稳定与全身姿态需要分别约束</h3>
<p>逐帧重投影可以通过改变人体位置和关节角度来减小误差。当相机围绕近似静止的人物移动时，这种自由度也可能把观察方向的变化解释成人体运动。姿态先验约束关节配置，地面约束脚底高度和朝向，跨帧接触约束则限制着地点的漂移。三者不能互相替代。</p>
<p>本段视频中，筛选出的鞋面纹理在世界坐标中较稳定，因此比较“仅有地面接触”和“增加脚底位置稳定”两种条件。稳定项只作用于脚底，头部和手臂仍可运动。所有组均从原始观测重新初始化。</p>
<figure><a href="body-fitting-method/motion-convergence/fixed-left90.jpg"><img src="body-fitting-method/motion-convergence/fixed-left90.jpg" alt="相同侧面相机：1400步接触对照、5000步接触对照、5000步脚底稳定和原图三维纹理"></a><figcaption>同一时刻23及同一相机。增加步数和脚底稳定后，脚部运动较稳定，但膝部弯曲仍然明显。右侧仅为真实纹理的稀疏三维支持。</figcaption></figure>
<figure style="max-width:820px"><a href="body-fitting-method/motion-convergence/source-000045.jpg"><img src="body-fitting-method/motion-convergence/source-000045.jpg" alt="真实侧面照片与三组裸人体投影对照"></a><figcaption>实拍45。裸人体应与衣服下的解剖位置相容，但宽松衣物轮廓不能直接作为皮肤或关节真值。</figcaption></figure>
<p>更长优化降低了图像预测残差，仍没有恢复可靠的整体姿态。局部检查也没有支持“只要去掉姿态先验就会变直”：单独改变膝角时，足部投影与接触关系更容易被破坏。下一步需要检查髋、膝、踝及整体位置的联合解释，并用独立的时序人体预测进行对照。</p>
<details><summary>实验附录：运动一致性、优化预算与局部目标检查</summary>
<p>017三组各1400步，分别为接触控制、共享下身运动惩罚、脚底位置稳定惩罚。三组初始数组相同。稳定项按每个脚底顶点在59帧中的世界位置方差约束，模型尺度归一化为1厘米，使用平滑鲁棒损失；250步后用200步逐渐加入。双脚着地是本素材的假设，不能直接推广到行走动作。</p>
<p>018重新从原始输入运行两组各5000步，分别为接触控制和脚底稳定。两组1400步中间输出与017对应组逐字节相同，没有沿用旧拟合结果。017/018共14200次身体更新，0次高斯更新。</p>
<p>66条稳定鞋面纹理共1289次观测，用估计脚部坐标系传递后，018控制与稳定组的投影残差中位数为43.36、7.32像素。静态纹理拟合残差为0.63像素。这是稳定子集上的近似刚体检查，相机和纹理估计、鞋与裸脚的关系都会影响结果，不能当作独立运动精度。</p>
<p>固定时刻23的左右膝几何弯曲，1400步控制约23.39°、21.63°，5000步控制约20.01°、21.05°，5000步稳定组约19.25°、19.95°。这些角度来自模型，真实膝角未知。四张关键图检查后，整体均未通过。</p>
<p>只读扰动检查中，将稳定组两个膝关节的局部X旋转各减少5°，模型膝弯曲约降至14.55°、15.04°。加权目标增加约0.425，其中足部图像项增加0.184、身体图像项0.076、接触项0.091、稳定项0.067，两个单目三维先验项合计仅增加约0.00074。此检查固定了髋部、踝部、体形与整体参数，所以只能排查单关节局部解释，不能排除更好的联合解。</p>
<p>已核对<a href="https://github.com/zju3dv/GVHMR">GVHMR官方代码</a>：它预测SMPL-X身体参数，并提供静止关节和逆运动学后处理，适合作为下一条独立时序参考。尚未运行本地GVHMR实验。接入前需验证原视频帧率、真实相机内参和旋转，分别保留原始预测与后处理结果。<a href="https://github.com/MotrixLab/SMPLest-X">SMPLest-X</a>作为后备单帧先验，官方Huge权重约8.2GB，当前未下载。</p>
<div class="route-links"><a href="body-fitting-method/motion-convergence/source-000023.jpg">正面叠图</a><a href="body-fitting-method/motion-convergence/fixed-above25.jpg">俯视</a><a href="body-fitting-method/motion-convergence/review-manifest.json">数值对照</a><a href="body-fitting-method/motion-convergence/motion-audit.json">纹理与运动</a><a href="body-fitting-method/motion-convergence/knee-objective-audit.json">局部目标检查</a></div>
</details>
'''
page.write_text(text.replace(anchor,section+anchor,1),'utf-8')

status=read(BASE/'work-status.json')
status.update(updated=stamp,current_phase='motion_convergence_reviewed_next_sequence_prior',local_training_running=False)
status['current_experiment'].update(status='fitting_reviewed_not_accepted',accepted=False,
 latest_fitting_trial='018',latest_fitting_steps=5000,latest_fitting_arms=2,latest_fitting_gaussian_steps=0,
 current_training_running=False,remote_deployed=False,eyewear_specialization=False)
status['motion_convergence_closeout']=summary;status['next_action']=next_step
save(BASE/'work-status.json',status)
plan=BASE/'long-term-plan.md';text=plan.read_text('utf-8')
new_top='''<!-- CURRENT_GAUSSIANAVATARS_SMPLX -->
当前优先解决GaussianAvatar / GaussianAvatars_SMPLX的全身配准。所有新实验从原始观测、相机、许可模板与官方预训练输出开始，旧学习结果只读对照。眼镜专项停止，不部署00，取消E盘同步。

VPoser V02_05已下载、接入并通过RTX5080核验，无需重下。012/013四组各1400步未通过整体姿态检查。014原始纹理对应经同帧去重与留出三角化，943条候选中312条稳定。015遮挡剔除及016接触约束改善部分投影/脚底朝向，但膝部仍未通过。原始观测、先验、源码与必要证据保留。

017三组各1400步比较接触控制、共享下身、脚底位置稳定。三组起点相同。脚底稳定减少了66条稳定鞋面纹理的模型传递残差，但整体仍未通过。018两组从头各5000步比较接触控制/脚底稳定，1400步中间输出逐字节复现017对应组，未从旧断点续跑。

018鞋面传递残差中位数43.36降至7.32px，但固定时刻23膝弯曲仍约19–20度。较长优化只带来有限变化。四张关键图已检查，不能把训练误差或脚底约束改善当作独立准确率。017/018共14200次新的身体拟合，0次高斯更新，整体不采用。

只读膝部扰动显示：固定其他参数，膝局部X各减少5度时，足部图像、身体图像、接触和稳定项是主要新增代价，单目三维先验代价变化很小。这不能证明先验没有影响，也不能排除髋/膝/踝与整体位置的更好联合解，不再把重复去除先验或单纯增加步数作为默认下一步。

下一步核验GVHMR官方时序人体预测作为独立起点。已阅读官方安装、demo和静止关节/IK源码，确认预测SMPL-X参数，尚未下载新权重或运行该模型。先核查现有依赖和权重、视频原始时基与裁切、真实相机K及旋转约定，不能把59个每3帧采样观测直接当30fps，也不能将移动相机设为静止。分别保留原始预测和后处理结果，在相同相机对照后再开始新的个体拟合。SMPLest-X为后备，官方Huge权重8.2GB，未下载；大文件尽量在00获取，已有本地资源优先复用。

通过裸网格原图叠加、固定时刻多方向与时间一致性后，再从20万新高斯和新优化器训练18000步。禁止恢复180000步任务。沿用同一55页技术PPT及现有网页。

[身体先验](http://127.0.0.1:8870/body-smplx.html#vposer-ready) · [运动与优化预算](http://127.0.0.1:8870/body-smplx.html#motion-convergence-018)
<!-- END_CURRENT_GAUSSIANAVATARS_SMPLX -->'''
text,count=re.subn(r'<!-- CURRENT_GAUSSIANAVATARS_SMPLX -->.*?<!-- END_CURRENT_GAUSSIANAVATARS_SMPLX -->',lambda _:new_top,text,count=1,flags=re.S)
assert count==1;plan.write_text(text,'utf-8')
print(json.dumps({'reviewed':['017','018'],'accepted':False,'updates':14200,'page':str(page)},ensure_ascii=False))
