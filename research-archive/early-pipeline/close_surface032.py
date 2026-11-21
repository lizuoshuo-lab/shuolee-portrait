from pathlib import Path
import datetime,hashlib,json,os,shutil
import numpy as np
from PIL import Image

LAB=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab')
BASE=LAB/'outputs/mini-splatting2-mip-depth-001/shuolee'
RUN=BASE/'current/smplx-absolute-surface-fresh-032'
AUDIT=BASE/'current/smplx-absolute-surface-depth-031'
VIEWER=BASE.parent/'viewer'
NOW=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def load(p):
    with np.load(p) as z:return {k:z[k].copy() for k in z.files}

assert not (RUN/'phase-conclusion.json').exists()
review=read(RUN/'review/comparison.json');audit=read(AUDIT/'depth-audit.json')
assert max(review['control_replay_vs_023'].values())==0
assert review['initial_arrays_identical']
ev=load(RUN/'review/depth-evaluation.npz');obs=ev['observations'];s0=audit['world_per_nominal_model_m']
which=ev['reliable']&np.isin(obs[:,8],[1,2,3])&~ev['fitmask']
z0,z1=ev['depth_control'],ev['depth_depth-order'];band=np.maximum(obs[:,5],.01*s0)
common=which&np.isfinite(z0)&np.isfinite(z1)
paired=dict(observations=int(which.sum()),common_hits=int(common.sum()),
    violations_control=int((common&(obs[:,4]-z0>band)).sum()),
    violations_candidate=int((common&(obs[:,4]-z1>band)).sum()),
    lost_intersections=int((which&np.isfinite(z0)&~np.isfinite(z1)).sum()),
    gained_intersections=int((which&~np.isfinite(z0)&np.isfinite(z1)).sum()))
st=review['statistics'];delta=100*(st['depth-order']['mesh_volume_in_common_nominal_units']/st['control']['mesh_volume_in_common_nominal_units']-1)
next_action='停止单向深度权重扫描。先核对不同先验与标定相机的三维配准、衣物遮挡下体形和关节深度的可辨识性；SAM3DBody授权权重到位后进行原图独立预测。人体姿态与几何未通过前，不启动20万点/18000步高斯训练。'
summary=dict(created=NOW,state='depth_order_component_improved_body_geometry_not_accepted',
    hypothesis='Absolute exterior-surface depth can prevent a predicted bare body from protruding in front of observed clothing.',
    from_scratch=True,old_person_initialization=False,arms=2,steps_per_arm=1200,all59_real_images=True,
    initialization_arrays_identical=True,control_equals_preserved023_exactly=True,
    trained_regions='Torso/upper-leg/lower-leg image-height bins; these are not anatomical labels',
    reliable_original_tracks=audit['reliable_tracks'],reliable_original_observations=audit['reliable_observations'],
    train_tracks=62,diagnostic_tracks=60,unoptimized_matched_rays=paired,
    diagnostic_violations={a:st[a]['exterior']['unoptimized_diagnostic']['violations'] for a in st},
    diagnostic_intersections={a:st[a]['exterior']['unoptimized_diagnostic']['hits'] for a in st},
    body_2d_median_px={a:st[a]['body_2d_px']['median'] for a in st},
    frame23_knee_flexion_degrees={a:{k:v['frame23'] for k,v in st[a]['model_knee_flexion_degrees_not_GT'].items()} for a in st},
    frame23_mesh_volume_change_percent=delta,
    visual_observation='Source overlays and fixed world front/left/right views show limited changes. Side knee flexion, upper-arm placement, and torso shape remain insufficient. No visual geometry breakthrough.',
    conclusion='Absolute depth ordering reduces one observed inconsistency, but a bare body can still be too small or have wrong articulation while staying behind clothes.',
    limitations=['Fixed SfM cameras and source features are not independent ground truth.',
        'Nominal model meters use a fitted scale, not a measured physical scale.',
        'Diagnostic point IDs were already inspected. This is not an unseen-view or independent subject test.',
        'Training uses periodic triangle selection, checked by exact final ray intersections.',
        'A smaller enclosed volume and fewer ray intersections require caution; lower depth loss alone cannot establish anatomy.'],
    next=next_action,quality_accepted=False,gaussian_updates=0,viewer_model_replaced=False,remote_deployment=False,
    evidence=['review/six-source-comparison.jpg','review/fixed-world-comparison.jpg','review/source-000045.jpg','review/comparison.json'])
save(RUN/'phase-conclusion.json',summary)
save(RUN/'visual-review.json',dict(reviewed=NOW,
    inspected=['All six full source overlays 0/11/23/33/45/58','Fixed world23 front/left90/right90 actual mesh renders',
        'Original exterior track source23/source45 and six-view audit sheet'],
    result=summary['visual_observation'],quality_accepted=False))
review.update(state='visually_reviewed_not_promoted',quality_accepted=False,paired_depth_diagnostic=paired)
save(RUN/'review/comparison.json',review)
for arm in ['control','depth-order']:
    q=RUN/arm/'run-state.json';d=read(q);d.update(state=summary['state'],accepted=False,running=False);save(q,d)
save(RUN/'run-state.json',dict(created=NOW,state=summary['state'],running=False,body_fit_updates=2400,gaussian_updates=0,accepted=False))
a=read(AUDIT/'run-state.json');a.update(state='read_only_depth_support_reviewed',running=False,accepted=False);save(AUDIT/'run-state.json',a)

# Preserve full evidence panels when assembling a single report illustration.
left=Image.open(RUN/'review/source-000045.jpg').convert('RGB')
right=Image.open(RUN/'review/fixed-left90.jpg').convert('RGB')
right=right.resize((round(right.width*left.height/right.height),left.height),Image.Resampling.LANCZOS)
sheet=Image.new('RGB',(left.width+24+right.width,left.height),'white')
sheet.paste(left,(0,0));sheet.paste(right,(left.width+24,0));sheet.save(RUN/'review/geometry-comparison.jpg',quality=95)

status=read(BASE/'work-status.json')
status.update(updated=NOW,current_phase='absolute_surface_depth_032',local_training_running=False,
    current_decision=summary['conclusion'],next_action=next_action)
status['body_absolute_surface_032']={**summary,'report':str(RUN/'phase-conclusion.json')}
status['latest_local_execution']=dict(kind='absolute_surface_depth_032',state=summary['state'],running=False,
    directory=str(RUN),from_scratch=True,arms=2,body_fit_updates=2400,gaussian_updates=0,accepted=False)
current=status['current_experiment'];current.update(latest_fitting_trial='032',latest_fitting_steps=1200,
    latest_fitting_arms=2,latest_fitting_gaussian_steps=0,candidate_accepted=False,current_training_running=False,
    status='depth_order_component_improved_geometry_not_accepted')
save(BASE/'work-status.json',status)
plan=BASE/'long-term-plan.md';text=plan.read_text(encoding='utf-8-sig')
anchor='通过裸网格原图叠加、固定时刻多方向与时间一致性后，再从20万新高斯和新优化器训练18000步。'
assert text.count(anchor)==1
para=(f'031/032核对原始静态一致纹理的绝对深度，并完成两组从头1200步对照。312条原始稳定轨迹中，182条、3385次观测通过保守深度支持条件；每次深度排除所在连续10帧区块后重三角化，相机保持原SfM值。衣物和鞋面只提供外表面次序，不强迫裸人体贴合衣物。躯干与腿部62条轨迹参与损失、60条作已见诊断，两组初始化逐数组一致，对照精确重现023。诊断冲突478降至282，命中911降至900；共同命中的{paired["common_hits"]}条射线上冲突由{paired["violations_control"]}降至{paired["violations_candidate"]}。二维残差中位数8.71升至9.91像素，23帧网格体积下降{abs(delta):.2f}%，局部屈膝未改善。实际六视图与固定世界正侧面没有明显几何突破，候选不晋级、无高斯训练。此结果只支持外表面次序约束的局部作用，不证明裸人体解唯一或准确。'+next_action+'\n\n')
plan.write_text(text.replace(anchor,para+anchor),encoding='utf-8')

asset=VIEWER/'body-fitting-method/absolute-surface-depth';asset.mkdir(exist_ok=True)
for name in ['geometry-comparison.jpg','six-source-comparison.jpg','fixed-world-comparison.jpg']:
    dest=asset/name;assert not dest.exists();shutil.copy2(RUN/'review'/name,dest);assert sha(dest)==sha(RUN/'review'/name)
save(asset/'conclusion.json',summary)
page=VIEWER/'body-smplx.html';html=page.read_text(encoding='utf-8-sig');before=sha(page)
marker='</details>\n</section>\n<!-- GVHMR_PRIOR_019_END -->';assert html.count(marker)==1
addition=f'''<h4 id="body-absolute-depth-032">衣物深度只能约束身体不能穿出的方向</h4>
<p>图像中可见的外套、裤子和鞋面位于人体外部。沿同一观测射线，裸人体表面应在它们之后，允许原始纹理三角化的不确定性。这给出了单向的深度顺序约束。直接把人体顶点拉到衣物纹理点，会把衣物厚度和褶皱误当作人体形状。</p>
<figure><a href="body-fitting-method/absolute-surface-depth/geometry-comparison.jpg"><img src="body-fitting-method/absolute-surface-depth/geometry-comparison.jpg" alt="左组三列为原图、从头对照、增加外表面深度顺序约束；右组为相同世界相机下的两种身体网格"></a><figcaption>本地实际网格渲染。左组：45帧原图、对照、候选。右组：固定23帧、相同世界相机下的两种侧面。两组各从头拟合1200步，图中没有生成补全。</figcaption></figure>
<p>保守筛选后的躯干与腿部纹理分成62条拟合轨迹和60条诊断轨迹。诊断组在两种身体均有交点的{paired['common_hits']}次观测中，深度次序冲突由{paired['violations_control']}降至{paired['violations_candidate']}。但侧面屈膝和上臂配准仍未改善，二维残差中位数由8.71增至9.91像素，23帧网格体积减小约{abs(delta):.2f}%。这说明单向约束能减少部分穿出，却不能独立确定身体厚度、局部姿态或遮挡侧形状。本轮不升级为高斯训练底模。</p>
<p>这里的诊断点来自同一段已反复检查的视频，相机和纹理也共享SfM估计，不能当作独立三维真值。两组都从原始预测、原相机和原观测重新初始化，未接着旧个体拟合训练。<a href="body-fitting-method/absolute-surface-depth/six-source-comparison.jpg">完整六视图</a> · <a href="body-fitting-method/absolute-surface-depth/conclusion.json">精简判定</a></p>
'''
page.write_text(html.replace(marker,addition+marker),encoding='utf-8')
save(RUN/'publication-receipt.json',dict(created=NOW,page=str(page),before_sha256=before,after_sha256=sha(page),
    url='http://127.0.0.1:8870/body-smplx.html#body-absolute-depth-032',scope='Existing collapsed appendix',
    ppt_changed=False,viewer_model_changed=False,remote_deployment=False))
print(json.dumps(dict(state=summary['state'],paired=paired,volume_change_percent=delta),ensure_ascii=False))
