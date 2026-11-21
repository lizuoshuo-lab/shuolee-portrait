from pathlib import Path
from datetime import datetime
import json,hashlib
BASE=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001\shuolee')
ROOT=BASE/'current/smplx-gvhmr-shared-world-fresh-025'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
now=datetime.now().astimezone().isoformat(timespec='seconds')
review=read(ROOT/'review/review-manifest.json');bounds=read(ROOT/'parameter-bounds-audit.json')
joint=read(ROOT/'joint-observation-audit.json');vit=read(ROOT/'vitpose-joint-observation-audit.json')
assert max(review['starting_state_max_abs'].values())==0
conclusion=dict(created=now,state='shared_world_hypothesis_not_adopted',
    new_body_updates=2400,new_body_updates_this_goal_turn=4800,new_gaussian_updates=0,
    all_runs_from_scratch=True,initial_vertices_and_parameters_identical=True,
    statistics=review['statistics'],parameter_bounds=bounds['results'],
    findings=['Shared-world body lowers source-conditioned texture residual, as expected for previously selected static features.',
      'Whole-body image median10.07->12.01px; shoulder/knee errors increase; side body shape is not corrected.',
      'Lower-limb and torso pose bounds are not near saturation. Wrist components near0.97 of combined bound are a separate issue.',
      'Existing image-joint static fits still have left/right knee14.72/16.18px median; excluded time blocks21.88/22.49px.',
      'Independent ViTPose observations differ from existing detector at elbows22.60/23.34px and knees26.42/19.74px. Neither detector is GT; ViTPose left knee excluded-block error is31.94px, not a blanket improvement.',
      'Source-conditioned motion residual and low per-frame image error alone cannot select correct absolute body geometry.'],
    visual_review=dict(full_source_frames=[0,11,23,45,58],fixed_world_views=['left90'],
        fixed_time_snapshot_views=['left90'],time_snapshots=[0,11,23,33,45,58],
        full_continuous_video_accepted=False,all59_source_images_rendered_but_not_all_visually_reviewed=True),
    body_geometry_accepted=False,independent_GT=False,remote_deployed=False,viewer_replaced=False,
    next='Inspect real-image shoulder/elbow/knee observations from existing detector versus official ViTPose; verify pixel/crop conventions and visible-joint reliability before testing observation uncertainty. Neither output is ground truth. Fresh controlled fitting only; no blind budget increase or full-body freezing.',
    retention='Keep023 component baseline and025 shared state only as necessary diagnostic control until observation audit; remove024/025 free failed arrays and redundant exports after preserving compact evidence. Do not initialize next experiment from learned states.')
save(ROOT/'phase-conclusion.json',conclusion)
state=read(BASE/'work-status.json');assert state['current_phase']=='gvhmr_shared_world_fresh_025'
state['updated']=now;state['current_decision']='024/025共4组从头各1200步：纹理运动改善，但共享姿态没有修正侧面几何。两套检测器在肘/膝位置分歧约20–26px，下一步先核对实际观测和不确定性，不直接换检测器当真值。无高斯更新、不部署00。'
state['latest_local_execution']=dict(kind='gvhmr_shared_world_fresh_025',state='shared_world_hypothesis_not_adopted',running=False,
    directory=str(ROOT),arms=['free','shared'],steps_per_arm=1200,new_body_fit_updates=2400,
    new_body_fit_updates_this_goal_turn=4800,gaussian_updates=0,accepted=False)
state['gvhmr_shared_world_025']=dict(conclusion=str(ROOT/'phase-conclusion.json'),new_body_updates=2400,new_gaussian_updates=0,body_geometry_accepted=False)
state['local_training_running']=False
state['current_experiment'].update(latest_fitting_trial='025',latest_fitting_steps=1200,latest_fitting_arms=2,
    latest_fitting_gaussian_steps=0,current_training_running=False,candidate_accepted=False,
    status='observation_consistency_audit_required',current_candidate=state['current_experiment']['current_candidate'])
save(BASE/'work-status.json',state)
plan=BASE/'long-term-plan.md';text=plan.read_text(encoding='utf-8-sig')
start=text.index('下一步先检验躯干和下肢是否被逐帧自由度误解释为运动：')
end=text.index('\n\n通过裸网格',start)
new='''024先核对284条原始身体纹理的运动对应，再对有可靠单骨骼支持的骨盆、上胸和腿部加入纹理项。中段脊柱权重不足0.65，未放宽门槛凑点。两组从头各1200步，控制组逐数组及文件哈希复现023。55点参与、53点930次观测仅诊断，后者24.23→2.97px；身体关节点8.71→10.06px，左膝7.58→17.71px。相对运动改善而绝对侧面形态未改，不作为完整人体采用。

025进一步检验世界根变换、躯干和下身共享的假设。两组都从官方原始预测的相同SO(3)均值及位移中位数开始，初始网格和参数最大差0；局部1–11号关节和世界根在共享组固定为跨帧公共变量，头颈、锁骨、手臂与手指仍随时间。各1200步完成。纹理诊断2.49→0.95px，身体关节点10.07→12.01px，右肩13.94→26.03px、左膝18.88→28.43px，侧面形态未修正。共享组不采用。膝部和躯干参数远未触及界限，不默认放宽范围或增加步数。

只读观测核验中，独立静态三维点拟合原关节点后，左右膝中位残差仍14.72/16.18px，按连续10帧留出后21.88/22.49px。原检测器与已下载ViTPose-H的肘部位置分歧22.60/23.34px，膝部26.42/19.74px。ViTPose左膝留出残差31.94px，也不是统一更好。这里混合真实小幅运动、检测偏差与相机误差，没有关节GT；不能只用低二维拟合残差证明人体正确。

024与025这次续跑共4800次新身体更新、0次高斯更新。保留023作为必要组件基线，025共享状态仅作下一次观测核验的诊断控制；不以它替换查看器。下一步检查两套检测器在真实肩、肘、膝位置的差异、裁切/像素约定与可见性，再做观测不确定性的从头对照；不直接把ViTPose当真值，不重复整个人体冻结或盲目加步数。所有原素材、相机、官方先验和许可模板保护；继续停止眼镜专项、不部署00。'''
text=text[:start]+new+text[end:];plan.write_text(text,encoding='utf-8')
print(json.dumps({'state':conclusion['state'],'new_body_updates_this_turn':4800,'gaussian_updates':0}))
