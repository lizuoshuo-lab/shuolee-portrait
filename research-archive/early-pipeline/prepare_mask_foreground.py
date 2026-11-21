from pathlib import Path
import shutil,json,datetime,hashlib
L=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab')
B=L/'outputs/mini-splatting2-mip-depth-001/shuolee';D=B/'current/static-white-edge-audit-068/mask-photometry'
R=B/'current/person-static-mask-foreground-fresh-070';S=B/'current/person-clothing-pipeline-fresh-060'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
assert not R.exists(),'Never overwrite an experiment'
diag=read(D/'diagnosis.json');test=read(D/'regression-check.json')
src=L/'repos/PGSR-shuolee/shuolee_static.py'
assert diag['completed'] and diag['model_unchanged'] and test['passed']
assert test['adapter_sha256']==sha(src)
stamp=datetime.datetime.now().astimezone().isoformat(timespec='seconds')
protocol=dict(created_before_training=stamp,reference='064',candidate='070',
    problem='Observed person boundary contains opaque strips; white blank ROI pixels are included in the old RGB/SSIM objective.',
    direct_evidence=diag['summary'],
    supporting_evidence='Frozen 064: blank-target DC descent brightens 59.8-92.1 percent of nonzero selected-head coefficient components in six fixed views.',
    alternatives=['Mask/camera conflicts and limited vertical coverage remain possible.',
        'Selected excess head points also explain real foreground; deleting them increases holes.',
        'Removing outside RGB removes some color-based geometric pressure; original alpha may be insufficient.'],
    single_main_change='Photometric support and normalization: foreground-only L1 and fully foreground-supported SSIM.',
    exact_objective=dict(valid_mask='a > 0.99',l1='mean absolute RGB over valid pixels and channels',
        ssim='mean SSIM over centers with all 11x11 pixels valid',weights=[.8,.2],
        invalid_pixels='zero photometric gradient, including SSIM neighborhood paths',
        soft_boundary='excluded from color when a <= .99; unchanged original soft mask still supervises alpha'),
    fixed=['175 actual RGB/masks','59 calibrated and 116 interpolated cameras','1536 full-image height',
        'same fresh fit/200000-point initialization','seed 6060 and 18000-step schedule','RGB resize and targets from064',
        'white renderer background','full-image alpha L1 coefficient .1','scale100, normal .015, geo .03, NCC .15',
        'same densification, pruning and opacity reset','no dynamic field; no model resumed'],
    interpretation='Support selection and per-valid-pixel normalization are one defined foreground objective; not a pure rescaling-only ablation.',
    prediction_support='Reduced boundary RGB/alpha mismatch and outside overcoverage with preserved interior and no new holes in the fixed six observed views.',
    prediction_refute='White color fades but outside alpha persists, holes rise, or strips merely change color; reject as complete repair.',
    protection=dict(same_views=[0,23,69,99,135,174],mean_interior_rgb_mae_max_factor=1.05,
        mean_inner_undercoverage_increase_max_percentage_points=.5,
        any_view_inner_undercoverage_increase_max_percentage_points=1.,
        mean_boundary_alpha_target_factor=.9,mean_outer_overcoverage_target_factor=.8,
        mandatory_visual=['hair, face, collar, sleeve, shoes on black/gray/white','fixed unseen directions','continuous camera path'],
        numeric_thresholds='Predeclared local screening criteria, not statistical significance or independent geometry GT.'),
    stop=['nonfinite loss/gradient or missing frame', 'fresh-init/camera mismatch',
        'obvious catastrophic body collapse at intermediate review; no automatic parameter sweep',
        '18k completed regardless of score; failed acceptance is not extended training'],
    quality_accepted=False,pc00_deployment_authorized_after_local_acceptance_only=True)
save(R/'analysis-before-training.json',protocol)
inputs=[]
for rel in ['data/inputs.json','data/training-inputs.json','data/observations.npz','inference/prediction-postprocessed.npz','inference/prediction-raw.npz','inference/vitpose-keypoints.pt','inference/time-camera-inputs.npz','preparation.json','mask-audit.json','feature-contract.json']:
    src_in=S/rel;dst=R/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src_in,dst);inputs.append(dict(file=rel,sha256=sha(src_in)))
save(R/'source-inputs.json',dict(created=stamp,inputs=inputs,from_scratch=True,old_subject_fit_loaded=False,
    old_gaussian_loaded=False,old_optimizer_loaded=False,real_frames=175,initial_points=200000,steps=18000,
    representation='shared_static_world',dynamic_route_abandoned=True,reference_round='064',
    controlled_change=protocol['single_main_change'],quality_accepted=False))
frozen=R/'source-final';frozen.mkdir();shutil.copy2(src,frozen/src.name)
save(frozen/'manifest.json',dict(created=stamp,files={src.name:sha(src)}))
shutil.copy2(D/'regression-check.json',R/'mask-loss-preflight.json')
st=read(B/'work-status.json');st.update(updated=stamp,current_phase='static_foreground_only_photometry',
    current_decision='先完成只读归因；070仅改颜色损失为mask内部，透明度及几何项不变。全流程从头，未验收。',
    current_experiment='070 foreground-only photometry',local_training_running=False,
    next_action='重新拟合和初始化，核对与064条件一致，再训练18k并做轮廓/主体/未知视角验收。')
st['latest_local_execution']=dict(kind='static_mask_foreground_070',directory=str(R),state='fresh_preparation',running=False,from_scratch=True,quality_accepted=False)
st.setdefault('iteration_policy',{}).update(new_training_ready=True,protocol=str(R/'analysis-before-training.json'),read_only_diagnosis=str(D/'diagnosis.json'))
save(B/'work-status.json',st)
p=B/'long-term-plan.md';t=p.read_text(encoding='utf-8-sig')
old='下一步是只读归因：固定头后、脸前、衣领、鞋边，按真实合成贡献追踪问题高斯，在已标定和插值相机分别核查遮挡与轮廓支持。不能仅凭点中心投影或相反梯度判定相机错误。若只支持监督稀释，下一次仅改变轮廓归一化；若支持投影或分割冲突，先修复输入。070尚未启动，不扫描权重。每轮仍提供标准+Y PLY，并分别记录格式兼容与质量验收。'
new='只读归因已完成：用真实遮挡合成贡献追踪064/069边界点，部分点同时解释人物前景；直接删除会增加空洞。175帧训练ROI中69.64%为空白，旧颜色损失确实包含它们；六视图空白贡献11.42%的L1，不能仅凭面积判定其主导。按用户最新要求，070仅把颜色监督改为mask内部：L1使用a>0.99的像素并按有效像素归一化，SSIM只使用完整11×11前景窗口；原始soft mask的全图alpha约束仍保留。外部RGB随机改动不改变损失或内部梯度，外部颜色梯度严格为零，预检通过。相机、重采样、几何项、初始20万点和18k预算不变，禁止叠加069双背景组合。先重新拟合/初始化并核对，后训练和审阅；尚未证明修复。每轮仍提供标准+Y PLY，格式与质量分别验收。'
assert old in t;p.write_text(t.replace(old,new,1),encoding='utf8')
print(R)
