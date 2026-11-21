"""Freeze one from-scratch camera-input comparison before any fitting/training."""
from pathlib import Path
import datetime, hashlib, json, shutil, sys
import numpy as np

L=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab')
B=L/'outputs/mini-splatting2-mip-depth-001/shuolee'
A=B/'current/person-static-mask-foreground-fresh-070'
C=B/'current/static-camera-registration-071'
R=B/'current/person-static-registered-camera-fresh-072'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

assert not R.exists(), 'Never overwrite an experiment'
now=datetime.datetime.now().astimezone().isoformat(timespec='seconds')
registered=read(C/'training-inputs-registered.json')
sys.path.insert(0,str(L/'repos/PGSR-shuolee'))
from camera_contract import require_registered_cameras
counts=require_registered_cameras(registered)
assert counts['frames']==175 and counts['kinds']==dict(calibrated=59,registered_background_pnp=116)
assert registered['registration_audit']['sha256']==sha(C/'audit.json')
review=read(C/'review-decision.json')
assert review['camera_correspondence_review_accepted'] and not review['gaussian_retraining_started']
old=read(A/'data/training-inputs.json')
for before,after in zip(old['records'],registered['records']):
    for key in before:
        if key not in ('w2c','camera_kind'):
            assert before[key]==after[key], (before['frame'],key)
    if before['camera_kind']=='calibrated':
        assert before['w2c']==after['w2c']

# Verify no photometric/geometry training implementation changed since070.
old_source=(A/'source-final/shuolee_static.py').read_text(encoding='utf8')
adapted=old_source.replace('import torch.nn.functional as F\n',
    'import torch.nn.functional as F\nfrom camera_contract import require_registered_cameras\n',1)
adapted=adapted.replace("    rows=read(root/'data/training-inputs.json')['records'];assert len(rows)==175",
    "    document=read(root/'data/training-inputs.json');camera_contract=require_registered_cameras(document)\n    rows=document['records'];assert len(rows)==175",1)
adapted=adapted.replace("camera_source='original59 and time-interpolated116;063 candidate poses not used'",'camera_source=camera_contract',1)
assert adapted==(L/'repos/PGSR-shuolee/shuolee_static.py').read_text(encoding='utf8')

protocol=dict(created_before_training=now,reference='070',candidate='072',
    problem='Static learned silhouettes disagree with observed person masks;116 of175 prior training cameras were only temporal interpolation.',
    direct_evidence=dict(camera_registration=str(C/'review-decision.json'),
        camera_counts=counts,held_background=review['held_background'],
        frozen070_contour_px=[2.2172879904364806,2.191883444114796],
        frozen070_head_alpha=[.26956113715742724,.2711828419845807]),
    hypotheses=[
        'Small per-frame camera errors may bias initialization, Gaussian growth and learned boundary placement even when a frozen-model camera swap is weak.',
        'Mask uncertainty, physical subject motion, opacity support and limited vertical observations may dominate; a camera repair need not fix boundaries.'],
    single_input_variable='116 additional camera extrinsics: temporal interpolation replaced by observed background PnP registration.59 original SfM cameras unchanged.',
    fixed=['same175 actual source images and masks','same original official predictions and observations',
        'same1200-step fresh body fit, used only for static initialization proposal box',
        'same silhouette-hull algorithm, weights1.0 for original59 and0.5 for additional116',
        'same200000 points,seed6060,1536 image height,18000 optimizer steps',
        'same foreground-only L1/complete-window SSIM; original full-image alpha0.1',
        'same scale100,normal0.015,geo0.03,NCC0.15,optimizer,densification,pruning',
        'one shared static field; no body-driven deformation; no old learned state'],
    controlled_scope='Camera-input pipeline comparison. Fresh fit, visual-hull sampling and multiview-neighbor choices may change downstream of corrected extrinsics; not an identical-initial-point direct pose ablation.',
    assessment=dict(primary='Both070 and072 rendered under the same071 camera list on all175 images; also retain070 native-old-camera statistics.',
        primary_contour_mean_target_factor=.9,mean_boundary_alpha_target_factor=.9,
        protection=dict(mean_interior_rgb_mae_max_factor=1.05,
            mean_inner_undercoverage_increase_max_percentage_points=.5,
            any_fixed_view_inner_undercoverage_increase_max_percentage_points=1.,
            mean_outer_overcoverage_must_not_increase=True),
        fixed_views=[0,23,69,99,135,174],
        visual=['original image / control / candidate at exact same pixels',
            'hair,face,collar,sleeve,shoe boundaries on gray and black',
            'fixed unseen elevations and lateral views','continuous camera path',
            'standard+Y PLY independent import in Gaussian and point modes'],
        interpretation='Predeclared local screening thresholds; one seed and one scene, not independent geometric GT.'),
    support='Material contour improvement without new holes or lost detail supports correcting per-frame input poses as useful for this pipeline.',
    refute='No material improvement or new holes rejects camera replacement alone as sufficient boundary repair; do not extend steps or scan settings automatically.',
    stop=['missing or altered raw input','unregistered or misindexed camera','nonfinite loss/gradient',
        'catastrophic collapse at1800/6000/9000 review','stop at18000 regardless of score'],
    quality_accepted=False,remote_deployment=False)
save(R/'analysis-before-training.json',protocol)

copied=[]
for rel in ['data/observations.npz','inference/prediction-postprocessed.npz','inference/prediction-raw.npz',
    'inference/vitpose-keypoints.pt','inference/time-camera-inputs.npz','preparation.json','mask-audit.json','feature-contract.json']:
    src=A/rel;dst=R/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    copied.append(dict(file=rel,source=str(src),sha256=sha(src),kind='raw_observation_or_official_prediction'))
save(R/'data/training-inputs.json',registered)
fit=read(A/'data/inputs.json')
for before,after in zip(fit['records'],registered['records']):
    assert before['source_frame_index']==after['source_frame_index']
    before['w2c']=after['w2c'];before['camera_kind']=after['camera_kind']
    if 'camera_registration' in after:before['camera_registration']=after['camera_registration']
fit.pop('camera_interpolation',None)
fit.update({k:v for k,v in registered.items() if k!='records'})
save(R/'data/inputs.json',fit)
require_registered_cameras(fit)
save(R/'source-inputs.json',dict(created=now,inputs=copied,from_scratch=True,
    old_subject_fit_loaded=False,old_gaussian_loaded=False,old_optimizer_loaded=False,
    real_frames=175,initial_points=200000,steps=18000,representation='shared_static_world',
    dynamic_route_abandoned=True,reference_round='070',controlled_change=protocol['single_input_variable'],
    camera_manifest_sha256=sha(R/'data/training-inputs.json'),quality_accepted=False))
frozen=R/'source-final';frozen.mkdir()
files={}
for rel in ['PGSR-shuolee/shuolee_static.py','PGSR-shuolee/camera_contract.py',
    'GVHMR-shuolee/shuolee/fit_fresh_sequence.py','GVHMR-shuolee/shuolee/clothing_surface_support.py',
    'GaussianAvatars_SMPLX-shuolee/initialize_static_clothing_cloud.py']:
    src=L/'repos'/rel;shutil.copy2(src,frozen/src.name);files[rel]=sha(src)
save(frozen/'manifest.json',dict(created=now,files=files,training_math_unchanged_from070=True))
status=read(B/'work-status.json')
status.update(updated=now,current_phase='static_registered_camera_fresh_comparison',
    current_decision='072仅替换新增116帧相机配准，其他训练配置保持070；从头拟合与初始化，先验证再训练。',
    current_experiment='072 registered-camera input comparison',local_training_running=False,
    next_action='完成本轮新初始化核对及原生渲染检查后18k；随后同相机比较070/072轮廓和未知视角。')
status['latest_local_execution']=dict(kind='static_registered_camera_072',directory=str(R),
    state='fresh_preparation',running=False,from_scratch=True,quality_accepted=False)
status.setdefault('iteration_policy',{}).update(new_training_ready=True,
    protocol=str(R/'analysis-before-training.json'),read_only_diagnosis=str(C/'review-decision.json'))
save(B/'work-status.json',status)
plan=B/'long-term-plan.md';text=plan.read_text(encoding='utf-8-sig')
text=text.replace('下一步：先写相机输入单项对照协议，从原始175帧和已验证的逐帧相机重新拟合粗范围、初始化20万点与优化器，再18k。保持070的mask内颜色、透明度和几何项不变，不能复用旧高斯，不自动叠加轮廓权重/增密调整。当前没有新Gaussian训练运行。',
    '072相机输入单项对照协议已预先固定：从原始175帧和071相机重新拟合粗范围、初始化20万点与优化器，再18k。保持070的mask内颜色、透明度、几何项与原始59/新增116帧的初始化权重不变。相机改动导致的新粗范围、点采样和邻居选择属于输入修复的下游效应；不是固定初始化的直接位姿消融。当前处于从头初始化阶段，未宣称改善。')
text=text.replace('其中59个相机已标定、116个为插值；插值不会补充未拍摄的俯仰观测。',
    '原070及更早对照使用59个SfM相机和116个插值相机；当前071已补齐175帧观测配准。相机配准不会补充未拍摄的俯仰观测。',1)
plan.write_text(text,encoding='utf8')
print(json.dumps(dict(run=str(R),cameras=counts,old_learned_state_loaded=False,stage='prepared'),ensure_ascii=False))
