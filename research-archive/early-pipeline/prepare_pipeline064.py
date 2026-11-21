"""Prepare a fresh PGSR static research comparison; preserve completed deliveries."""
from pathlib import Path
import json, shutil, hashlib, datetime
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
S=B/'current/person-clothing-pipeline-fresh-060'
R=B/'current/person-static-pgsr-fresh-064'
A=B/'current/static-camera-observations-063'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8');t.replace(p)
stamp=datetime.datetime.now().astimezone().isoformat(timespec='seconds')
audit=read(A/'audit.json')
save(A/'conclusion.json',dict(created=stamp,state='diagnostic_completed',summary=audit['summary'],
    conclusion='Background camera residual improves modestly; person residual changes only slightly. This does not establish camera interpolation as the main cause of large tears.',
    selection_set_is_independent_test=False,filtered_correspondence_bias=True,raw_cameras_modified=False,
    new_gaussian_updates=0,candidate_cameras_used_for_training=False,
    next_route='Fresh static PGSR surface regularization using original input cameras.'))
assert not R.exists(),'Do not overwrite a round'
R.mkdir();sources=[]
for rel in ['data/inputs.json','data/training-inputs.json','data/observations.npz','inference/prediction-postprocessed.npz','inference/prediction-raw.npz','inference/vitpose-keypoints.pt','inference/time-camera-inputs.npz','preparation.json','mask-audit.json','feature-contract.json']:
    src=S/rel;dst=R/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    sources.append(dict(file=rel,sha256=hashlib.sha256(src.read_bytes()).hexdigest(),source=str(src),role='Raw observation / official pretrained output; no personalized fit'))
save(R/'source-inputs.json',dict(created=stamp,inputs=sources,old_subject_fit_loaded=False,
    old_gaussian_loaded=False,old_optimizer_loaded=False,representation='shared_static_world',
    real_frames=175,initial_points=200000,steps=18000,
    body_refit_usage='Only broad initialization box and scale; no body warp or body attraction.',
    hypothesis='Does static planar Gaussian surface consistency reduce tearing in the clothed person?',
    reference_repository='https://github.com/zju3dv/PGSR',commit='de24f1a38b350387e8d8fe381b2cd70c1ae946e7',
    license_scope='Local research/evaluation only; commercial PGSR authorization not established.',
    protected_raw_input_dependency=str(S/'data'),quality_accepted=False))
st=read(B/'work-status.json');st['local_training_running']=False
st['latest_local_execution']=dict(kind='static_pgsr_fresh_064',state='preparing_fresh_initialization_and_native_renderer',running=False,
    directory=str(R),representation='shared_static_world',from_scratch=True,quality_accepted=False,
    previous_completed_delivery='062',camera_diagnostic_063=str(A/'conclusion.json'))
save(B/'work-status.json',st)
p=B/'long-term-plan.md';t=p.read_text(encoding='utf-8-sig')
old='下一项：先核验175帧原图特征与相机投影，分别统计59个标定相机和116个插值相机的残差、覆盖与遮挡，区分相机误差、真实微动、遮罩错误。证据支持后再选择从原图重估全帧相机或有界相机优化；下一轮高斯仍重新初始化，不接着062修补。不得只凭训练误差变小宣布新视角质量通过。'
new='063相机诊断完成：175帧原图特征核验中，116个插值相机有79个通过有界背景PnP筛选；插值帧背景保留观测的中位残差1.010→0.691像素，人物1.345→1.313像素。选择集参与筛选，不能称独立测试；特征筛选亦有偏差。没有足够证据认为这点相机误差解释了大面积破碎，原相机与062均保持原样。\n\n当前064：准备从头验证PGSR静态表面约束，175个真实帧共享同一高斯场，新初始化20万点，目标18000步；使用原相机以控制变量，未见视角不作训练真值。官方源码de24f1a，独立编译渲染模块；本地研究许可不等于已取得该项目商业授权。先做原生渲染、标准PLY与完整相机投影核验，再训练。062仍为已交付的静态对照，064在实际完成和质量检查前不晋级。'
assert old in t;t=t.replace(old,new,1).replace('当前训练已完成，GPU没有正在跑下一轮；先做静态相机诊断。','062训练已完成；064初始化与原生渲染接入进行中。',1)
p.write_text(t,encoding='utf8')
print(R)
