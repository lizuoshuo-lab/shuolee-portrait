"""Preserve the actual failed trial preparation; never claim Gaussian steps."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
from PIL import Image
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';R=T/'shuolee/current/robust-portrait-122';O=REC/'grass-exploratory'
ROOT=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-local-surface-123/grass-exploratory')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat()
assert not (ROOT/'gaussian-run').exists()
fit=read(ROOT/'fit/geometry-gate.json');volume=read(O/'surface-feasibility.json');support=read(O/'camera-support-audit.json')
assert not fit['coarse_initialization_gate_passed'] and volume['occupied_voxels']==0
review=read(ROOT/'reviews/fit-decision.json');assert review['actual_visual_review'] and not review['allow_surface_initialization']
save(O/'mask-visual-review.json',dict(updated=stamp,actual_images_reviewed=True,
    frames=['000001','000019','000038','000056','000075','000094'],
    summary='Six full-body source/mask pairs inspected. Principal subject, garment, hair and shoes are retained; front shoes partly merge, small sleeve holes and occluded feet remain uncertain. No gross loss of the person or inclusion of another person in these samples.',
    independent_pixel_accuracy_verified=False,mask_sha256=sha(ROOT/'mask-audit.json')))
for name,src in [('geometry-gate.json',ROOT/'fit/geometry-gate.json'),('fit-visual-review.json',ROOT/'reviews/fit-decision.json')]:shutil.copy2(src,O/name)
for frame in ['000001','000019','000038','000056','000075','000094']:
    shutil.copy2(REC/'grass-exploratory-fit-000000-review'/f'frame_{frame}_body.png',O/f'frame_{frame}_body.png')
summary='草地94帧试训准备已实际运行，但本轮未进入高斯优化：粗人体明显错位，同一组相机下的共同衣物体积为空。'
nexttext='复用本轮已保存的原生特征、相机、遮罩与轨迹，核查长轨迹的实际材料身份、近远景支持和相机几何。不得因背景平均残差小而放行；不重复提取，不插值补相机，不放宽衣物交集阈值。'
result=dict(updated=stamp,state='trial_stopped_before_gaussians_invalid_geometry',summary=summary,
    fresh_camera_frames=94,fresh_prior_frames=94,fresh_body_fit_steps=1200,gaussian_steps=0,gaussian_optimizer_created=False,
    body_fit_median_body_fraction=fit['median_body_fraction'],body_fit_p90_body_fraction=fit['p90_body_fraction'],
    body_fit_weighted_observation_median_px=read(ROOT/'fit/progress.json')['weighted_observation_median_px'],
    camera_fit_observations=support['observations'],camera_foreground_observations=support['foreground_observations'],
    camera_wholly_foreground_tracks=support['groups']['all_inside_mask']['points'],
    surface_occupied_voxels=0,max_common_views_in_registered_volume=volume['max_views_in_common'],
    known='The present candidate and observations do not provide a usable shared static body/clothing initialization.',
    unknown='Physical camera error, material mismatch, and subject movement are not uniquely separated. Mask labels are not verified stationary or physical identities.',
    old_checkpoint_or_optimizer_loaded=False,physical_accuracy_accepted=False,quality_accepted=False,
    failed_coarse_fit_preserved=True,user_authorized_exploratory_training=True,next=nexttext,
    scope='This candidate stopped. The ongoing general reconstruction goal remains active.')
save(O/'result.json',result)
save(O/'progress.json',dict(updated=stamp,stage=result['state'],pid=None,command=None,exploratory=True,
    cameras=94,body_fit_steps=1200,gaussian_steps=0,quality_accepted=False,physical_camera_accepted=False))
for folder in [REC,R]:
    q=read(folder/'queue-state.json');q.update(updated=stamp,current_stage=result['state'],active_subject=None,current_subject='grass',
        worker_pid=None,worker_command=None,training_running=False,training_started=False,gaussian_steps=0,
        next=nexttext,latest_trial=str(O/'result.json'),goal_active=True)
    save(folder/'queue-state.json',q)
q=read(REC/'evidence-ledger.json');q['updated']=stamp
q['entries']=[x for x in q['entries'] if not (isinstance(x,dict) and x.get('id')=='grass_exploratory_trial')]
q['entries'].append(dict(id='grass_exploratory_trial',updated=stamp,summary=summary,evidence=str(O/'result.json'),gaussian_steps=0));save(REC/'evidence-ledger.json',q)
q=read(R/'evidence-ledger.json');q['updated']=stamp
q['confirmed'].append(summary)
for name in ['result.json','fit-isolation.json','surface-feasibility.json','camera-support-audit.json','fit-visual-review.json']:
    if str(O/name) not in q['artifacts']:q['artifacts'].append(str(O/name))
save(R/'evidence-ledger.json',q)
plan=T/'shuolee/long-term-plan.md'
with plan.open('a',encoding='utf-8') as f:
    f.write('\n\n### 2026-10-29 草地94帧探索性试训\n\n'+summary+' 已重新求94帧相机、推理本人物先验并做1200步粗拟合。身体观测中位约59.97px，远非1px。固定相机的自由身体点也不一致；已登记2.638M体素中最多75/94帧支持，97%门槛下无占用。实际相机拟合的50396条观察只有25条落在人物mask内，不能把其整体小残差当成人像精度。此相机候选保留并停止，没有新PLY。'+nexttext+'\n')
# A compact scientific figure, no labels burned into images. Two actual views,
# raw left / mesh overlay right, preserve aspect and the same crop per pair.
ims=[Image.open(O/f'frame_{f}_body.png').convert('RGB') for f in ['000001','000038']]
canvas=Image.new('RGB',(576,384),'white')
for j,im in enumerate(ims):
    ratio=min(280/im.width,384/im.height);im=im.resize((round(im.width*ratio),round(im.height*ratio)),Image.Resampling.LANCZOS)
    canvas.paste(im,(j*288+(288-im.width)//2,(384-im.height)//2))
figure=O/'trial_source_and_failed_mesh.png';canvas.save(figure)
save(O/'slide-review.json',dict(actual_images_reviewed=False,figure=str(figure),
    title='草地人物试训：初始化未成立',subtitle='94帧相机与人体初值从头重建',
    result='粗人体明显错位；共同衣物体积为空。本轮未启动高斯优化，不能称约1像素对齐。',
    caption='两组视角，每组左为原图、右为SMPL-X投影。相同裁片与比例，图上未加文字。',
    notes=[summary,'1200步粗拟合后有效身体观察中位59.97px。相机拟合50396条观察中仅25条位于人物mask内；这是图像区域支持，不是物理身份真值。',
           '固定相机下分别拟合静态身体点，中位误差17.50至81.52px，不能单独归因到SMPL-X。登记范围2,637,985体素，2px遮罩容差、97%视角交集，0个占用体素；最多75/94视角支持。全94包围框视锥也无共同正深度点。',
           '没有放宽阈值、训练高斯、生成PLY或替换旧查看器模型。相机、材料对应与人物微动仍未唯一分离。',nexttext]))
print(json.dumps(result,ensure_ascii=False))
