from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
from PIL import Image,ImageDraw,ImageFont
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-hand-body-alignment-101';P=L/'outputs/dynamic-human-enerf-hand-visibility-100'
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=read(O/'summary.json');r=read(O/'ray-occluders/summary.json');assert s['source_checkpoints_unchanged'] and r['source_checkpoint_unchanged'];assert len(r['records'])==3
# Reformat annotation headers without changing native photo/overlay pixels.
visual=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12)
for rec in s['projection_panels']:
 p=O/rec['file'];a=Image.open(p);arr=a.crop((0,48,a.width,a.height));out=Image.new('RGB',(max(480,a.width),arr.height+48),'white');out.paste(arr,(0,48));d=ImageDraw.Draw(out)
 d.text((3,2),f"{rec['camera']}/{rec['frame']} | Source and projected model comparison",font=font,fill='black')
 d.text((3,17),'Blue: MANO; orange: SMPL wrist/hand; purple: HaMeR candidate',font=font,fill='black')
 d.text((3,32),'Native pixels. Model comparison, not ground truth or reconstruction.',font=font,fill='black')
 target=p.with_name(p.stem+'-labeled.png');out.save(target)
 if rec['frame'] in [720,726,732]:visual.append(dict(camera=rec['camera'],frame=rec['frame'],file=str(target.relative_to(O)),sha256=sha(target)))
assert len(visual)==9
save(O/'visual-review.json',dict(status='nine_native_training_crops_model_projections_inspected',files=visual,
 observations=['Targetis light-clothed person in all105training contexts, confirmed by sourcephotos.','Body wrist andMANOroot areclose butnot coincident; displayedpose/wrist differences remain.','MANOfinger lines do not consistently trace source finger contours; closedhand/toy occlusion andopenhandshape areuncertain.','Overlayis a modeldiagnostic only, not native rendered reconstruction.'],overlay_is_reconstruction=False,clarity_accepted=False))
frozen=[sum(v for k,v in q['group_mean_weights'].items() if k.startswith('frozen')) for q in r['records']]
active=[q['group_mean_weights']['associated1937']+q['group_mean_weights']['other_context4393'] for q in r['records']]
public=dict(status='105training_alignment_and48native_pixel_occluder_diagnostics_complete_no_new_training',training_images=105,
 body_MANO_wrist_distance_scene_units_quantiles=s['target_wrist_distance_world_quantiles'],body_MANO_wrist_projection_distance_px_quantiles=s['target_wrist_projection_distance_px_quantiles'],
 target_identity=s['target_person_name'],body_reforward_max_error=s['canonical_reforward_max_error'],
 selected_pixel_contexts=[{k:q[k] for k in ['camera','frame','selected_pixels','group_mean_weights','max_CUDA_error']} for q in r['records']],
 active_context_visible_weight_range=[min(active),max(active)],frozen_visible_weight_range=[min(frozen),max(frozen)],
 caveat='The48rays are selected model-occluded training pixels, not a global statistic or semantic hand GT. SMPL anchor ownership is heuristic. Wrist models are independently fitted, not anatomy ground truth. No model was shifted, replaced or retrained.',
 source_models_unchanged=True,licensed_geometry_exported=False,clarity_accepted=False,goal_complete=False,
 next_action='Qualify MANO pose/depth against training-photo contours and multi-view evidence before another reconstruction trial. Use traced contribution evidence to define a persistent, spatially qualified old/new competition control; do not repeat an opacity-only initialization or blindly snap MANO to the body wrist.')
save(O/'diagnostic-summary.json',public)
# Only compact aggregate data are public; licensed geometry/traces remain local.
p=L/'experiments/37_dynamic_human_reconstruction.md';txt=p.read_text(encoding='utf8');assert '### 人体与手部坐标、逐像素遮挡核对' not in txt
txt+='''\n\n### 人体与手部坐标、逐像素遮挡核对\n\n在相同的105张训练照片及估计拍摄时刻下，重新计算身体和MANO关节。身体前向结果与原参考几何一致，手模型对应浅色上衣人物，没有两个人之间的身份错配；两套模型的手腕距离中位数为0.0209场景单位，投影距离中位数为8.66像素。这是独立拟合模型之间的差异，不能据此判断哪一套是真实关节位置。三个训练机位的720、726、732帧投影检查仍能看到指尖和源照片轮廓的偏差，因此没有把手模型直接平移到身体手腕上。\n\n随后从03、04、13机位732帧分别选取16个模型判定为新增表面被遮挡的像素，按渲染器实际排序逐项重算不透明度合成，与CUDA输出的最大差异小于0.0000004。原6330点手部邻域的累计可见权重约为0.77至0.80，其他冻结高斯约为0.20至0.22；冻结贡献主要关联同一人物的手、腕和前臂。这些关联来自身体蒙皮锚点，不是物体语义标签；48个选定像素也不代表整个数据集。该诊断说明，新增手部表面的图像贡献同时受到旧邻域、冻结表面和自身姿态误差影响，单次调整初始化不能代替共同几何与可见性优化。[诊断摘要](../outputs/dynamic-human-hand-body-alignment-101/diagnostic-summary.json)。\n'''
p.write_text(txt,encoding='utf8')
p=Path('D:/Projects/EngineLab/engine-home/scripts/sync-gs-lab.mjs');txt=p.read_text(encoding='utf8');needle="outputs/dynamic-human-enerf-hand-visibility-100/${name}";lines=txt.splitlines(keepends=True);i=next(i for i,l in enumerate(lines) if needle in l)+1
lines.insert(i,"  ['outputs/dynamic-human-hand-body-alignment-101/diagnostic-summary.json', '人体手部对齐与遮挡诊断'],\n");p.write_text(''.join(lines),encoding='utf8')
p=P/'published-resources.json';resources=read(p);resources['paths'][O.name]=['diagnostic-summary.json'];save(p,resources)
p=L/'manifests/dynamic-human-work.json';m=read(p);m.update(status='100paired_training_and101geometry_occlusion_audit_complete_clarity_not_accepted',current_runs=[],next_action=public['next_action'],updated_at=datetime.now().astimezone().isoformat(),last_recorded_utc=datetime.now(timezone.utc).isoformat())
m['outdoor_hand_body_alignment']=dict(stage='101',summary='outputs/'+O.name+'/diagnostic-summary.json',status=public['status'],adopted=False,licensed_geometry_kept_local=True)
line='101:105trainingcontextSMPL/MANO wrist alignment and48native ray contributions verified; no identity swap, poseandocclusion remain unresolved; no newtraining or clarityclaim.'
if line not in m['completed']:m['completed'].append(line)
save(p,m);shutil.copy2(__file__,O/Path(__file__).name);print(json.dumps(dict(active_context_weight_range=public['active_context_visible_weight_range'],frozen_weight_range=public['frozen_visible_weight_range'],clarity_accepted=False)))
