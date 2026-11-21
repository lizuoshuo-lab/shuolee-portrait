from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,html,shutil
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';R=T/'shuolee/current/robust-portrait-122'
O=REC/'grayT-face-observation-cross-check';C=H.parent/'shuolee-coordination'
L=Path('D:/00/工作日志/2026-10-29/人物对齐核验123')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat();a=read(O/'result.json');assert a['source_files_unchanged']
summary=('00新面部观测对照已只读回放：同一物理皮肤标记的35个原留出视角，从全身RTMW驱动逐帧拟合的中位5.119、最大9.569原生像素，降到脸部FAN驱动的中位2.417、最大3.797像素，但32/35仍≥1像素。独立公式回放差小于0.000058像素，95帧身份、K换算和原观测划分一致，未重新拟合。两条路线仍使用每帧19个头、颈、下颌、表情变量，共1805变量；40–80帧头部参数最大相对差约24.69°与25.79°，不能称已测准的小幅微动或静态模型。当前权重只覆盖鼻部852次、嘴部1900次，眼部和眉部均为0。FAN拟合报告中的1.039像素属于899×1600工作坐标，原生面部语义残差中位实际1.757像素。六个原生裁片显示侧脸鼻嘴观测确有变化，但语义点和重复使用的一处皮肤标记不足以证明全脸材料准确。相机、模板和真实运动仍未唯一分离，本静态路线未准入。')
next_text=('不重复已完成的FAN/RTMW拟合，也不把逐帧变形结果换成静态模型发布。固定相机下，一处皮肤材料点的静态一致性此前已经失败，单纯换检测器不能消除这一证据。下一步检查尚未完成的R7原生残余相机训练入口与调度接入，推进可独立完成的工程；同时只读审阅00正在运行的结果，保持它与本静态目标的验收分开。')
observations=['41/49帧两组鼻嘴点总体位置接近，口角和鼻翼仍存在差异。','57帧可见闭眼；当前损失对眼眉没有任何有效观测，不能据鼻嘴值宣称眼部对齐。','65/73/79帧侧脸的FAN鼻翼与嘴部相对RTMW有明显变化，存在不可见侧语义推测，未认证为物理材料真值。','第四列为同一眼角附近皮肤标记；橙色回放较接近，但仍有大于1px的偏差。','RTMW原运行未保留fit-script.py，FAN执行源码与报告哈希吻合，不能声称两次历史运行严格单因素复现。']
review=dict(updated=stamp,actual_visual_review=True,frames=[41,49,57,65,73,79],original_pixel_crops_unscaled=True,summary=summary,observations=observations,quality_accepted=False)
save(O/'actual-review.json',review)
idx=read(O/'index.json');idx['images']=[]
for item in idx['gallery']:
 idx['images'].append(dict(frame=f'frame_{item["frame"]:06d}',region='已完成对照的只读回放；仍未通过静态对齐',file=item['file'],raw=f'frame_{item["frame"]:03d}_raw.png',columns=['原图','RTMW有效鼻嘴点','FAN有效鼻嘴点','同一皮肤点：绿圈观测／紫叉原对照／橙叉FAN对照'],source=item['source'],source_sha256=item['source_sha256'],crop=item['crop'],actual_visual_review=True,image_sha256=sha(O/item['file'])))
save(O/'index.json',idx)
con=dict(updated=stamp,summary=summary,next=next_text,read_only_result=str(O/'result.json'),active_regions=a['active_regions'],comparisons={k:{x:v[x] for x in ['held','anchors','replay_max_abs_px','semantic_residual_native_px','head_parameters_skin_window','neck_parameters_skin_window','per_frame_parameter_count']} for k,v in a['variants'].items()},limitations=a['limitations']+['RTMW has no archived fitter source; no strict single-factor reproducibility claim','One reused physical mark is secondary validation, not a newly sealed generalization set'],baseline_note=a['baseline_comparability'],observations=observations,quality_accepted=False,camera_admitted=False,static_training_admitted=False,training_started=False,evidence=[str(O/p) for p in ['plan.json','result.json','actual-review.json','index.json']])
save(REC/'face-observation-conclusion.json',con)
for qpath,stage in [(REC/'queue-state.json','face-observation-control-reviewed'),(R/'queue-state.json','local123_face_observation_control_reviewed')]:
 q=read(qpath);q.update(updated=stamp,current_stage=stage,next=next_text,next_action=next_text,latest_diagnosis=str(REC/'face-observation-conclusion.json'),training_allowed=False,training_started=False,quality_accepted=False);save(qpath,q)
n=read(REC/'next-action.json');n.update(updated=stamp,next=next_text,next_action=next_text,latest_diagnosis=str(REC/'face-observation-conclusion.json'),latest_actual_checkpoint_review=str(O/'actual-review.json'),completed_this_turn=['Replayed saved controls without optimization','All35nativeHELD errors retained','Zero eye and brow weights confirmed','Six unscaled native views reviewed'],diagnostic_DOM_verified=False,actual_UI_verified=False,next_plan=None);save(REC/'next-action.json',n)
e=read(REC/'evidence-ledger.json');identifier='grayT123-face-observation-cross-check';assert not any(x.get('id')==identifier for x in e['entries'])
e['entries'].append(dict(id=identifier,updated=stamp,summary=summary,evidence=str(REC/'face-observation-conclusion.json'),quality_accepted=False));e['updated']=stamp;save(REC/'evidence-ledger.json',e)
e=read(R/'evidence-ledger.json');e['updated']=stamp;e['confirmed'].append(summary);e['artifacts'].append(str(REC/'face-observation-conclusion.json'));save(R/'evidence-ledger.json',e)
c=read(C/'local-research-status.json');c.update(updated_at=stamp,next=next_text,current_local_face_observation_audit=con);save(C/'local-research-status.json',c)
lp=T/'shuolee/long-term-plan.md';s=lp.read_text(encoding='utf-8');marker='### 2026-10-29 新面部观测结果回放';assert marker not in s;lp.write_text(s+'\n\n'+marker+'\n\n'+summary+'\n\n下一步：'+next_text+'\n',encoding='utf-8')
(L/O.name).mkdir(exist_ok=True)
for f in O.iterdir():
 if f.is_file():shutil.copy2(f,L/O.name/f.name)
shutil.copy2(REC/'face-observation-conclusion.json',L/'face-observation-conclusion.json')
section='<section id="face-observation"><h1>换面部观测有改善，静态对齐仍未通过</h1><p>'+html.escape(summary)+'</p><table><thead><tr><th>同一35留出视角</th><th>中位误差</th><th>最大误差</th><th>≥1像素</th></tr></thead><tbody>'
for key,label in [('dynamic-face-control','RTMW驱动逐帧拟合'),('dynamic-face-fan-control','FAN驱动逐帧拟合')]:
 v=a['variants'][key]['held'];section+=f'<tr><td>{label}</td><td>{v["median"]:.3f}px</td><td>{v["maximum"]:.3f}px</td><td>{v["ge1"]}/35</td></tr>'
section+='</tbody></table><p>这是一处重复使用的物理标记，不覆盖眼鼻嘴全部材料，也不是新未见视频的泛化验证。原RTMW执行源码未保留，不称为严格单因素实验。</p>'
for im in idx['images']:section+='<figure><figcaption>'+html.escape(im['frame']+' / '+' / '.join(im['columns']))+'</figcaption><img src="'+O.name+'/'+im['file']+'"></figure>'
section+='<p>下一步：'+html.escape(next_text)+'</p></section>'
log=L/'index.html';s=log.read_text(encoding='utf-8');assert 'id="face-observation"' not in s;pos=s.index('</style>')+len('</style>');log.write_text(s[:pos]+section+s[pos:],encoding='utf-8')
save(REC/'face-observation-completion.json',dict(updated=stamp,state='evidence_recorded_publication_pending',daily_log=str(log),actual_native_image_review=True,training_started=False,quality_accepted=False,diagnostic_DOM_verified=False,browser_pixel_review=False,PPT_updated=False,previous_goal_turn='progress'))
print(json.dumps(dict(conclusion=str(REC/'face-observation-conclusion.json'),daily_log=str(log)),ensure_ascii=False))
