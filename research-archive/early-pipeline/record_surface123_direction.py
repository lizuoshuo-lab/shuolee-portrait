from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,html,shutil
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';R=T/'shuolee/current/robust-portrait-122'
O=REC/'grayT-fixed-direction-review';C=H.parent/'shuolee-coordination'
L=Path('D:/00/工作日志/2026-10-29/人物对齐核验123')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');q.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat();a=read(O/'result-completed.json');m=read(O/'measurements-completed.json');old=read(O/'measurements.json')
assert a['photometric_pass'] and a['source_hashes_unchanged'] and a['camera_or_point_updates']==0
for x,y in zip(old['lines'],m['lines']):
 if x['frame']!=64:assert x==y
summary=('灰衣人物已完成一次冻结相机的栏杆方向控制：56和72帧的6条边缘只用于估计一条公共空间方向，64和80帧的6条边缘、480个采样点留出。'
 '研究侧与00相机的留出方向残差RMS分别0.569和0.593原生像素，最大2.200和2.350像素。这里每条预测线都通过观测中心，检验的是方向，不能当成绝对重投影或人脸误差。'
 '数学负对照表明，绕这条公共方向旋转7°，预测线方向仍保持不变；任何平移也不进入此检验。'
 '原始栏杆边缘的上下半段斜率还相差0.12°至3.88°，其物理直线性和精确平行性未认证。'
 '因此这项控制无法选择哪一套相机正确，也无法把人物微动与外参误差分开。相机和SMPL-X均未修改，正式静态训练仍未准入。')
next_text=('本轮单一方向控制已完成，不追加线段或放宽门槛。下一步只读复核00对话已在运行的灰衣对照及原有皮肤留出，区分它的表观改善与本静态目标的实际准入。'
 '相机仍缺能同时约束旋转和平移的可信固定场景证据；后续原生训练工程只推进尚未完成的部分，不重复已完成的控制或用未合格人物启动训练。')
review=dict(updated=stamp,actual_visual_review=True,summary=summary,quality_accepted=False,
    completed_native_triplets_reviewed=[56,64,72,80],initial_failed_triplets_reviewed=[56,64,72,80],
    observations='原生三列分别为原图、亮度边缘和两套冻结相机的方向预测。杆边整体方向接近，但局部边缘弯折/亮度交界仍明显。初始64帧标注搜索偏离杆边，修正后保持相同物理杆件、行范围、角色和测量规则。',
    initial_annotation_error_owned_by_analysis=True,one_annotation_repair_documented=True,
    no_camera_reprojection_acceptance=True,no_subpixel_material_truth=True)
save(O/'actual-review.json',review)
idx=read(O/'index.json')
for im in idx['images']:
 im['actual_visual_review']=True;im['image_sha256']=sha(O/im['file'])
save(O/'index.json',idx)
con=dict(updated=stamp,summary=summary,next=next_text,runs=a['runs'],controls=a['controls'],
 scope=dict(source_frames=95,measurement_frames=4,anchor_frames=[56,72],held_frames=[64,80],held_lines=6,held_samples=480),
 line_half_slope_difference_range_deg=[min(x['first_second_half_direction_difference_deg'] for x in m['lines']),max(x['first_second_half_direction_difference_deg'] for x in m['lines'])],
 endpoint_half_pixel_jitter_sensitivity_range_deg=[min(x['endpoint_half_pixel_opposite_jitter_bound_deg'] for x in m['lines']),max(x['endpoint_half_pixel_opposite_jitter_bound_deg'] for x in m['lines'])],
 limitations=['Only one assumed common structural direction','Per-line center is fixed from observations, so position is not tested','Yaw around the direction and all translation are unobservable','Conditional K, bar shape and residual image warp remain coupled','Only two held views, not all95views','Not independent face material evidence'],
 annotation_repair=read(O/'seed-annotation-repair.json'),quality_accepted=False,camera_admitted=False,training_started=False,
 evidence=[str(O/n) for n in ['plan.json','result.json','seed-annotation-repair.json','result-completed.json','measurements-completed.json','actual-review.json']])
save(REC/'fixed-direction-conclusion.json',con)
q=read(REC/'queue-state.json');q.update(updated=stamp,current_stage='fixed-direction-control-reviewed',reason=summary,
 latest_diagnosis=str(REC/'fixed-direction-conclusion.json'),training_allowed=False,training_started=False,training_running=False,
 next=next_text,goal_stays_active=True);save(REC/'queue-state.json',q)
n=read(REC/'next-action.json');n.update(updated=stamp,next=next_text,latest_diagnosis=str(REC/'fixed-direction-conclusion.json'),
 latest_actual_checkpoint_review=str(O/'actual-review.json'),completed_this_turn=['Frozen-camera native railing-direction control with separate anchors and held views','Documented image-only annotation repair; nine measurements reused, no camera or point updates','Observable and unobservable degrees of freedom explicitly verified'],
 diagnostic_DOM_verified=False,actual_UI_verified=False,training_allowed=False,next_plan=None);save(REC/'next-action.json',n)
e=read(REC/'evidence-ledger.json');identifier='grayT123-fixed-scene-direction-control'
assert not any(x.get('id')==identifier for x in e['entries'])
e['entries'].append(dict(id=identifier,updated=stamp,summary=summary,evidence=str(REC/'fixed-direction-conclusion.json'),quality_accepted=False));e['updated']=stamp;save(REC/'evidence-ledger.json',e)
e=read(R/'evidence-ledger.json');e['updated']=stamp;e['confirmed'].append(summary);e['artifacts'].append(str(REC/'fixed-direction-conclusion.json'));save(R/'evidence-ledger.json',e)
q=read(R/'queue-state.json');q.update(updated=stamp,current_stage='local123_fixed_direction_control_reviewed',next=next_text,
 latest_evidence=str(REC/'fixed-direction-conclusion.json'),quality_accepted=False);save(R/'queue-state.json',q)
c=read(C/'local-research-status.json');c.update(updated_at=stamp,next=next_text,current_local_fixed_direction_control=con);save(C/'local-research-status.json',c)
lp=T/'shuolee/long-term-plan.md';s=lp.read_text(encoding='utf-8');marker='### 2026-10-29 固定栏杆方向的独立控制'
assert marker not in s;lp.write_text(s+'\n\n'+marker+'\n\n'+summary+'\n\n下一步：'+next_text+'\n',encoding='utf-8')
(L/O.name).mkdir(exist_ok=True)
for f in O.iterdir():
 if f.is_file():shutil.copy2(f,L/O.name/f.name)
shutil.copy2(REC/'fixed-direction-conclusion.json',L/'fixed-direction-conclusion.json')
section='<section id="fixed-direction"><h1>固定栏杆方向不能证明完整相机对齐</h1><p>'+html.escape(summary)+'</p>'
section+='<p>第64帧第一次测量失败来自本次分析的端点标注错误：把阴影边界延伸为杆边。修正仅涉及3个起点的横坐标，保留初始结果，复用其他9条测量。没有更换物理杆件、删除留出或放宽测量门槛。</p>'
section+='<table><thead><tr><th>冻结相机</th><th>两个留出视角方向RMS</th><th>最大偏差</th><th>正式准入</th></tr></thead><tbody>'
for key,label in [('research','研究侧'),('company00','00对话本机复现')]:
 v=a['runs'][key]['held_direction_only_residual_px'];section+=f'<tr><td>{label}</td><td>{v["rms"]:.3f}px</td><td>{v["maximum"]:.3f}px</td><td>未通过</td></tr>'
section+='</tbody></table><p>仅有竖直方向看起来接近，并不能排除偏航和平移误差。当前也未证明实际栏杆精确平行、所有边缘完全直线，不能将本轮归因于相机或鱼眼残余畸变。</p>'
for im in idx['images']:
 if '_completed_' not in im['file']:continue
 section+='<figure><figcaption>'+html.escape(im['frame']+' / '+' / '.join(im['columns']))+'</figcaption><img src="'+O.name+'/'+im['file']+'"></figure>'
section+='<p>下一步：'+html.escape(next_text)+'</p></section>'
log=L/'index.html';s=log.read_text(encoding='utf-8');assert 'id="fixed-direction"' not in s
pos=s.index('</style>')+len('</style>');s=s[:pos]+section+s[pos:];log.write_text(s,encoding='utf-8')
save(REC/'fixed-direction-completion.json',dict(updated=stamp,state='evidence_recorded_publication_pending',
 daily_log=str(log),actual_native_image_review=True,training_started=False,quality_accepted=False,
 diagnostic_DOM_verified=False,browser_pixel_review=False,PPT_updated=False,previous_goal_turn='progress'))
print(json.dumps(dict(conclusion=str(REC/'fixed-direction-conclusion.json'),daily_log=str(log)),ensure_ascii=False))
