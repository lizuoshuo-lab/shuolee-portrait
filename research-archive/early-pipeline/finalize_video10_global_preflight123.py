"""Close input preflight and register a bounded synthetic global initializer prototype."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,html,shutil
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';U=R/'video10-remaining-material-review';REC=R.parent/'local-clothed-surface-123'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat(); d=read(U/'decision.json'); pf=read(U/'global-input-preflight/result.json')
minimal=read(U/'global-input-preflight/essential-minimal-graph.json'); assert minimal['connected']
new=R/'video10-global-initializer';new.mkdir(exist_ok=True);assert not (new/'plan.json').exists()
inputs=[U/'global-input-preflight/fit-observations.json',U/'global-input-preflight/heldout-observations.json',U/'global-input-preflight/conditional-camera.json',U/'global-input-preflight/result.json',U/'global-input-preflight/essential-minimal-graph.json',U/'generic-observability/synthetic-input-and-jacobian.npz']
plan=dict(registered=stamp,state='registered_not_executed',hypothesis='Separate global initialization from incremental PnP. A rotation-averaging and global bearing-position prototype can exploit shared tracks jointly, but must first pass known-geometry controls.',
 source_sha256={str(p):sha(p) for p in inputs},
 inputs='Use only saved synthetic geometry plus frozen reviewed FIT visibility. Real pixel arrays are packaged but are not to be used in this synthetic run. HELD never used for optimization.',
 implementation='Calibrated known relative rotations on the fixed>=5-correspondence applicability graph -> connection-Laplacian rotation averaging withSO3 projection and exact first-camera gauge -> joint world bearing constraints on all points and centers. No sequential PnP. No old real poses, fitted bodies, camera interpolation or intrinsic variables.',
 controls=[
   'Exact known relative rotations on the33-view synthetic graph: recover each rotation up to the removed global gauge. Full rank does not replace this implemented control.',
   'Using recovered rotations and exact synthetic bearing observations, solve centers/points jointly; fix first center and scale/sign. Compare all centers/points to known geometry after one documented similarity gauge; positive depth required.',
   'Pure rotation/zero-parallax synthetic negative: position ambiguity must be detected and rejected, not converted to finite depth.',
   'Predeclared disconnected two-block synthetic rotation graph: refuse a full-camera result. Do not bridge it with arbitrary poses.'
 ],
 accuracy=dict(exact_rotation_angle_max_rad=1e-7,exact_position_relative_max=1e-6,exact_reprojection_max_px=1e-5),
 budget=dict(CPU_seconds=120,synthetic_control_cases=4,configuration_searches=0,real_camera_estimates=0,GPU_tasks=0,downloads=0,training_steps=0),
 boundaries='Known relative rotations are a synthetic component control, not solved real poses. Actual essential-matrix branches, robustness, real parallax and independent background/face projection require subsequent separately registered evaluation. The>=5 graph tests mathematical applicability only, never lower old robustness/admission gates.',
 stop='Run each fixed control once. Preserve failure. Repair a demonstrated implementation bug only; no threshold sweeps, random restarts, real BA or model publication from these controls.')
save(new/'plan.json',plan)
additional='原生输入封装核对通过：162组FIT共563次观测、45组HELD共161次观测，与原浮点关键点逐项一致。4000×3000第一K经条件缩放/居中裁剪复现既有横屏K。源码确认旧global_fixedk121仍是双视图起步、逐帧PnP，再对已注册视角联合BA，尚非全局初始化。'
next_text='实现并执行已登记的4项合成全局初始化控制：旋转平均、共享轨迹的全局位置/结构恢复、纯旋转退化拒绝和断图拒绝。固定K，不使用实拍像素估计相机；通过后再单独登记实拍33帧候选与独立保护。'
d.update(updated=stamp,summary=d['summary']+' '+additional,global_preflight=pf,next=next_text,next_plan=str(new/'plan.json'))
save(U/'decision.json',d)
save(U/'global-input-preflight/decision.json',dict(updated=stamp,preflight_complete=True,summary=additional,
 mathematical_five_point_graph_connected=True,graph_is_not_robust_pose_acceptance=True,source=str(U/'global-input-preflight/result.json'),next=next_text,next_plan=str(new/'plan.json'),training_started=False))
p=read(U/'progress.json');p.update(updated=stamp,state='review_and_global_input_preflight_complete',next_plan=str(new/'plan.json'));save(U/'progress.json',p)
# Update the same report and the same diagnostic entry, keeping all model fields unchanged.
article=(U/'viewer-article.html').read_text(encoding='utf-8');old_summary=article[article.index('<p>')+3:article.index('</p>')]
oldnext=read(R/'queue-state.json')['next_action']
newarticle=article.replace(old_summary,html.escape(d['summary']),1).replace(html.escape(oldnext),html.escape(next_text),1)
(U/'viewer-article.html').write_text(newarticle,encoding='utf-8')
addhtml='<h2>全局初始化接口核验</h2><p>'+html.escape(additional)+'</p><p>逐项核对的724次原生观测保持原划分，不存在跨FIT/HELD共用同一特征ID。相机模型仍是已经去畸变视频的条件针孔K，D=0，不再次套用原始FISHEYE_8。精确导出内参尚未独立认证。</p><p>本地已核验PoseLib 2.0.5和SciPy；PyCOLMAP 3.10.0只暴露增量映射。已检查PATH中没有GLOMAP入口，未进行全盘软件扫描或安装。按标定两视图最小5对应形成的图连通，仅说明算法的数学适用性，不降低鲁棒性和最终精度要求。</p>'
s=(U/'index.html').read_text(encoding='utf-8');s=s.replace('<h2>下一步</h2>',addhtml+'<h2>下一步</h2>',1).replace(html.escape(oldnext),html.escape(next_text),1)
(U/'index.html').write_text(s,encoding='utf-8')
for folder,name in [(T/'viewer','body-smplx.html'),(Path('D:/00/工作/查看器'),'progress.html')]:
 p=folder/name;s=p.read_text(encoding='utf-8');assert article in s;p.write_text(s.replace(article,newarticle,1),encoding='utf-8')
 p=folder/'latest-result.json';x=read(p);model={k:v for k,v in x.items() if k!='latestDiagnostic'}
 x['latestDiagnostic'].update(updated=stamp,summary=d['summary']);save(p,x);assert model=={k:v for k,v in read(p).items() if k!='latestDiagnostic'}
for root in [R,REC]:
 p=root/'queue-state.json';x=read(p);x.update(updated=stamp,current_stage='video10_global_input_preflight_complete',phase='global_initializer_synthetic_controls',next=next_text,next_action=next_text,next_plan=str(new/'plan.json'),reason=d['summary'])
 x['last_goal_turn'].update(updated=stamp,summary=d['summary'],next=next_text);save(p,x)
p=R/'protocol.json';x=read(p);x.update(updated=stamp,current_stage='video10_global_input_preflight_complete',next=next_text);save(p,x)
p=REC/'next-action.json';x=read(p);x.update(updated=stamp,next=next_text,next_action=next_text,next_plan=str(new/'plan.json'));save(p,x)
p=R/'evidence-ledger.json';x=read(p);x['confirmed'].append(additional);x['artifacts'] += [str(U/'global-input-preflight/result.json'),str(new/'plan.json')];x['updated']=stamp;save(p,x)
p=REC/'evidence-ledger.json';x=read(p);x['entries'].append(dict(id='video10_global_input_preflight',updated=stamp,summary=additional,evidence=str(U/'global-input-preflight/result.json'),quality_accepted=False));save(p,x)
p=H.parent/'shuolee-coordination/local-research-status.json';x=read(p);x.update(updated_at=stamp,source_queue_updated=stamp,next=next_text)
x['current_video10_remaining_material'].update(summary=d['summary'],global_preflight_complete=True,next_plan=str(new/'plan.json'));save(p,x)
p=T/'shuolee/long-term-plan.md';s=p.read_text(encoding='utf-8');p.write_text(s+'\n\n'+additional+'\n\n'+next_text+'\n',encoding='utf-8')
p=Path('D:/00/工作日志/2026-10-29/人物对齐核验123/index.html');s=p.read_text(encoding='utf-8');s=s.replace(html.escape(oldnext),html.escape(next_text),1).replace('<section id="video10-remaining">','<section id="video10-remaining"><p>'+html.escape(additional)+'</p>',1);p.write_text(s,encoding='utf-8')
for dest in [T/'viewer/diagnostics/video10-remaining-material-review',Path('D:/00/工作/查看器/diagnostics/video10-remaining-material-review'),Path('D:/00/工作日志/2026-10-29/人物对齐核验123/视频10材料与相机初始化核验')]:
 for f in ['index.html','decision.json']:shutil.copy2(U/f,dest/f)
 for f in ['result.json','conditional-camera.json','decision.json']:shutil.copy2(U/'global-input-preflight'/f,dest/('global-preflight-'+f))
print(json.dumps(dict(summary=additional,next_plan=str(new/'plan.json')),ensure_ascii=False))
