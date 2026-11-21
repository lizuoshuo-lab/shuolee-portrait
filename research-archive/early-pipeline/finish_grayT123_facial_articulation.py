"""Record an actually reviewed negative control and a frozen cross-run audit."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
from PIL import Image,ImageDraw
import numpy as np
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';R=T/'shuolee/current/robust-portrait-122'
O=REC/'grayT-facial-articulation-review';C=H.parent/'shuolee-coordination'
LOG=Path('D:/00/工作日志/2026-10-28/人物自由表面训练123')
A=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-local-surface-123/grayT')
B=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/smpl-x-alignment-20261028/grayT-7b1c0fd30822c3de-r2')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
stamp=datetime.now(timezone.utc).isoformat();r=read(O/'result.json');x=read(O/'cross-run-result.json')
assert r['controls']['all_passed'] and r['sources_unchanged'] and x['sources_unchanged']
review=dict(updated=stamp,actual_visual_review=True,frames=[40,48,56,64,72,80],
    reviewed_images=[dict(path=str(O/'all6_native_articulation_control.png'),sha256=sha(O/'all6_native_articulation_control.png'))],
    summary='已实际审阅6帧原生320px四列对照：原图、检测点青色、原SMPL-X紫色、共享表情/下颌候选橙色。40/48帧嘴部有局部形变；56帧后鼻嘴仍明显偏向脸颊；72/80侧脸的整组投影仍落在耳前/脸颊。增大表情自由度没有把不同视角投回同一鼻嘴。',
    raw_crops_have_no_burned_text=True,image_shape_preserved=True,physical_quality_accepted=False)
save(O/'actual-review.json',review)
idx=read(O/'index.json')
for i in idx['images']:i['region']='原图／观测青点／原模型紫点／共享表情与下颌橙点，四列原生320px裁片'
save(O/'index.json',idx)
summary=('已确认拟合源码的表达限制：29个鼻嘴点参与损失，但SMPL-X的10维表情和3维下颌未进入优化器，保持默认零值。'
    '固定95帧相机、身体、体型、根姿态、观测和权重，只增加一组共享表情/下颌，完成一次CPU对照；回放投影差0.000339px、梯度差9.61e-11。'
    '加权损失4.775→4.646（降2.70%），2732个有效语义残差中位43.99→48.69px；9个表情系数接近边界。'
    '6帧原生图实际审阅仍有侧脸鼻嘴落在脸颊，因此不能将开启表情当成对齐修复，也不能把微小损失下降等同质量提高。'
    '同时只读核对00对话独立完成的同视频95帧：换回原生坐标后K一致至2.3e-13，相机相对旋转却中位差1.54°、最大6.89°。'
    '00语义面部残差较低（原生17.71px），但相同皮肤标记35留出残差18.83→19.15px，两组35/35均≥1px。'
    '分辨率、特征集、遮罩和拟合代码同时变化，这不是单因素归因；较低的语义点残差仍没有证明物理对齐。研究侧未改变历史相机、模型或开启新高斯训练。')
next_text=('当前证据支持“相机—静态人物—语义观测尚不一致”，不是遗漏jaw/expression单独造成的。'
    '保留本次负对照，不能放大表情边界继续吸收误差。接下来只核对独立复现的背景支持、相机条件与同一物理材料轨迹；'
    '先区分相机差异与头部相对运动，不将更低拟合损失作为训练准入。已有近景控制主要共面，头部高度非共面场景支持仍是具体证据缺口。')
con=dict(updated=stamp,summary=summary,next=next_text,controls=r['controls'],articulation_metrics=r['metrics'],
    objective_reduction_fraction=r['objective_reduction_fraction'],near_bound_parameter_indices=r['near_bound_parameter_indices'],
    cross_run=dict(K_native_max_abs=x['K_native_max_abs'],relative_rotation_difference_deg=x['relative_rotation_difference_deg'],
        semantic_observation_difference_native_px=x['semantic_observation_difference_native_px'],company00_semantic_residual_native_px=x['company00_face_semantic_residual_native_px'],
        original_skin_held=x['baseline_M0']['held'],company00_skin_held=x['company00_M0']['held'],held_above_1px=[35,35],single_factor=False),
    actual_native_image_review=True,production_camera_or_geometry_changed=False,training_started=False,quality_accepted=False,
    sources=[dict(path=str(p),sha256=sha(p)) for p in [O/'plan.json',O/'controls.json',O/'result.json',O/'actual-review.json',O/'cross-run-plan.json',O/'cross-run-result.json']])
save(REC/'facial-articulation-conclusion.json',con);save(LOG/'facial-articulation-conclusion.json',con)
key='grayT123-static-facial-articulation-and-cross-run-audit'
for base in [REC,R]:
    p=base/'evidence-ledger.json';d=read(p)
    if base==REC:
        assert not any(v.get('id')==key for v in d['entries'])
        d['entries'].append(dict(id=key,updated=stamp,summary=summary,evidence=str(REC/'facial-articulation-conclusion.json'),quality_accepted=False))
    else:d['confirmed'].append(summary);d['artifacts'].append(str(REC/'facial-articulation-conclusion.json'))
    d['updated']=stamp;save(p,d)
q=read(REC/'queue-state.json');q.update(updated=stamp,current_stage='facial-articulation-control-completed',reason=summary,
    next_action=next_text,next_plan=None,latest_diagnosis=str(REC/'facial-articulation-conclusion.json'),training_started=False,
    training_running=False,training_allowed=False,local_GPU_task_active=False,worker_pid=None,worker_command=None)
save(REC/'queue-state.json',q)
n=read(REC/'next-action.json');n.update(updated=stamp,next=next_text,next_plan=None,latest_diagnosis=str(REC/'facial-articulation-conclusion.json'),actual_UI_verified=False,diagnostic_DOM_verified=False);save(REC/'next-action.json',n)
c=read(C/'local-research-status.json');c.update(updated_at=stamp,current_local_facial_articulation_control=con,next=next_text,local_GPU_task_active=False);save(C/'local-research-status.json',c)
with (T/'shuolee/long-term-plan.md').open('a',encoding='utf-8') as f:f.write('\n\n## 2026-10-28 共享下颌/表情与独立复现核对\n\n'+summary+'\n\n'+next_text+'\n\n证据：'+str(REC/'facial-articulation-conclusion.json')+'\n')
save(REC/'facial-articulation-completion.json',dict(updated=stamp,state='control_reviewed_publication_pending',actual_native_image_review=True,
    browser_pixel_review=False,diagnostic_DOM_verified=False,PPT_updated=False,goal_active=True,quality_accepted=False,company00_handoff_sent=False))
for f in ['queue-state.json','next-action.json','facial-articulation-completion.json']:shutil.copy2(REC/f,LOG/f)
print(summary)
