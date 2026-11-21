from pathlib import Path
from datetime import datetime,timezone
import json,shutil,hashlib
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001');R=T/'shuolee/current/robust-portrait-122';REC=T/'shuolee/current/local-clothed-surface-123';O=R/'video10-static-material';B=R/'video10-temporal-bridge';C=H.parent/'shuolee-coordination';L=Path('D:/00/工作日志/2026-10-29/人物对齐核验123')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
stamp=datetime.now(timezone.utc).isoformat();receipt=read(H/'video10-material123-deck-receipt.json');assert receipt['canonical_replaced'] and receipt['actual_last_page_visual_review']
assert read(B/'progress.json')['state']=='native24_ready_region_review_pending'
for name,needle in [('browser-report-ax.txt','视频10：三视角对应中的物理错接'),('browser-gallery-ax.txt','鞋架被误配'),('browser-viewer-ax.txt','smplx123-video10-material')]:
 txt=(O/name).read_text(encoding='utf-8')
 if name=='browser-gallery-ax.txt':assert '鞋架跳到了冰箱便签' in txt
 else:assert needle in txt
view=(O/'browser-viewer-ax.txt').read_text(encoding='utf-8');assert '309,523' in view and '已加载 93 个相机' in view
save(O/'viewer-ui-review.json',dict(updated=stamp,report_and_gallery_DOM_verified=True,latestDiagnostic_visible=True,historical_model='室外白T恤人物 · 本地123自由表面 · 18000步',point_count_UI=309523,camera_count_UI=93,model_not_replaced=True,actual_new_model_pixel_review=False,screenshot_limitation='Previously documented zero-width IAB screenshot unavailable; did not repeatedly retry. Actual saved scientific crops and slide52 pixels reviewed using view_image.',new_model_export=False,company00_verified=False,SIBR_or_externalSpark_verified=False,tab15_preserved=True))
p=read(O/'publication.json');p.update(updated=stamp,browser_DOM_verified=True,actual_saved_images_reviewed=True,browser_pixel_UI_verified=False,PPT_pending=False,PPT_updated=True,PPT_canonical=receipt['canonical_path'],PPT_sha256=receipt['final_sha256'],PPT_changed_slides=[52],PPT_first51_render_and_other_package_members_identical=True);save(O/'publication.json',p)
next_text='视频10一次时间桥接诊断已登记：24个新中间帧已原生提取，和9个冻结锚点组成33帧，最大间隔27帧。下一步先审阅24帧及排除区域，再只对24个新帧执行一次同参数特征提取/近邻匹配，复用旧9帧和36对缓存；原族划分不变，混组连接隔离。未提取新特征、未求新相机、未拟合或训练。'
for root in [R,REC]:
 q=read(root/'queue-state.json');q.update(updated=stamp,current_stage='video10_temporal_bridge_native24_ready',phase='native_region_review_pending',next=next_text,next_action=next_text,next_plan=str(B/'plan.json'),running=None)
 q['last_goal_turn'].update(next=next_text,temporal_bridge_inputs=str(B/'native-inputs.json'));save(root/'queue-state.json',q)
q=read(R/'protocol.json');q.update(updated=stamp,current_stage='video10_temporal_bridge_native24_ready',next=next_text);save(R/'protocol.json',q)
n=read(REC/'next-action.json');n.update(updated=stamp,next=next_text,next_action=next_text,next_plan=str(B/'plan.json'));save(REC/'next-action.json',n)
c=read(C/'local-research-status.json');c.update(updated_at=stamp,source_queue_updated=stamp,as_of_date='2026-10-29',next=next_text,local_GPU_task_active=False,current_video10_temporal_bridge=dict(plan=str(B/'plan.json'),state='native24_ready_region_review_pending',decoded_new_frames=24,all_diagnostic_frames=33,features_extracted=False,cameras_estimated=False,training_started=False))
c['historical_training_started']=bool(c.get('historical_training_started',False) or c.get('training_started',False));c['training_started']=False;c['training_running']=False;save(C/'local-research-status.json',c)
for dest in [L/'视频10材料对应核验',T/'viewer/diagnostics/video10-static-material',Path('D:/00/工作/查看器/diagnostics/video10-static-material')]:
 for name in ['publication.json','viewer-ui-review.json','browser-report-ax.txt','browser-gallery-ax.txt','browser-viewer-ax.txt']:shutil.copy2(O/name,dest/name)
save(O/'completion.json',dict(updated=stamp,state='complete_reviewed_published',result=str(O/'result-reviewed.json'),goal_complete=False,goal_turn_classification='progress',next=next_text,next_plan=str(B/'plan.json'),new_training=False))
print(json.dumps(dict(published=p['report'],PPT=p['PPT_canonical'],next=next_text),ensure_ascii=False))
