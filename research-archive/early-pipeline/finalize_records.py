from pathlib import Path
import json,hashlib
from datetime import datetime
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
W=Path('C:/Users/Administrator/Documents/New project/shuolee-slide-update')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
now=datetime.now().astimezone().isoformat(timespec='seconds')
report=B/'reports/02_有限视角重建优化路线.pptx'
digest=hashlib.sha256(report.read_bytes()).hexdigest()
assert digest=='b11a18d22ecf5c393b06ee79f62c1ccb8d039fa47b4f940c6ff50f774ff55269'
check=read(W/'unchanged-slide-check.json');assert not check['unexpectedContentOrLayoutDifferences']
write(W/'visual-review.json',{'created':now,'changed_slides_viewed':[2,33,34,41,51,52,53],'all_slides_rendered':53,'whole_deck_montage_scanned':True,'unchanged_content_layout_match':True,'overflow_repair':'Imported slide2 and34 manual linebreaks restored before final export.','source_slide_count':51,'final_slide_count':53,'final_sha256':digest,'native_powerpoint_opened':False})
m=read(B/'reports/report-manifest.json');m['updated']=now
for r in m['reports']:
 if r['key']=='reconstruction-optimization':r.update(slides=53,sha256=digest,bytes=report.stat().st_size,local_validation=str(W/'validation.json'),local_visual_review=str(W/'visual-review.json'))
m['last_local_revision']={'sha256':digest,'bytes':report.stat().st_size,'slides':53,'remote_delivered':False,'remote_status':'local_first_dynamic_and_native_surfel_candidate_review','updated':now}
write(B/'reports/report-manifest.json',m)
s=read(B/'work-status.json')
if s.get('current_experiment',{}).get('name')=='VHAP_multiview_initialization_and_prior_study':s['prior_head_alignment_reference']=s['current_experiment']
s['current_experiment']={'name':'Dynamic_all59_body_with_static_background_and_native_2DGS_control','status':'reviewed_partial_improvement_not_accepted','accepted':False,'main_training_complete':True,'main_steps':18000,'points':191595,'pilot_steps_per_arm':1180,'pilot_updates':2360,'evidence':str(B/'current/novel-view-face/head-structure/flame-photometric-registration/whole-body-gaussianavatar/all-views-background/surface-gaussian-pilot/root-review.json'),'new_best_model':False,'remote_deployed':False,'eyewear_specialization':False}
s['latest_local_report_revision']={'path':str(report),'sha256':digest,'slides':53,'remote_delivered':False}
s['updated']=now
write(B/'work-status.json',s)
page=B.parent/'viewer/body-dynamic.html'
v=page.read_text(encoding='utf-8')
v=v.replace('左边是原视频，右边是对应时刻的人体与真实背景联合渲染。','左边是原视频，中间是训练前，右边是本轮对应时刻的人体与真实背景联合渲染。')
page.write_text(v,encoding='utf-8')
legacy=B.parent/'viewer/body-long.html'
v=legacy.read_text(encoding='utf-8-sig')
v=v.replace('下一轮接入全部 59 个拍摄视角与真实背景，重新按 18,000 步执行。','当前已沿用动态人物结构，将全部 59 个拍摄视角与真实背景接入，并完成新的 18,000 步训练，效果见上方当前页面入口。')
legacy.write_text(v,encoding='utf-8')
print(json.dumps({'report_sha256':digest,'slides':53,'remote_delivered':False,'training_running':s['local_training_running']},ensure_ascii=False))
