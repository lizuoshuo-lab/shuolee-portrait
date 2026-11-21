from pathlib import Path
import json,hashlib,datetime,re,shutil
import numpy as np
BASE=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
VIEW=BASE.parent/'viewer';ROOT=BASE/'current/smplx-gvhmr-allviews-fresh-059';RUN=ROOT/'gaussian-run'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
now=datetime.datetime.now().astimezone().isoformat()
result=read(RUN/'result.json');audit=read(RUN/'export-audit/audit.json')
assert result['step']==18000 and audit['state']=='passed'
model_id='whole-body-gvhmr-current';target=VIEW/'models'/model_id/'point_cloud.ply'
assert sha(target)==result['world_ply_sha256']
old=read(VIEW/'model-whole-body-smplx-fresh.json')
mesh=np.load(ROOT/'fitted.npz',allow_pickle=False)['vertices_world'][33]
meta=dict(label='GVHMR 全帧联合拟合 · 1.8万步试验',model=f'models/{model_id}/point_cloud.ply',
    status='已完成训练与PLY重载核验；侧面衣袖、裤脚仍破碎，未通过整体质量',
    sha256=sha(target),gaussianCount=result['points'],initialGaussianCount=200000,
    focus=((mesh.min(0)+mesh.max(0))/2).tolist(),initialView=33,cameraSource='capture',
    cameras=old['cameras'],filter=old['filter'],report='body-smplx.html#gvhmr-joint-current',
    qualityAccepted=False,deliveryAccepted=False,trained=True,steps=18000,fromScratch=True,
    oldCheckpointLoaded=False,all59UsedForTraining=True,independentTest=False,
    timeSnapshot=33,singlePlyIsNotDynamic=True,localAdaptation=True,
    novelViewReview='body-gvhmr-joint/novel-current.jpg',exportAudit='body-gvhmr-joint/export-audit.json')
save(VIEW/f'model-{model_id}.json',meta)
script=VIEW/'viewer.js';s=script.read_text(encoding='utf-8')
if f'"{model_id}"' not in s:s=s.replace('const BUILTINS = [',f'const BUILTINS = ["{model_id}", ',1)
script.write_text(s,encoding='utf-8')
shutil.copy2(RUN/'export-audit/audit.json',VIEW/'body-gvhmr-joint/export-audit.json')
shutil.copy2(RUN/'export-audit/world-ply-views.jpg',VIEW/'body-gvhmr-joint/world-ply-views.jpg')
page=VIEW/'body-smplx.html';html=page.read_text(encoding='utf-8')
quality='18,000步已完成。拍摄视角能恢复人物外观；固定姿态转到侧面和仰视时，衣袖、外套下摆、裤脚仍存在明显破碎，侧面脸部细节不足。当前结果未通过整体质量验收。PLY世界位置与导出前完全一致，八个相机重载比较最多差一个8位色阶；导出核验通过不代表几何正确。'
html=re.sub(r'<p id="joint-quality">.*?</p>',f'<p id="joint-quality">{quality}</p>',html,count=1,flags=re.S)
html=html.replace('<p id="joint-model-link" hidden>','<p id="joint-model-link">')
html=html.replace('没有新增高斯训练或部署00。以下均为本地有界诊断。','以下为当时未进行新高斯训练的历史诊断；最新18,000步训练结果见本页顶部。尚未部署00。')
html=html.replace('当前优先比较身体配准与三维约束；增加点数不能保证纠正错误姿态。','此前优先比较身体配准与三维约束；当前已完成新高斯训练，结果见顶部。增加点数不能保证纠正错误姿态。')
html=html.replace('当前先核查真实图像对应与人物运动，再决定如何约束关节深度。通过原图叠加、固定身体侧面和时间一致性检查后，再从新高斯与新优化器开始训练。','这些历史诊断之后，已完成全帧联合拟合及20万点、18,000步新高斯训练。当前以实际渲染继续判断侧面破碎，见本页顶部。')
page.write_text(html,encoding='utf-8')
review=dict(created=now,state='training_complete_novel_view_quality_not_accepted',training_completed=True,
    quality_accepted=False,geometry_accepted=False,export_integrity_passed=True,gaussian_steps=18000,
    initial_points=200000,final_points=result['points'],frames=59,from_scratch=True,
    reviewed=['review-current.jpg','novel-current.jpg'],findings=[quality],
    next_action='Continue one new Gaussian-training comparison targeting surface binding/fragmentation. Keep source059 as diagnostic control; no geometry-only weight sweep.',
    model_url=f'http://127.0.0.1:8870/?model={model_id}',
    ply_url=f'http://127.0.0.1:8870/models/{model_id}/point_cloud.ply?download=1',sha256=sha(target))
save(ROOT/'phase-conclusion.json',review)
state_path=BASE/'work-status.json';state=read(state_path)
state.update(updated=now,current_phase='gvhmr059_gaussians_completed_continue_fragmentation',local_training_running=False,
    current_decision='全59帧联合拟合与20万新高斯、18000步已完成；PLY重载通过，侧面衣袖裤脚仍破碎。已提供可下载点云，继续高斯层面的新对照。')
state['latest_body_handoff']=review
state['latest_local_execution'].update(state='completed_reviewed_not_accepted',running=False,quality_accepted=False)
state['latest_report']['sha256']=sha(Path(state['latest_report']['path']))
state['latest_report']['updated']=now
state['current_experiment'].update(status='059_reviewed_gaussian_candidate',current_training_running=False,
    current_candidate_role='本轮已提供点云；未知视角未通过，作为后续对照保留')
save(state_path,state)
plan=BASE/'long-term-plan.md';text=plan.read_text(encoding='utf-8')
text=text.replace('目前20万新高斯、18000步实际训练已启动','20万新高斯、18000步实际训练已完成（340.89秒，最终193446点）')
text=text.replace('当前任务是完成训练、检查六个训练视角与七个固定视角、验证原生渲染与世界PLY重载一致，再给出可查看结果。','六个训练视角与七个固定视角已检查，世界PLY重载核验通过，并提供点云下载。侧面衣袖、外套下摆和裤脚仍破碎，当前转向高斯表面绑定的有界从头对照。')
plan.write_text(text,encoding='utf-8')
print(json.dumps({'model':meta['model'],'points':result['points'],'sha256':meta['sha256'],'qualityAccepted':False},ensure_ascii=False))
