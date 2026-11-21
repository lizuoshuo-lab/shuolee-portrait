from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
stamp=datetime.now(timezone.utc).isoformat()
folder=REC/'winterhat-material-samples-review'
material=dict(updated=stamp,actual_visual_review=True,index_sha256=sha(folder/'index.json'),
    reviewed_tracks=read(folder/'plan.json')['selected'],quality_accepted=False,physical_identity_passed=False,
    summary='按轨迹ID等距取6条候选，未按误差筛选。已实际看完每条的早/中/晚三张128×128原生纹理裁片。第3、4条明确是帽子图案和毛边，不能当人脸材料真值；第6条主要在下颌/颈部区域。第1、2、5条覆盖真实面部，但眨眼、嘴角形状及视角发生变化，目前仅确认区域连续，不能认证同一物理材料点的亚像素定位。未用这些审阅结果改相机、删点或改变训练。',
    serialization_repair='第一次图片生成成功，索引写入因NumPy int32无法JSON序列化失败；仅将整数转为Python int，重建索引时核验6张图片像素完全一致，无图像/匹配/求解重跑。')
save(folder/'actual-review.json',material)
folder=REC/'winterhat-train-006000-review';idx=read(folder/'index.json')
assert len([x for x in idx['images'] if x['region']=='body'])==6
review=dict(updated=stamp,actual_visual_review=True,index_sha256=sha(folder/'index.json'),steps=6000,
    summary='六个同相机全身与实际中心已审阅，并放大检查frame_45/67/112面部。比1000步，衣服/裙褶与正面鼻嘴更清楚，包的颜色更完整，中心继续分布在衣服和裙面。frame_112眼鼻仍明显复影，frame_45侧脸前方仍有拖影，肩袖/裙边有长条溅射，靴底/脚尖仍有双轮廓。质量未通过。继续已授权同一有限18k运行，未改变相机或优化目标。',
    actually_reviewed_body_frames=[x['frame'] for x in idx['images'] if x['region']=='body'],
    detailed_face_frames=['frame_000045','frame_000067','frame_000112'],
    rear_hat_not_counted_as_face=True,quality_accepted=False,physical_calibration_accepted=False,
    new_training_run_started=False,training_parameters_changed=False)
save(folder/'actual-review.json',review)
folder=REC/'winterhat-regional-support-review';folder.mkdir(exist_ok=True)
audit=read(REC/'winterhat-regional-support/result.json')
save(folder/'index.json',dict(subject='winterhat',images=[],kind='frozen track regional numerical audit'))
summary='冻结留出观测只读统计：mask外候选28419次，中位0.665px；人物内部1854次，中位2.269px。自动面部ROI只有38条轨迹、142次观察，覆盖71/112个视角，109次误差>=1px，最大15.136px。全部留出还有48条非正深度轨迹，所有异常保留。背景候选也包含高误差，不能当已标定静止真值。局部裁片已发现帽子ROI混入，因此数字不能替代真实人脸物理点验收，也不能唯一归因于畸变、外参或微动。未进行任何新的匹配/BA/参数更新。'
save(folder/'actual-review.json',dict(updated=stamp,summary=summary,actual_visual_review=True,
    numerical_audit=str(REC/'winterhat-regional-support/result.json'),physical_material_truth=False,quality_accepted=False))
for dest in [T/'viewer/local-surface-123/winterhat-regional-support',Path('D:/00/工作/查看器/local-surface-123/winterhat-regional-support'),Path('D:/00/工作日志/2026-10-28/人物自由表面训练123/winterhat-regional-support')]:
    dest.mkdir(parents=True,exist_ok=True)
    for n in ['plan.json','result.json']:shutil.copy2(REC/'winterhat-regional-support'/n,dest/n)
ledger=read(REC/'evidence-ledger.json');kind='winterhat_6000_and_frozen_material_audit'
if not any(x.get('kind')==kind for x in ledger['entries']):
    ledger['entries'].append(dict(kind=kind,date=stamp,summary=review['summary'],regional_audit=summary,
        actual_material_sample_review=str(REC/'winterhat-material-samples-review/actual-review.json'),quality_accepted=False))
ledger['updated']=stamp;save(REC/'evidence-ledger.json',ledger)
print(json.dumps(dict(review=review['summary'],regional=summary),ensure_ascii=False))
