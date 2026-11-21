"""Record the actual six-camera and 36-angle review performed on 2026-10-28."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123'
ROOT=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-local-surface-123/winterhat')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)
now=datetime.now(timezone.utc).isoformat()
digest=sha(ROOT/'export/point_cloud.ply');v=read(ROOT/'export/validation.json');assert digest==v['sha256']
summary='戴帽蓝衣人物已从头完成18000步，294054个自由高斯分布在帽子、蓝色外套、黄色裙面、红包及靴子周围。六个同相机原图/渲染/实际中心已看完，面部眼鼻嘴仍有复影，肩袖和裙边存在细长拖尾，靴底与脚尖双轮廓仍明显。连续36个环绕视角已逐张审阅，未拍到的背面出现空洞与破碎条纹。衣物表面分布成立，重影修复和物理相机标定未通过；本轮发布为可检查的研究结果。'
frames=[f'frame_{n:06d}' for n in [1,23,45,67,89,112]]
review=REC/'winterhat-train-018000-review'
save(review/'actual-review.json',dict(updated=now,actual_visual_review=True,index_sha256=sha(review/'index.json'),steps=18000,summary=summary,actually_reviewed_body_frames=frames,actually_reviewed_face_regions=frames,face_region_warning='frame_000001 detector face crop is hat only; not valid facial support',quality_accepted=False,physical_calibration_accepted=False,training_parameters_changed=False,source_sha256=digest,export_validation=v))
novel=REC/'winterhat-novel-final-review'
save(novel/'actual-review.json',dict(updated=now,actual_visual_review=True,index_sha256=sha(novel/'index.json'),summary='已看完novel_000至035共36幅，依10度间隔顺序。可见正面与侧面仍有面部拖尾和脚部复影；背面大量空洞、破碎和透明感，不能宣称可靠恢复。没有未知视角实拍真值。',continuous_novel_views_reviewed=36,sheets=[dict(path=str(p),sha256=sha(p)) for p in sorted(novel.glob('views_*.png'))],quality_accepted=False))
quality=dict(updated=now,subject='winterhat',source_sha256=digest,steps=18000,points=v['points'],camera_count=112,initial_points=200000,actual_same_camera_review=True,same_camera_frames=frames,continuous_novel_views_reviewed=36,quality_accepted=False,clothed_center_distribution_observed=True,ghosting_resolved=False,subpixel_material_admission_passed=False,summary=summary,physical_calibration_accepted=False,novel_view_ground_truth_available=False,source_reviews=[str(review/'actual-review.json'),str(novel/'actual-review.json')],format_passed=v['format_passed'],local_viewer_UI='pending',official_SIBR_GUI_verified=False,independent_Spark_GUI_verified=False,company00_verified=False)
save(REC/'winterhat-quality-review.json',quality)
ledger=read(REC/'evidence-ledger.json')
if not any(e.get('kind')=='winterhat_final18k_actual_quality_review' for e in ledger['entries']):
    ledger['entries'].append(dict(kind='winterhat_final18k_actual_quality_review',date=now,summary=summary,quality_accepted=False,source_sha256=digest,quality_record=str(REC/'winterhat-quality-review.json')))
ledger['updated']=now;save(REC/'evidence-ledger.json',ledger)
print(json.dumps(quality,ensure_ascii=False))
