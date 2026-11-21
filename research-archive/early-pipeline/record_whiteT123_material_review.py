"""Record the six actually viewed, preselected native material crops. No fitting."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil

T = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC = T / 'shuolee/current/local-clothed-surface-123'
folder = REC / 'whiteT-material-samples-review'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p, d): p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
idx = read(folder / 'index.json')
assert len(idx['images']) == 6
for item in idx['images']:
    assert sha(folder / item['file']) == item['sha256']
summary = ('按轨迹ID等距预选6条候选，已实际看完每条早/中/晚三张128×128原生裁片。'
           '第1、2、4条覆盖眼鼻、嘴胡须或面部轮廓，眨眼、视角及背景遮挡变化可见，尚不能认证同一物理材料点的亚像素身份。'
           '第3条为背景横幅图案；第5条为建筑/网格与脸部遮挡边缘；第6条为树叶，后一个观察有头部遮挡。'
           '这些候选曾进入自动面部ROI，不能把全部20条ROI轨迹直接算作真实脸部材料轨迹。'
           '所有冻结观察及拟合/留出划分保留，未按误差或此次审阅删点，未修改训练相机或正在运行的代码。')
review = dict(updated=datetime.now(timezone.utc).isoformat(), actual_visual_review=True,
    index_sha256=sha(folder / 'index.json'), reviewed_tracks=[x['frame'] for x in idx['images']],
    quality_accepted=False, physical_identity_passed=False, summary=summary,
    region_reviews=[
        dict(sample=1, finding='真实面部区域，闭眼/睁眼变化；材料身份与亚像素定位未认证'),
        dict(sample=2, finding='嘴、胡须与面部轮廓候选；背景与可见轮廓变化'),
        dict(sample=3, finding='背景横幅重复图案，不能算人脸材料'),
        dict(sample=4, finding='眼鼻与面部轮廓候选，未认证亚像素材料身份'),
        dict(sample=5, finding='建筑/网格与脸部遮挡边缘混入，不能直接当人脸材料'),
        dict(sample=6, finding='背景树叶，末视角有人头遮挡，不能直接当人脸材料')])
save(folder / 'actual-review.json', review)
regional = read(REC / 'whiteT-regional-support/result.json')
stats = regional['groups']['RTMW_face_ROI_only']
stat_summary = ('冻结留出候选的自动面部ROI含54次观察、20条轨迹，只覆盖41/93个视角；'
                '47次误差≥1px，中位3.436px，最大28.644px。'
                '整图中位0.510px不能代表脸部精度；ROI又混入背景，数字本身不是真实人脸标定验收。'
                '全部留出中40条非正深度轨迹及大误差原样保留。相机误差、错误对应与人物微动的贡献仍未唯一分离。')
overview = REC / 'whiteT-regional-support-review'
overview.mkdir(exist_ok=True)
save(overview / 'index.json', dict(subject='whiteT', images=[], kind='read_only_frozen_regional_audit'))
save(overview / 'actual-review.json', dict(updated=review['updated'], summary=stat_summary,
    source_result_sha256=sha(REC / 'whiteT-regional-support/result.json'),
    quality_accepted=False, formal_calibration_accepted=False, face_roi_candidates=stats))
ledger_path = REC / 'evidence-ledger.json'
ledger = read(ledger_path)
kind = 'whiteT_frozen_material_regional_audit'
if not any(e.get('kind') == kind for e in ledger['entries']):
    ledger['entries'].append(dict(kind=kind, date=review['updated'], summary=stat_summary,
        actual_material_review=str(folder / 'actual-review.json'), material_summary=summary,
        quality_accepted=False, camera_or_observation_mutated=False))
    save(ledger_path, ledger)
log = Path('D:/00/工作日志/2026-10-28/人物自由表面训练123')
out = log / 'whiteT-regional-support'
out.mkdir(exist_ok=True)
for name in ['plan.json', 'result.json']:
    shutil.copy2(REC / 'whiteT-regional-support' / name, out / name)
print(json.dumps(dict(actual_review=str(folder / 'actual-review.json'), summary=stat_summary), ensure_ascii=False))
