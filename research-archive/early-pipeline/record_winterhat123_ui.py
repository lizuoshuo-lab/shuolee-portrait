"""Record actual OpenFile ZIP checks performed in local viewer tab31."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
rec=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
stage=read(rec/'winterhat-staged-delivery.json');assert read(rec/'winterhat-package-camera-parity.json')['passed']
shots=[rec/'winterhat-openfile-gaussian-ui.png',rec/'winterhat-openfile-points-ui.png'];assert all(p.exists() for p in shots)
result=dict(updated=datetime.now(timezone.utc).isoformat(),file_opened_through_UI=True,input=str(stage['zip']),source_sha256=stage['source_sha256'],observed_count=294054,camera_count=112,modes=['Gaussian','points'],actual_source_frame_selected=330,screenshots=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in shots],local_viewer_version='2026.10.27.5',notes='打开文件→scene.zip，界面实际显示294054高斯、三阶SH、112配套相机SHA匹配；手动选择原帧330，分别实际看过高斯和中心点模式。衣物表面分布可见；未宣称Spark与训练原生图像一致。',quality_accepted=False,official_SIBR_GUI_verified=False,independent_Spark_GUI_verified=False,company00_verified=False)
save(rec/'winterhat-viewer-ui-review.json',result)
q=read(rec/'winterhat-quality-review.json');q['local_viewer_UI']='passed_actual_OpenFile_Gaussian_points_112cameras';q['updated']=result['updated'];save(rec/'winterhat-quality-review.json',q)
print(json.dumps(result,ensure_ascii=False))
