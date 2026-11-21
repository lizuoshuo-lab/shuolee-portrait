"""Bind manually observed UI counts and screenshots to a specific staged package."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib
p=argparse.ArgumentParser();p.add_argument('--subject',required=True)
p.add_argument('--observed-points',type=int,required=True);p.add_argument('--observed-cameras',type=int,required=True)
p.add_argument('--source-frame',type=int,required=True);p.add_argument('--notes',required=True)
p.add_argument('--observed-both-modes',action='store_true',required=True)
a=p.parse_args()
rec=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
stage=read(rec/f'{a.subject}-staged-delivery.json')
assert read(rec/f'{a.subject}-package-camera-parity.json')['passed']
assert a.observed_points==stage['points'] and a.observed_cameras==stage['camera_count']
shots=[rec/f'{a.subject}-openfile-gaussian-ui.png',rec/f'{a.subject}-openfile-points-ui.png']
assert all(p.exists() and p.stat().st_size>10000 for p in shots)
assert not (rec/f'{a.subject}-viewer-ui-review.json').exists()
result=dict(updated=datetime.now(timezone.utc).isoformat(),file_opened_through_UI=True,input=stage['zip'],
    source_sha256=stage['source_sha256'],observed_count=a.observed_points,camera_count=a.observed_cameras,
    modes=['Gaussian','points'],actual_source_frame_selected=a.source_frame,
    screenshots=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in shots],
    local_viewer_version='2026.10.27.5',notes=a.notes,quality_accepted=False,
    official_SIBR_GUI_verified=False,independent_Spark_GUI_verified=False,company00_verified=False)
save(rec/f'{a.subject}-viewer-ui-review.json',result)
q=read(rec/f'{a.subject}-quality-review.json');q['local_viewer_UI']='passed_actual_OpenFile_Gaussian_points'
q['updated']=result['updated'];save(rec/f'{a.subject}-quality-review.json',q)
print(json.dumps(result,ensure_ascii=False))
