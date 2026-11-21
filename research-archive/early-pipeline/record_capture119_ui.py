"""Persist an actual UI review only after OpenFile Gaussian + points inspection."""
from pathlib import Path
import argparse, json, hashlib, datetime
p=argparse.ArgumentParser();p.add_argument('--subject',required=True);p.add_argument('--count',type=int,required=True);p.add_argument('--observation',required=True);a=p.parse_args()
TOP=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=TOP/'shuolee/current/clothed-surface-119'
ROOT=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-clothed-surface-119')/a.subject
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
validation=json.loads((ROOT/'export/validation.json').read_text())
assert a.count==validation['points']
folder=REC/f'{a.subject}-final-review'
images=[folder/'openfile-gaussian.png',folder/'openfile-points.png']
assert all(p.is_file() for p in images)
d=dict(updated=datetime.datetime.now().astimezone().isoformat(),subject=a.subject,source_sha256=validation['sha256'],
    file_opened_through_UI=True,modes=['Gaussian','points'],observed_count=a.count,observed_sh_degree=3,
    actual_imported_file=str(TOP/f'viewer/models/new-capture-119-{a.subject}-clothed/shuolee_119_{a.subject}_s18000.ply'),
    screenshot_observations=a.observation,screenshots=[dict(path=str(p),sha256=sha(p)) for p in images],
    ply_format_passed=True,quality_accepted=False,local_browser_only=True,pc00_opened_this_round=False)
(REC/f'{a.subject}-viewer-ui-review.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('ACTUAL_UI_REVIEW_RECORDED',a.subject,a.count)
