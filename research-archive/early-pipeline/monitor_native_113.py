"""Bounded progress-file sync, no training or model promotion; exits with run."""
from pathlib import Path
import json,time,shutil,hashlib
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee');R=B/'current/person-static-native-align-fresh-113';D=B.parent/'viewer/body-native113'
D.mkdir(exist_ok=True);read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
last=None;image_hash=None;start=time.time();reported=-1
while time.time()-start<4*3600:
    p=read(R/'gaussian-run/progress.json');failure=R/'gaussian-run/failure.json';result=R/'gaussian-run/result.json'
    completed=result.exists();failed=failure.exists()
    review=read(R/'gaussian-run/review-metrics.json');jpg=R/'gaussian-run/review-current.jpg';h=hashlib.sha256(jpg.read_bytes()).hexdigest()
    if h!=image_hash:
        temp=D/'review.tmp.jpg';shutil.copyfile(jpg,temp);temp.replace(D/'review.jpg');image_hash=h
    state='训练完成，等待新视角与导出验收' if completed else '训练已停止，需要检查' if failed else '正在训练'
    record={**p,'state_label':state,'review_step':review['step'],'updated':time.time(),'quality_accepted':False,'actual_published_model':'098'}
    tmp=D/'progress.json.tmp';tmp.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8');tmp.replace(D/'progress.json')
    if p['step']//1000!=reported:
        reported=p['step']//1000;print(json.dumps(dict(step=p['step'],review=review['step'],state=state),ensure_ascii=False),flush=True)
    if completed or failed:break
    time.sleep(10)
