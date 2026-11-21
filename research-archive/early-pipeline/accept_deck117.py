from pathlib import Path
import datetime as dt,json,hashlib,shutil,zipfile
B=Path('C:/Users/Administrator/Documents/New project/shuolee-slide-update')
ROOT=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
O=ROOT/'current/portrait-causal-diagnosis-117'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def copy(p,d):
 d.parent.mkdir(parents=True,exist_ok=True);tmp=d.with_suffix(d.suffix+'.tmp');shutil.copy2(p,tmp);tmp.replace(d);assert sha(p)==sha(d)
r=read(B/'human117-deck-receipt.json');s=Path(r['source']);f=Path(r['finalPath'])
assert sha(s)==r['source_sha256'] and sha(f)==r['final_sha256']
with zipfile.ZipFile(s) as a,zipfile.ZipFile(f) as b:
 assert set(a.namelist())==set(b.namelist())
 changed=[n for n in a.namelist() if a.read(n)!=b.read(n)]
 assert changed==read(B/'human117-exact-package.json')['changed']
 assert not any(a.read(f'ppt/slides/slide{i}.xml')!=b.read(f'ppt/slides/slide{i}.xml') for i in range(1,52))
copy(f,s);delivery=Path('D:/00/报告/02_有限视角重建优化路线.pptx');copy(f,delivery)
now=dt.datetime.now(dt.timezone.utc).isoformat()
r.update(updated=now,canonical_replaced=True,delivery=str(delivery),actual_last_slide_visual_review=True,first51_package_and_render_preserved=True)
save(B/'human117-deck-receipt.json',r)
save(O/'ppt-update.json',dict(updated=now,canonical=str(s),delivery=str(delivery),sha256=sha(f),changed_slides=[52],
 first51_package_and_render_preserved=True,actual_visual_review=True,quality_accepted=False))
pub=read(O/'publication.json');pub.update(updated=now,ui_review_pending=False,viewer_ui_review='viewer-ui-review.json',ppt_update='ppt-update.json');save(O/'publication.json',pub)
save(O/'viewer-ui-review.json',dict(updated=now,viewer='http://127.0.0.1:8890/?model=latest',report='http://127.0.0.1:8890/progress.html#diagnosis',
 actual_browser_review=True,diagnostic_autorefresh_verified=True,old_model_preserved='114b',new_gaussian=False,
 screenshots=['D:/00/工作日志/2026-10-26/查看器验证/查看器_材料对应诊断.png','D:/00/工作日志/2026-10-26/查看器验证/原图证据_材料对应说明.png'],
 remote00_verified=False,quality_accepted=False))
statepath=Path('D:/00/整理状态.json');state=read(statepath);state.update(updated=now,latest_diagnosis='117D2 complete, root cause unresolved',latest_report=str(delivery),report_last_slide=52,diagnostic_viewer_actually_checked=True);save(statepath,state)
print(json.dumps(dict(ppt=str(delivery),bytes=delivery.stat().st_size,changed_slides=[52],first51_unchanged=True),ensure_ascii=False))
