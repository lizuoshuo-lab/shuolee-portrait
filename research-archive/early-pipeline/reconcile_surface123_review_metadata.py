from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)
now=datetime.now(timezone.utc).isoformat()
n=R/'winterhat-novel-final-review';review=read(n/'actual-review.json')
for r in review['sheets']:assert sha(Path(r['path']))==r['sha256']
review['index_sha256']=sha(n/'index.json');review['index_metadata_correction']='Presentation index now references the six reviewed contact sheets. Original pixels and 36 source image identities unchanged.';save(n/'actual-review.json',review)
folder=R/'whiteT-input-000000-review';idx=read(folder/'index.json')
save(folder/'actual-review.json',dict(updated=now,actual_visual_review=True,index_sha256=sha(folder/'index.json'),reviewed_frames=[i['frame'] for i in idx['images'] if i['region']=='body'],summary='实际看过6个原图和mask。人物为室外白T、花纹短裤、双手扶腰男子，六视角脸部朝向变化可见。遮罩整体覆盖头发、袖子、短裤、腿和鞋；腋下/手臂间孔洞有不稳定，未拍背面不能当可靠覆盖。',quality_accepted=False))
save(R/'publication-repair.json',dict(updated=now,previous=read(R/'publication-worker.json'),cause='New novel contact-sheet index lacked frame/region/columns expected by existing gallery publisher. Publisher stopped, training process continued.',repair='Adapted only new novel index metadata, preserved sheet hashes and source hashes. Added actual delivered model status to page. No training runtime, camera or optimizer changes.',training_modified=False))
print('Metadata and actual reviews reconciled')
