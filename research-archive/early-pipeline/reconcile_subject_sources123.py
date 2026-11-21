from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
H=Path(__file__).parent
REC=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
rows=[]
for p in sorted(Path('D:/00/工作/数据集').glob('*.mp4')):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    rows.append(dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest()))
matches=[]
for s in read(REC/'protocol.json')['subjects']:
    same=[r for r in rows if r['sha256']==s['video_sha256']]
    matches.append(dict(subject=s['id'],old_name=s['video'],expected_sha256=s['video_sha256'],exact_matches=same))
record=dict(updated=datetime.now(timezone.utc).isoformat(),scope='Only ten MP4 files directly inside the current dataset directory. Match by full SHA256, never by filename, size or guessed identity.',files=rows,subjects=matches)
(REC/'source-name-reconciliation.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
failure=dict(updated=record['updated'],state='stopped_before_decode',exception='FileNotFoundError',missing='/mnt/d/00/工作/数据集/视频-fc2a55a8ce9d4263b1ebcb7915a6b18c.mp4',cause='Registered old basename absent; source-file content identity checked before any path amendment.',decoder_started=False,camera_fitted=False,training_started=False)
(REC/'pattern-preflight/initial-failure.json').write_text(json.dumps(failure,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(matches,ensure_ascii=False))
