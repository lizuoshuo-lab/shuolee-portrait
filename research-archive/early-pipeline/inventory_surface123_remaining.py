"""Read-only inventory of the existing seven-subject queue; no launch or calibration."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
p=read(R/'protocol.json');q=read(R/'queue-state.json');rows=[]
for x in p['subjects']:
    root=Path(x['run_root']);entry=next(y for y in q['subjects'] if y['id']==x['id'])
    files=[]
    if root.exists():
        for f in sorted(root.rglob('*.json')):
            if len(f.relative_to(root).parts)>3:continue
            if any(t in f.parts for t in ['site-packages','node_modules']):continue
            files.append(dict(path=str(f),bytes=f.stat().st_size))
    rows.append(dict(subject=x['id'],label=x['label'],queue_stage=entry['stage'],selected_frames=x['selected_frames'],run_root=str(root),root_exists=root.exists(),top_level=sorted(y.name for y in root.iterdir()) if root.exists() else [],json_files=files))
out=dict(updated=datetime.now(timezone.utc).isoformat(),read_only=True,subjects=rows,no_training_or_camera_fit_started=True)
dst=R/'remaining-subject-inventory.json';dst.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,ensure_ascii=False))
