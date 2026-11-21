from pathlib import Path
import json,shutil
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/grayT-scene-support-cross-run-review')
L=Path('D:/00/工作日志/2026-10-29/人物对齐核验123')
p=R/'index.json';d=json.loads(p.read_text(encoding='utf-8'))
target=next(x for x in d['images'] if x['file']=='company00_FIT_02_d0326813da_native.png')
old=target['region'];new=old.replace('部分节点未进入BA','全部节点进入BA')
target['region']=new
p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
shutil.copy2(p,L/R.name/p.name)
log=L/'index.html';s=log.read_text(encoding='utf-8');assert s.count(old)==1
log.write_text(s.replace(old,new),encoding='utf-8')
print('All-seven-node original family caption corrected.')
