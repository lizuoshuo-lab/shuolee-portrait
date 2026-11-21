from pathlib import Path
import time,shutil,json
from PIL import Image
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=B/'shuolee/current/person-clothing-pipeline-fresh-060/gaussian-run';P=B/'viewer/body-clothing-current'
seen={};start=time.monotonic()
while time.monotonic()-start<1800:
    for name in ['progress.json','review-current.jpg','novel-current.jpg']:
        src=R/name
        if not src.exists() or seen.get(name)==src.stat().st_mtime_ns:continue
        try:
            stamp=src.stat().st_mtime_ns
            if name.endswith('.jpg'):
                with Image.open(src) as im:im.verify()
            else:json.loads(src.read_text())
            tmp=P/(name+'.tmp');shutil.copyfile(src,tmp)
            if src.stat().st_mtime_ns!=stamp:continue
            tmp.replace(P/name);seen[name]=stamp
        except (OSError,ValueError):pass
    if (R/'result.json').exists():break
    time.sleep(5)
print('Public progress synchronization stopped.')
