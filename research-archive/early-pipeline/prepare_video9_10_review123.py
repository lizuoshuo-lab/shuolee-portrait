"""Produce native, unwarped inspection crops from already registered samples."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
from PIL import Image
O=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video9-10-native-intake')
assert not (O/'review-crops.json').exists()
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
data=read(O/'result.json'); entries=[]
for v in data['videos']:
    stem=Path(v['name']).stem
    box=(1100,130,1800,730) if stem=='video9' else (1060,90,1760,690)
    crops=[]
    for s in v['samples']:
        p=O/s['file']; im=Image.open(p).convert('RGB'); dest=p.with_name(p.stem+'_head-context.png')
        im.crop(box).save(dest); crops.append(dest)
        entries.append(dict(source=s['file'],source_sha256=s['sha256'],crop=str(dest.relative_to(O)),box_xyxy=box,
                            resize=False,pixels_unchanged=True,semantic='Inspection region, not tracked correspondence or exact head mask.'))
    for j in range(3):
        page=Image.new('RGB',(700,1800),'white')
        for k in range(3): page.paste(Image.open(crops[j*3+k]),(0,600*k))
        page.save(O/stem/f'heads_native_{j:02d}.png')
(O/'review-crops.json').write_text(json.dumps(dict(updated=datetime.now(timezone.utc).isoformat(),entries=entries,caption_burned=False),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Prepared 18 native crops and 6 unscaled inspection pages.')
