from pathlib import Path
from PIL import Image
import json,hashlib
H=Path(__file__).parent
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
F=R/'winterhat-train-006000-review'
paths=[F/'frame_000067_body.png',F/'frame_000067_face.png',F/'frame_000112_face.png']
im=[Image.open(p).convert('RGB') for p in paths]
assert [x.height for x in im]==[480,240,240]
sheet=Image.new('RGB',(im[0].width+24+max(im[1].width,im[2].width),480),'white')
sheet.paste(im[0],(0,0));sheet.paste(im[1],(im[0].width+24,0));sheet.paste(im[2],(im[0].width+24,240))
out=R/'surface123-6000-figure.png';sheet.save(out)
(H/'surface123-6000-figure.json').write_text(json.dumps(dict(path=str(out),width=sheet.width,height=sheet.height,
    sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
    image_content_unchanged=True,text_in_pixels=False),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
template=(H/'preserve_deck_residualFit_122.py').read_text(encoding='utf-8')
(H/'preserve_deck_surface123_6000.py').write_text(template.replace('residualFit-122','surface123-6000'),encoding='utf-8')
print(out)
