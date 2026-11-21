from pathlib import Path
from datetime import datetime, timezone
from PIL import Image, ImageDraw
import hashlib, json, shutil
O=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video10-temporal-bridge')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
p=json.loads((O/'regions.json').read_text(encoding='utf-8'))
assert not (O/'run').exists() and not (O/'region-review.json').exists()
shutil.copy2(O/'regions.json',O/'regions-before-final-tv-correction.json')
shutil.copy2(O/'regions_01.jpg',O/'regions_01_before_final_tv_correction.jpg')
next(r for r in p['rows'] if r['frame']==161)['exclude_background'].append([2300,380,2704,1160])
p['final_tv_correction']='Frame161 television at right edge was partly uncovered. Added explicit exclusion before any feature extraction; page01 must be checked again.'
save(O/'regions.json',p)
page=Image.new('RGB',(1352,2280),'white')
for k,r in enumerate(p['rows'][3:6]):
 with Image.open(r['source']) as im: im=im.convert('RGBA')
 layer=Image.new('RGBA',im.size,(0,0,0,0));dr=ImageDraw.Draw(layer)
 for b in r['exclude_background']:dr.rectangle(b,fill=(235,40,40,75),outline=(235,40,40,255),width=4)
 dr.rectangle(r['body'],fill=(120,40,220,45),outline=(120,40,220,255),width=6);dr.rectangle(r['head'],outline=(255,180,0,255),width=6)
 page.paste(Image.alpha_composite(im,layer).convert('RGB').resize((1352,760)),(0,760*k))
page.save(O/'regions_01.jpg',quality=93)
print('TV exclusion corrected before extraction; review corrected page01 before admission.')
