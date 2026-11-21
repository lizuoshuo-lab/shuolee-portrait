from pathlib import Path
import json,hashlib
from PIL import Image,ImageDraw,ImageFont
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/person-static-native-align-fresh-113')
read=lambda p:json.loads(p.read_text(encoding='utf8'))
review=read(R/'gaussian-run/review-metrics.json');src=R/'gaussian-run/review-current.jpg';data=src.read_bytes();im=Image.open(src).convert('RGB')
assert im.size==(1800,950) and review['step']==1800
# Unaltered actual render pairs, fixed cameras34/70/100, preserving pair labels.
sheet=Image.new('RGB',(1800,520),'white')
for j,box in enumerate([(600,0,1200,475),(1200,0,1800,475),(0,475,600,950)]):sheet.paste(im.crop(box),(600*j,0))
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',24)
ImageDraw.Draw(sheet).text((18,482),'本轮第1,800步阶段渲染 · 每组左侧实拍、右侧渲染 · 训练中间结果，非质量验收',font=font,fill=(30,45,55))
sheet.save(R/'presentation-stage.png')
assert read(R/'gaussian-run/review-metrics.json')['step']==review['step']
(R/'presentation-evidence.json').write_text(json.dumps(dict(step=review['step'],source=str(src),source_sha256=hashlib.sha256(data).hexdigest(),frames=[34,70,100],real_native_render=True,only_crop_and_layout=True,quality_accepted=False),indent=2),encoding='utf8')
