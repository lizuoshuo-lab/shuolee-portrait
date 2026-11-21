from pathlib import Path
from PIL import Image
import json,hashlib
H=Path(__file__).parent
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/native-residual-pipeline')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
assert len(read(R/'cpu-result.json')['checks'])==22 and len(read(R/'gpu-result.json')['checks'])==14
sheet=Image.new('RGB',(192,80));sources=[]
for row,kind in enumerate(['target','readback']):
 for col in range(4):
  p=R/f'synthetic-native-images/{kind}-{col}.png';im=Image.open(p);assert im.size==(48,40)
  sheet.paste(im,(col*48,row*40));sources.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
sheet.save(R/'synthetic-native-comparison.png')
(R/'synthetic-figure.json').write_text(json.dumps(dict(path=str(R/'synthetic-native-comparison.png'),width=192,height=80,rows=['known target','trained native NPZ readback'],columns=['camera0','camera1','camera2','camera3'],native_pixels_unscaled=True,text_burned_into_image=False,sources=sources),indent=2),encoding='utf-8')
print(R/'synthetic-native-comparison.png')
