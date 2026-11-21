from pathlib import Path
from PIL import Image
import json,hashlib
H=Path(__file__).parent
REC=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
O=REC/'grayT-fixed-direction-review'
A=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-local-surface-123/grayT')
assert not (O/'prepared.json').exists()
O.mkdir(exist_ok=True)
records=json.loads((A/'data/inputs.json').read_text(encoding='utf-8'))['records']
crops={48:[1160,1060,1520,1340],56:[1070,1090,1520,1430],64:[1090,1100,1520,1450],72:[1180,1100,1520,1450],80:[1000,1100,1400,1410]}
items=[]
for n,box in crops.items():
 p=A/f'raw/frame_{n:06d}.png';im=Image.open(p);assert im.size==(1520,2704)
 dest=O/f'frame_{n:03d}_native_rail_crop.png';im.crop(box).save(dest)
 items.append(dict(frame=n,crop=box,image=str(p),image_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),file=dest.name))
(O/'prepared.json').write_text(json.dumps(dict(purpose='Visual seed selection before fitting directions; no new feature extraction or camera update',original_frames_examined=[40,48,56,64,72,80],frame40_not_used='Gray foreground railing too short at this view; no substitute dark-fence family silently assumed parallel',crops=items),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(str(O))
