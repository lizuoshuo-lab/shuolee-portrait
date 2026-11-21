"""Source-only native crop sheet, before the 072 model is evaluated."""
from pathlib import Path
import numpy as np,json,hashlib
from PIL import Image,ImageDraw,ImageFont
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-forearm-matte-repair-072';I=L/'outputs/dynamic-human-enerf-articulated-control-064'
out=R/'raw-forearm-evaluation';out.mkdir(exist_ok=True)
frames=[749,751,753,755];x0,y0,x1,y1=660,420,870,640
sheet=Image.new('RGB',(4*250,270),'white');d=ImageDraw.Draw(sheet);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',11)
for i,t in enumerate(frames):
 z=np.load(I/'native-input'/f'camera01-frame{t:06d}.npz');sheet.paste(Image.fromarray(z['rgb'][y0:y1,x0:x1]),(i*250+35,40));d.text((i*250+35,2),f'camera01 frame{t}',fill='black',font=font)
 for x in [660,700,740,780,820,860]:d.text((i*250+35+x-x0-8,25),str(x),fill='black',font=font)
 for y in [420,460,500,540,580,620]:d.text((i*250+3,40+y-y0-4),str(y),fill='black',font=font)
sheet.save(out/'source-only-selection.png')
print('Saved four source-only native crops; no model predictions read.')
