from pathlib import Path
import json
import numpy as np
from PIL import Image

R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video-update-20261027')
N=R/'native-temporal-material'; M=R/'material-visibility'
seeds=json.loads((N/'seeds.json').read_text())['seeds']
n1=np.load(N/'observations.npz')
reference=Image.open(N/'frame000278.png').convert('RGB')
def crop(image,uv):
    if not np.isfinite(uv).all():return Image.new('RGB',(96,96),(238,238,238))
    x,y=np.rint(uv).astype(int)
    return image.crop((x-48,y-48,x+48,y+48))
rows=[]
for frame in [0,70,266,556]:
    z=np.load(M/f'frame{frame:06d}.npz')
    target=Image.open(N/f'frame{frame:06d}.png').convert('RGB')
    sheet=Image.new('RGB',(288,864),'white')
    for i in range(9):
        for col,im,uv in [(0,reference,seeds[i]['uv']),(1,target,n1['uv'][frame,i]),(2,target,z['uv'][i])]:
            sheet.paste(crop(im,uv),(col*96,i*96))
        rows.append(dict(frame=frame,track=i,confidence=float(z['confidence'][i]),fb=float(z['fb'][i]),eligible=bool(z['eligible'][i]),n1_valid=bool(n1['valid'][frame,i]),roma_minus_n1_px=float(z['difference_from_n1'][i]),roma_uv=z['uv'][i].tolist()))
    sheet.save(M/f'face_review_{frame:06d}.png')
cases=json.loads((N/'counterexamples_index.json').read_text())
sheet=Image.new('RGB',(288,576),'white')
for row,c in enumerate(cases):
    frame,i=c['frame'],c['track'];target=Image.open(N/f'frame{frame:06d}.png').convert('RGB');z=np.load(M/f'frame{frame:06d}.npz')
    for col,im,uv in [(0,reference,seeds[i]['uv']),(1,target,c['uv']),(2,target,z['uv'][i])]:sheet.paste(crop(im,uv),(col*96,row*96))
sheet.save(M/'known_negative_review.png')
(M/'visual-index.json').write_text(json.dumps(dict(columns=['anchor278','temporal_N1','RoMa_N2'],native_crop_size=96,image_no_text_or_warp=True,face_rows=rows,negative_rows=cases),indent=2)+'\n')
print(json.dumps([r for r in rows if r['frame']==266],indent=2))
