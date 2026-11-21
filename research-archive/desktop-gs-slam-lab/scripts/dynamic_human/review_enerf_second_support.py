from pathlib import Path
import json,argparse
import numpy as np
from PIL import Image,ImageDraw
L=Path(__file__).resolve().parents[2];S=L/'outputs/dynamic-human-enerf-second-interval-079';O=L/'outputs/dynamic-human-enerf-support-correction-081'
ap=argparse.ArgumentParser();ap.add_argument('--final',action='store_true');a=ap.parse_args()
folder=O/('accepted-input' if a.final else 'candidates');dest=O/('review-final' if a.final else 'review-draft');dest.mkdir(exist_ok=True)
for camera in ['01','02','03','04','05','07','08','09','10','11','13','14','15','16','17']:
 files=[folder/f'camera{camera}-frame{t:06d}.npz' for t in range(720,733,2)]
 if not all(p.exists() for p in files):continue
 tiles=[]
 for t,f in zip(range(720,733,2),files):
  z=np.load(S/'native-input'/f.name);q=np.load(f);x0,y0,x1,y1=z['bbox'];rgb=z['rgb'];matte=q['matte'];target=np.rint(rgb*matte[...,None]+127.5*(1-matte[...,None])).astype(np.uint8)
  removed=(z['matte']>.5)&(matte<.5);overlay=rgb.copy();overlay[removed]=np.rint(.4*rgb[removed]+.6*np.array([255,0,0])).astype(np.uint8)
  w=round((x1-x0)*400/(y1-y0));tile=Image.new('RGB',(w,842),'white');d=ImageDraw.Draw(tile)
  for j,arr in enumerate([overlay,target]):tile.paste(Image.fromarray(arr[y0:y1,x0:x1]).resize((w,400)),(0,22+420*j))
  d.text((3,4),f'{camera}/{t} removed red',fill='black');d.text((3,424),'Training target',fill='black');tiles.append(tile)
 sheet=Image.new('RGB',(sum(i.width for i in tiles),842),'white');x=0
 for im in tiles:sheet.paste(im,(x,0));x+=im.width
 sheet.save(dest/f'camera{camera}-seven-frames.png');print(camera,flush=True)
