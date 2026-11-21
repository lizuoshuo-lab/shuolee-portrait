"""Assemble native face/feet evidence, without resizing or annotations."""
import argparse
from pathlib import Path
from PIL import Image
import audit_gaussianavatars_params123 as a

parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=int,required=True)
checkpoint=parser.parse_args().checkpoint
out=a.REC/'cross-subject-sh0-124/grass-temporal-refinement'/f'review-{checkpoint:03d}'
result=a.read(out/'result.json')
def local(path):return path[5].upper()+':/'+path[7:] if path.startswith('/mnt/') else path
tiles=[];pages=[]
for row in result['records']:
    images=[Image.open(local(row['images'][name]['path'])).convert('RGB') for name in ['face','feet']]
    tile=Image.new('RGB',(max(im.width for im in images),sum(im.height for im in images)))
    y=0
    for im in images:tile.paste(im,(0,y));y+=im.height
    tiles.append((row['frame'],tile))
for start in range(0,len(tiles),2):
    batch=tiles[start:start+2]
    canvas=Image.new('RGB',(sum(im.width for _,im in batch),max(im.height for _,im in batch)))
    x=0
    for _,im in batch:canvas.paste(im,(x,0));x+=im.width
    path=out/f'face-feet-{start//2+1:02d}.png';canvas.save(a.lp(path))
    pages.append(dict(path=str(path),frames=[name for name,_ in batch],size=canvas.size,resized=False))
a.save(out/'face-feet-pages.json',pages)
print(pages)
