"""Read-only contact sheets for actual visual inspection; no labels or warps."""
import argparse
from pathlib import Path
from PIL import Image

def sheet(paths,out,width=760):
    ims=[]
    for p in paths:
        im=Image.open(p).convert('RGB');im.resize((width,round(im.height*width/im.width)))
        im=im.resize((width,round(im.height*width/im.width)))
        ims.append(im)
    total=Image.new('RGB',(width,sum(x.height for x in ims)),(245,245,245));y=0
    for im in ims:total.paste(im,(0,y));y+=im.height
    total.save(out)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('pattern');p.add_argument('out');p.add_argument('--width',type=int,default=760)
    a=p.parse_args();sheet(sorted(Path(a.folder).glob(a.pattern)),a.out,a.width)
