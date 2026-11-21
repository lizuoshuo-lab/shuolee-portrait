from pathlib import Path
import argparse,numpy as np,torch
from PIL import Image,ImageDraw
from gsplat import rasterization
from enerf_native_loader import EnerfNativeDataset
from train_enerf_first_frame import tensor

ap=argparse.ArgumentParser();ap.add_argument('--frame',type=int,required=True);ap.add_argument('--out',required=True);ap.add_argument('--manifest',required=True);ap.add_argument('--root',required=True);a=ap.parse_args();out=Path(a.out);dest=out/'motion/training-preview';dest.mkdir(exist_ok=True);ds=EnerfNativeDataset(a.manifest,'train',a.root)
p=torch.load(out/f'motion/frame-{a.frame:03d}.pt',map_location='cuda',weights_only=False)
with torch.no_grad():
    for i,r in enumerate(ds.rows):
        if r['frame']!=a.frame or r['camera'] not in ['01','07']:continue
        o=ds[i];h,w=o['rgb'].shape[:2];im,al,_=rasterization(p['means'],p['quats'],p['scales'].exp(),p['opacities'].sigmoid(),torch.cat([p['sh0'],p['shN']],1),tensor(o['w2c'])[None],tensor(o['K'])[None],w,h,sh_degree=2,packed=False,near_plane=.1,far_plane=20,eps2d=.3);pr=((im[0]+.5*(1-al[0])).clamp(0,1).cpu().numpy()*255+.5).astype('uint8');gt=(o['rgb']*o['foreground_matte'][...,None]+127.5*(1-o['foreground_matte'][...,None])+.5).astype('uint8')
        y,x=np.where(o['foreground_matte']>.5);x0=max(0,int(x.min())-16);y0=max(0,int(y.min())-16);x1=min(w,int(x.max())+17);y1=min(h,int(y.max())+17);pr=pr[y0:y1,x0:x1];gt=gt[y0:y1,x0:x1];hh,ww=pr.shape[:2];panel=Image.new('RGB',(ww*2,hh+40),(245,245,245));d=ImageDraw.Draw(panel);d.text((10,12),'Reference',fill='black');d.text((ww+10,12),f'Training camera {r["camera"]}, frame {a.frame}',fill='black');panel.paste(Image.fromarray(gt),(0,40));panel.paste(Image.fromarray(pr),(ww,40));panel.save(dest/f'camera{r["camera"]}-frame{a.frame:03d}.png')
