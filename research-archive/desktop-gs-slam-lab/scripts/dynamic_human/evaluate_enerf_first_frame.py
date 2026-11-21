"""Fixed-final-checkpoint native heldout-view evaluation; no model optimization."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,torch,cv2
from PIL import Image,ImageDraw
from gsplat import rasterization
from enerf_native_loader import EnerfNativeDataset
from train_enerf_first_frame import ss,tensor,save

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--manifest',required=True);ap.add_argument('--root',required=True);ap.add_argument('--frame',type=int,default=0);ap.add_argument('--source-frames');a=ap.parse_args();out=Path(a.out);run=out/'first-frame';dest=run/'evaluation';dest.mkdir(exist_ok=True)
    source_frames=json.loads(Path(a.source_frames).read_text())['source_frames'] if a.source_frames else {}
    if (dest/'metrics.json').exists():raise FileExistsError('Prior evaluation preserved')
    comp=json.loads((run/'completion.json').read_text());assert comp['step']==6000
    cp=torch.load(run/'resume.pt',map_location='cpu',weights_only=False);assert cp['step']==6000;p={k:v.cuda() for k,v in cp['params'].items()}
    reports=[]
    with torch.no_grad():
        for split in ['heldout_view','train']:
            ds=EnerfNativeDataset(a.manifest,split,a.root)
            for i,r in enumerate(ds.rows):
                wanted=source_frames.get(r['camera'],a.frame) if split=='train' else a.frame
                matches=r['common_frame']==a.frame if 'common_frame' in r else r['frame']==wanted
                if not matches or (split=='train' and r['camera'] not in ['01','07','13']):continue
                o=ds[i];h,w=o['rgb'].shape[:2]
                image,alpha,_=rasterization(p['means'],p['quats'],p['scales'].exp(),p['opacities'].sigmoid(),torch.cat([p['sh0'],p['shN']],1),tensor(o['w2c'])[None],tensor(o['K'])[None],w,h,sh_degree=2,packed=False,near_plane=.1,far_plane=20,eps2d=.3)
                rgb=(image[0]+.5*(1-alpha[0])).clamp(0,1).cpu().numpy();ref=o['rgb']/255*o['foreground_matte'][...,None]+.5*(1-o['foreground_matte'][...,None]);pred=(rgb*255+.5).astype('uint8');target=(ref*255+.5).astype('uint8');mask=o['foreground_matte']>.9;valid=o['valid_pixels'];interior=mask&valid
                err=(rgb-ref)**2;psnr=-10*np.log10(err[interior].mean())
                name=f"{split}-camera{o['camera']}";Image.fromarray(pred).save(dest/f'{name}-render.png');Image.fromarray(target).save(dest/f'{name}-reference.png');Image.fromarray((o['foreground_matte']*255+.5).astype('uint8')).save(dest/f'{name}-matte.png')
                n,labels,stats,_=cv2.connectedComponentsWithStats((o['foreground_matte']>.5).astype('uint8'));regions=sorted([s for s in stats[1:] if s[4]>10000],key=lambda s:s[0]);person=[]
                for j,s in enumerate(regions):
                    x,y,bw,bh,_=map(int,s);x0=max(x-16,0);y0=max(y-16,0);x1=min(x+bw+16,w);y1=min(y+bh+16,h)
                    pr=pred[y0:y1,x0:x1];gt=target[y0:y1,x0:x1];e=((pr.astype(float)-gt)/255)**2;roi_psnr=float(-10*np.log10(e.mean()));roi_ssim=float(ss(tensor(pr/255).permute(2,0,1)[None],tensor(gt/255).permute(2,0,1)[None]))
                    canv=Image.new('RGB',(pr.shape[1]*2,pr.shape[0]+42),(245,245,245));d=ImageDraw.Draw(canv);d.text((10,12),'Reference (matte + gray)',fill='black');d.text((pr.shape[1]+10,12),'3DGS | unseen camera' if split=='heldout_view' else '3DGS | training camera',fill='black');canv.paste(Image.fromarray(gt),(0,42));canv.paste(Image.fromarray(pr),(pr.shape[1],42));canv.save(dest/f'{name}-person{j+1}-comparison.png')
                    person.append(dict(person_left_to_right=j+1,bbox=[x0,y0,x1,y1],bbox_psnr_db=roi_psnr,bbox_ssim=roi_ssim))
                reports.append(dict(split=split,camera=o['camera'],frame=o['frame'],foreground_psnr_db=float(psnr),foreground_pixels=int(interior.sum()),people=person,source_sha256=o['source_sha256']))
                print(json.dumps(reports[-1]),flush=True)
    save(dest/'metrics.json',dict(status='fixed_final_checkpoint_evaluated_visual_review_pending',checkpoint_sha256=hashlib.sha256((run/'resume.pt').read_bytes()).hexdigest(),step=6000,views=reports,scope='Native 1920x1080, foreground and held objects on neutral gray; neither gray nor reference-photo background is a learned static scene',metrics='Foreground PSNR on valid author matte >.9. Bounding-box PSNR/SSIM include surrounding gray and matte edges; not interchangeable. No heldout fitting or checkpoint selection.',enhancement=False))

if __name__=='__main__':main()
