"""Evaluate time/view holds without fitting; save real paired motion imagery."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,cv2,torch
from PIL import Image,ImageDraw
from gsplat import rasterization
from enerf_native_loader import EnerfNativeDataset
from train_enerf_first_frame import tensor,ss,save

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--manifest',required=True);ap.add_argument('--root',required=True);a=ap.parse_args();out=Path(a.out);run=out/'motion';dest=run/'evaluation';dest.mkdir(exist_ok=True)
    if (dest/'metrics.json').exists():raise FileExistsError('Prior evaluation')
    completion=json.loads((run/'completion.json').read_text());frames=completion['frames'];assert frames[-1]==59
    protocol=json.loads((run/'protocol.json').read_text());cache={};hashes={};results=[];boxes={};p0=torch.load(run/'frame-000.pt',map_location='cuda',weights_only=False)
    def load(t):
        if t not in cache:
            if len(cache)>2:cache.pop(next(iter(cache)))
            path=run/f'frame-{t:03d}.pt';hashes[t]=hashlib.sha256(path.read_bytes()).hexdigest();cache[t]=torch.load(path,weights_only=False,map_location='cuda')
        return cache[t]
    def params(t):
        if t in frames:return load(t)
        a=max(f for f in frames if f<t);b=min(f for f in frames if f>t);v=(t-a)/(b-a);pa,pb=load(a),load(b);p={}
        for k in pa:
            if k in ('scales','opacities'):assert torch.equal(pa[k],pb[k]);p[k]=pa[k]
            elif k=='quats':
                qa=torch.nn.functional.normalize(pa[k],dim=-1);qb=torch.nn.functional.normalize(pb[k],dim=-1);qb=qb*torch.where((qa*qb).sum(-1,keepdim=True)<0,-1.,1.);p[k]=torch.nn.functional.normalize(qa*(1-v)+qb*v,dim=-1)
            else:p[k]=pa[k]*(1-v)+pb[k]*v
        return p
    def render(p,o):
        h,w=o['rgb'].shape[:2];r,al,_=rasterization(p['means'],p['quats'],p['scales'].exp(),p['opacities'].sigmoid(),torch.cat([p['sh0'],p['shN']],1),tensor(o['w2c'])[None],tensor(o['K'])[None],w,h,sh_degree=2,packed=False,near_plane=.1,far_plane=20,eps2d=.3)
        return ((r[0]+.5*(1-al[0])).clamp(0,1).cpu().numpy()*255+.5).astype('uint8')
    with torch.no_grad():
        for split in ('heldout_view','heldout_time'):
            ds=EnerfNativeDataset(a.manifest,split,a.root)
            for i,row in enumerate(ds.rows):
                o=ds[i];t=o['frame'];cam=o['camera'];p=params(t);pred=render(p,o);ref=(o['rgb']/255*o['foreground_matte'][...,None]+.5*(1-o['foreground_matte'][...,None]));gt=(ref*255+.5).astype('uint8');mask=(o['foreground_matte']>.9)&o['valid_pixels'];error=((pred.astype(float)-gt)/255)**2;score=float(-10*np.log10(error[mask].mean()))
                stem=f'camera{cam}-frame{t:03d}';Image.fromarray(pred).save(dest/f'{stem}-render.png',compress_level=1);Image.fromarray(gt).save(dest/f'{stem}-reference.png',compress_level=1);Image.fromarray((o['foreground_matte']*255+.5).astype('uint8')).save(dest/f'{stem}-matte.png',compress_level=1)
                y,x=np.where(o['foreground_matte']>.5);box=[max(0,int(x.min())-20),max(0,int(y.min())-20),min(pred.shape[1],int(x.max())+21),min(pred.shape[0],int(y.max())+21)]
                boxes.setdefault(cam,[]).append(box);x0,y0,x1,y1=box;bbox_psnr=float(-10*np.log10(error[y0:y1,x0:x1].mean()))
                item=dict(camera=cam,frame=t,split=split,interpolated=t not in frames,foreground_psnr_db=score,bbox_psnr_db=bbox_psnr,bbox=box,source_sha256=o['source_sha256'])
                if cam in protocol['heldout_cameras']:
                    base=render(p0,o);Image.fromarray(base).save(dest/f'{stem}-static.png');item['static_foreground_psnr_db']=float(-10*np.log10((((base.astype(float)-gt)/255)**2)[mask].mean()))
                results.append(item)
                if len(results)%30==0:print('EVALUATED',len(results),flush=True)
    assert len(results)==270 and len(set((r['camera'],r['frame']) for r in results))==270
    # Fixed union crop across all 60 times, so the preview does not hide jitter.
    for cam in protocol['heldout_cameras']:
        bb=np.array(boxes[cam]);box=[int(bb[:,0].min()),int(bb[:,1].min()),int(bb[:,2].max()),int(bb[:,3].max())];x0,y0,x1,y1=box;pairdir=dest/f'video-camera{cam}';pairdir.mkdir(exist_ok=True)
        for t in range(60):
            stem=f'camera{cam}-frame{t:03d}';gt=Image.open(dest/f'{stem}-reference.png').crop(box);pr=Image.open(dest/f'{stem}-render.png').crop(box);w,h=gt.size;panel=Image.new('RGB',(w*2,h+42),(245,245,245));dr=ImageDraw.Draw(panel);dr.text((12,12),f'Reference | frame {t:03d}',fill='black');dr.text((w+12,12),'3DGS | unseen camera'+(' + unseen time' if t not in frames else ''),fill='black');panel.paste(gt,(0,42));panel.paste(pr,(w,42));panel.save(pairdir/f'{t:03d}.png')
            if t in [0,15,30,45,59]:panel.save(dest/f'{stem}-comparison.png')
        boxes[cam]=box
    view=[r for r in results if r['split']=='heldout_view'];tim=[r for r in results if r['split']=='heldout_time'];holdcam=[r for r in results if 'static_foreground_psnr_db' in r]
    report=dict(status='all270_native_heldout_pairs_rendered_visual_review_pending',views=results,summary=dict(heldout_view_count=len(view),heldout_view_psnr_mean=float(np.mean([r['foreground_psnr_db'] for r in view])),heldout_time_count=len(tim),heldout_time_psnr_mean=float(np.mean([r['foreground_psnr_db'] for r in tim])),three_heldout_cameras_all60_mean=float(np.mean([r['foreground_psnr_db'] for r in holdcam])),frozen_first_frame_same180_mean=float(np.mean([r['static_foreground_psnr_db'] for r in holdcam]))),checkpoint_hashes=hashes,video_fixed_boxes={c:boxes[c] for c in protocol['heldout_cameras']},playback='Encode at 30 fps for viewing only; capture frame rate not independently established',metrics='Saved 8-bit PNG foreground PSNR within author matte >.9 and rectification support; author matte includes held objects and imperfect borders',scope='Foreground motion over first 60 frames, no learned static background; unseen times interpolated between fitted states, not future prediction')
    save(dest/'metrics.json',report);print(json.dumps(report['summary']),flush=True)

if __name__=='__main__':main()
