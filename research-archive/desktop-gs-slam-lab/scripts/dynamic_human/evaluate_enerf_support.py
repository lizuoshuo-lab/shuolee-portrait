"""Frozen factorial study evaluation: RGB, silhouette and native gradient fidelity."""
from pathlib import Path
import json,hashlib,argparse
import cv2,numpy as np,torch
from PIL import Image
from gsplat import rasterization
from enerf_native_loader import EnerfNativeDataset
from train_enerf_first_frame import tensor,ss,save

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--root',required=True);ap.add_argument('--split',choices=['heldout_view','train'],default='heldout_view');a=ap.parse_args();out=Path(a.out);dest=out/('evaluation' if a.split=='heldout_view' else 'training-evaluation');dest.mkdir(exist_ok=True);assert not (dest/'metrics.json').exists();torch.set_num_threads(8)
    source=out.parent/'dynamic-human-enerf-local-time-029/offset-index';manifest=source/'appearance-manifest.json';protocol=json.loads((out/'protocol.json').read_text());assert sha(manifest)==protocol['manifest_sha256'];frames=protocol['frames'] if a.split=='heldout_view' else [34,46,58];groups=protocol['conditions'];cams=['00','06','12'] if a.split=='heldout_view' else ['01','07','14'];hashes={};rows=[]
    for g in groups:
        done=json.loads((out/g/'motion/completion.json').read_text());spec=json.loads((out/g/'motion/protocol.json').read_text());assert done['frames']==protocol['frames'] and done['updates']==14400 and done['points']==150543;assert spec['initial_checkpoint_sha256']==protocol['shared_initial_checkpoint_sha256'] and spec['source_manifest_sha256']==protocol['manifest_sha256']
    ds=EnerfNativeDataset(manifest,a.split,a.root);lookup={(r['camera'],r['common_frame']):i for i,r in enumerate(ds.rows)}
    def gradient(rgb):
        gray=cv2.cvtColor(rgb.astype('float32')/255,cv2.COLOR_RGB2GRAY)
        return np.stack([cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)/8,cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)/8],-1)
    with torch.no_grad():
        for t in frames:
            models={}
            for g in groups:
                path=out/g/'motion'/f'frame-{t:03d}.pt';hashes[path.relative_to(out).as_posix()]=sha(path);p=torch.load(path,weights_only=False,map_location='cuda');assert all(torch.isfinite(v).all() for v in p.values());models[g]=p
            for cam in cams:
                o=ds[lookup[cam,t]];h,w=o['rgb'].shape[:2];stem=f'camera{cam}-frame{t:03d}';matte=o['foreground_matte'];valid=o['valid_pixels'];gt=np.rint((o['rgb']/255*matte[...,None]+.5*(1-matte[...,None]))*255).clip(0,255).astype('uint8');inside=(matte>.9)&valid;sil=(matte>.5)&valid;gt_gradient=gradient(gt)
                Image.fromarray(gt).save(dest/f'{stem}-reference.png',compress_level=1);Image.fromarray(np.rint(matte*255).astype('uint8')).save(dest/f'{stem}-matte.png',compress_level=1)
                _,_,stats,_=cv2.connectedComponentsWithStats((matte>.5).astype('uint8'));regions=sorted([s for s in stats[1:] if s[4]>10000],key=lambda s:s[0]);assert len(regions)==2
                outside=(cv2.dilate((matte>.1).astype('uint8'),np.ones((33,33),'uint8'))==0)&valid
                for g,p in models.items():
                    r,al,_=rasterization(p['means'],p['quats'],p['scales'].exp(),p['opacities'].sigmoid(),torch.cat([p['sh0'],p['shN']],1),tensor(o['w2c'])[None],tensor(o['K'])[None],w,h,sh_degree=2,packed=False,near_plane=.1,far_plane=20,eps2d=.3)
                    pred=((r[0]+.5*(1-al[0])).clamp(0,1).cpu().numpy()*255+.5).astype('uint8');alpha=(al[0,...,0].clamp(0,1).cpu().numpy()*255+.5).astype('uint8');Image.fromarray(pred).save(dest/f'{stem}-{g}.png',compress_level=1);Image.fromarray(alpha).save(dest/f'{stem}-{g}-alpha.png',compress_level=1);err=((pred.astype(float)-gt)/255)**2;graderr=np.abs(gradient(pred)-gt_gradient);people=[]
                    for index,s in enumerate(regions):
                        x,y,bw,bh,_=map(int,s);box=[max(0,x-16),max(0,y-16),min(w,x+bw+16),min(h,y+bh+16)];x0,y0,x1,y1=box;upper=[x0,y0,x1,y0+(y1-y0)//2];u0,v0,u1,v1=upper
                        pr=tensor(pred[y0:y1,x0:x1]/255).permute(2,0,1)[None];gr=tensor(gt[y0:y1,x0:x1]/255).permute(2,0,1)[None]
                        people.append(dict(person=index+1,bbox=box,psnr_db=float(-10*np.log10(err[y0:y1,x0:x1].mean())),ssim=float(ss(pr,gr)),upper_body_bbox=upper,upper_body_psnr_db=float(-10*np.log10(err[v0:v1,u0:u1].mean())),upper_body_gradient_mae=float(graderr[v0:v1,u0:u1].mean())))
                    predsil=(alpha>=128)&valid;union=sil|predsil
                    rows.append(dict(camera=cam,common_frame=t,source_frame=o['frame'],model=g,source_sha256=o['source_sha256'],foreground_psnr_db=float(-10*np.log10(err[inside].mean())),full_canvas_psnr_db=float(-10*np.log10(err[valid].mean())),foreground_gradient_mae=float(graderr[inside].mean()),silhouette_iou=float((sil&predsil).sum()/union.sum()),alpha_foreground_mae=float(np.abs(alpha/255-matte)[sil].mean()),outside_alpha_mass_pixels=float((alpha[outside].astype(float)/255).sum()),outside_foreground_color_pixels=int((np.max(np.abs(pred.astype(float)/255-.5),axis=2)[outside]>.1).sum()),people=people))
            print('EVALUATED_FRAME',t,flush=True)
    assert len(rows)==len(frames)*len(cams)*len(groups)
    summary={g:dict(images=75,foreground_psnr_mean=float(np.mean([r['foreground_psnr_db'] for r in rows if r['model']==g])),upper_body_psnr_mean=float(np.mean([p['upper_body_psnr_db'] for r in rows if r['model']==g for p in r['people']])),silhouette_iou_mean=float(np.mean([r['silhouette_iou'] for r in rows if r['model']==g])),foreground_gradient_mae_mean=float(np.mean([r['foreground_gradient_mae'] for r in rows if r['model']==g])),upper_body_gradient_mae_mean=float(np.mean([p['upper_body_gradient_mae'] for r in rows if r['model']==g for p in r['people']])),outside_pixels_mean=float(np.mean([r['outside_foreground_color_pixels'] for r in rows if r['model']==g]))) for g in groups}
    for g in groups:summary[g]['images']=len(frames)*len(cams)
    scope='Same75appearance-heldout references as029; cameras calibration-visible in separate earlier times. No temporal holdout, no learned background. All states fixed before evaluation.' if a.split=='heldout_view' else '9fixed TRAINING-view probes per model at34,46,58 and cameras01,07,14; supervised appearances, not heldout quality or generalization.'
    save(dest/'metrics.json',dict(status=f'{len(rows)}frozen_native_renders_complete_independent_and_visual_review_pending',split=a.split,frames=frames,cameras=cams,models=groups,views=rows,summary=summary,state_hashes=hashes,scope=scope,metrics='PNG RGB quantities; foreground matte>.9 valid pixels. Threshold0.5 silhouette IoU against author matte, not independently labeled human geometry. Gradient is normalized luma Sobel/8 error against reference, not raw sharpness; upper boxes include gray and held objects, not segmented hands.',enhancement=False));print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
