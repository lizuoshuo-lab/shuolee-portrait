"""Three frozen models, identical native references, no evaluation optimization."""
from pathlib import Path
import json,hashlib,argparse
import cv2,numpy as np,torch
from PIL import Image
from gsplat import rasterization
from enerf_native_loader import EnerfNativeDataset
from train_enerf_first_frame import tensor,ss,save

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--root',required=True);a=ap.parse_args();out=Path(a.out);dest=out/'evaluation';dest.mkdir(exist_ok=True);assert not (dest/'metrics.json').exists();torch.set_num_threads(8)
    frames=list(range(34,59));cams=['00','06','12'];groups=['static','same-index','offset-index'];state_hashes={};baseframes={}
    for g in ['same-index','offset-index']:
        comp=json.loads((out/g/'motion/completion.json').read_text());assert comp['frames']==frames and comp['updates']==14400
        spec=json.loads((out/g/'appearance-manifest.json').read_text());baseframes[g]={(r['camera'],r['common_frame']):r for r in spec['images'] if r['split']=='heldout_view'}
    for key,r in baseframes['same-index'].items():assert r['sha256']==baseframes['offset-index'][key]['sha256'] and r['mask_sha256']==baseframes['offset-index'][key]['mask_sha256']
    assert json.loads((out/'offset-index/static-multitime/completion.json').read_text())['total_steps']==20400
    def load(path):
        state_hashes[path.relative_to(out).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest();p=torch.load(path,weights_only=False,map_location='cuda');assert all(torch.isfinite(v).all() for v in p.values());return p
    static=load(out/'offset-index/static-multitime/model.pt');ds=EnerfNativeDataset(out/'offset-index/appearance-manifest.json','heldout_view',a.root);lookup={(r['camera'],r['common_frame']):i for i,r in enumerate(ds.rows)};rows=[]
    with torch.no_grad():
        for t in frames:
            models=dict(static=static)
            for g in groups[1:]:models[g]=load(out/g/'motion'/f'frame-{t:03d}.pt')
            for cam in cams:
                o=ds[lookup[cam,t]];h,w=o['rgb'].shape[:2];stem=f'camera{cam}-frame{t:03d}';matte=o['foreground_matte'];valid=o['valid_pixels'];ref=o['rgb']/255*matte[...,None]+.5*(1-matte[...,None]);gt=np.rint(ref*255).clip(0,255).astype('uint8');inside=(matte>.9)&valid
                Image.fromarray(gt).save(dest/f'{stem}-reference.png',compress_level=1);Image.fromarray(np.rint(matte*255).astype('uint8')).save(dest/f'{stem}-matte.png',compress_level=1)
                _,labels,stats,_=cv2.connectedComponentsWithStats((matte>.5).astype('uint8'));regions=sorted([s for s in stats[1:] if s[4]>10000],key=lambda s:s[0]);assert len(regions)==2
                outside=(cv2.dilate((matte>.1).astype('uint8'),np.ones((33,33),'uint8'))==0)&valid
                for group,p in models.items():
                    r,al,_=rasterization(p['means'],p['quats'],p['scales'].exp(),p['opacities'].sigmoid(),torch.cat([p['sh0'],p['shN']],1),tensor(o['w2c'])[None],tensor(o['K'])[None],w,h,sh_degree=2,packed=False,near_plane=.1,far_plane=20,eps2d=.3)
                    pred=((r[0]+.5*(1-al[0])).clamp(0,1).cpu().numpy()*255+.5).astype('uint8');Image.fromarray(pred).save(dest/f'{stem}-{group}.png',compress_level=1);err=((pred.astype(float)-gt)/255)**2;people=[]
                    for index,s in enumerate(regions):
                        x,y,bw,bh,_=map(int,s);box=[max(0,x-16),max(0,y-16),min(w,x+bw+16),min(h,y+bh+16)];x0,y0,x1,y1=box;roi=err[y0:y1,x0:x1];pr=tensor(pred[y0:y1,x0:x1]/255).permute(2,0,1)[None];gr=tensor(gt[y0:y1,x0:x1]/255).permute(2,0,1)[None];upper=[x0,y0,x1,min(y1,y0+(y1-y0)//2)]
                        people.append(dict(person=index+1,bbox=box,psnr_db=float(-10*np.log10(roi.mean())),ssim=float(ss(pr,gr)),upper_body_bbox=upper,upper_body_psnr_db=float(-10*np.log10(err[upper[1]:upper[3],upper[0]:upper[2]].mean()))))
                    row=dict(common_frame=t,source_frame=o['frame'],camera=cam,model=group,foreground_psnr_db=float(-10*np.log10(err[inside].mean())),full_canvas_psnr_db=float(-10*np.log10(err[valid].mean())),outside_foreground_color_pixels=int((np.max(np.abs(pred.astype(float)/255-.5),axis=2)[outside]>.1).sum()),people=people,source_sha256=o['source_sha256']);rows.append(row)
            print('EVALUATED_COMMON_FRAME',t,flush=True)
    assert len(rows)==225
    summary={g:dict(images=75,foreground_psnr_mean=float(np.mean([r['foreground_psnr_db'] for r in rows if r['model']==g])),upper_body_psnr_mean=float(np.mean([p['upper_body_psnr_db'] for r in rows if r['model']==g for p in r['people']])),outside_pixels_mean=float(np.mean([r['outside_foreground_color_pixels'] for r in rows if r['model']==g]))) for g in groups}
    result=dict(status='225frozen_native_renders_complete_independent_and_visual_review_pending',frames=frames,cameras=cams,models=groups,views=rows,summary=summary,state_hashes=state_hashes,scope='75 identical appearance-heldout references per model; cameras had separate earlier temporal calibration, not globally unseen. No temporal holdout in this exploratory development study. Two foreground people and toys on gray, no learned scene background.',metrics='8bitPNG PSNR on author foreground matte>.9 and rectification-valid pixels. Person bbox SSIM and upper-half bbox PSNR include gray; upper-half is not a hand segmentation metric. Outside-color counts beyond16px matte dilation are artifact indicators, not ground-truth alpha error.',enhancement=False)
    save(dest/'metrics.json',result);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
