"""Native renderer metrics plus controlled gsplat scale/path diagnostics."""
import argparse
import json
import os
import subprocess
import sys
import numpy as np
import torch
from local_refinement import ROOT,BASE,cameras,read_model,render,evaluate,image_tensor,scores,ssim,SUBJECT_BOX,save_image,write_json

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args()
    out=ROOT/a.name;manifest=json.loads((out/'manifest.json').read_text());assert manifest['status']=='complete'
    runtime='/opt/gs-slam-lab/mini-splatting2-native-mip-001';repo=runtime+'/repo'
    env=os.environ.copy();env.update(PYTHONPATH=runtime+'/deps:'+repo+':/opt/gs-slam-lab/mini-splatting2-native-001/deps',OMP_NUM_THREADS='4')
    target=out/'model'
    with (out/'render-native.log').open('w') as log:
        subprocess.run([sys.executable,repo+'/render.py','-m',str(target),'--iteration','18000','--skip_train'],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    torch.set_num_threads(4)
    import lpips
    metric=lpips.LPIPS(net='alex').cuda().eval()
    cams=cameras();rows=[];saved=target/'test/ours_18000'
    for j,c in enumerate([c for c in cams if c['is_test']]):
        pred=image_tensor(saved/'renders'/f'{j:05d}.png');gt=image_tensor(saved/'gt'/f'{j:05d}.png');h,w=gt.shape[:2]
        box=SUBJECT_BOX[int(c['img_name'])];x0,y0,x1,y1=[round(v*s) for v,s in zip(box,[w,h,w,h])]
        row=dict(frame=c['img_name'],full=scores(pred,gt),subject_crop=scores(pred[y0:y1,x0:x1],gt[y0:y1,x0:x1]),
                 ssim=float(ssim(pred.permute(2,0,1),gt.permute(2,0,1))),
                 lpips=float(metric(pred.permute(2,0,1)[None]*2-1,gt.permute(2,0,1)[None]*2-1)))
        rows.append(row)
    del metric;torch.cuda.empty_cache()
    model=read_model(manifest['baked_ply'])
    scale=evaluate(model,cams,out/'gsplat-evaluation','mip',None)
    probe=[]
    for j,c in enumerate([c for c in cams if c['is_test']]):
        pred,_,_=render(model,c,mode='mip');native=image_tensor(saved/'renders'/f'{j:05d}.png')
        probe.append(dict(frame=c['img_name'],difference=scores(pred.clamp(0,1),native)))
    paths=json.loads((ROOT/'geometry-evaluation/paths.json').read_text())['frames']
    for path in [p for p in paths if p['index'] in [0,15,45]]:
        c=dict(cams[24]);R=np.array(path['rotation']);P=np.array(path['position']);V=np.eye(4);V[:3,:3]=R.T;V[:3,3]=-R.T@P
        c['viewmat']=torch.tensor(V,dtype=torch.float32,device='cuda')[None]
        im,_,_=render(model,c,divisor=2,mode='mip');save_image(out/f'{path["kind"]}-{path["index"]:02d}.png',im)
    write_json(out/'evaluation-summary.json',dict(status='complete',gaussians=len(model['means']),native_rows=rows,
        native_mean=dict(full_psnr=float(np.mean([r['full']['psnr'] for r in rows])),crop_psnr=float(np.mean([r['subject_crop']['psnr'] for r in rows])),
            ssim=float(np.mean([r['ssim'] for r in rows])),lpips_alex=float(np.mean([r['lpips'] for r in rows]))),
        scale_gsplat=scale,renderer_probe=probe,
        scope='same 8 development appearance holdouts; full native rasterizer used for native metrics; fixed full-image foreground rectangles; no heldout masks used; gsplat uses the 3D-baked PLY plus 2D antialiasing exactly once'))
    print(json.dumps(rows),flush=True)

if __name__=='__main__':main()
