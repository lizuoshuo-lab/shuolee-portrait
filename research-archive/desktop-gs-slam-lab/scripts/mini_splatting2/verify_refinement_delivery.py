"""Verify PLY filter baking and independent perceptual metrics."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import lpips
from local_refinement import BASE,ROOT,PLY,read_model,cameras,render,image_tensor,ssim,write_json
from evaluate_geometry import restore


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    assert hashlib.sha256(PLY.read_bytes()).hexdigest()=='0751327e63353b2c1cf9db302f8f5861d72571879aaade8e791a32abfc823eb8'
    cams=cameras();roundtrip=[]
    for name in ['mip-refine-s0','mip-depth-refine-s0']:
        model,mode,filt=restore(name);baked=read_model(ROOT/name/'point_cloud.ply')
        for div in [1,8]:
            a,_,_=render(model,cams[24],divisor=div,mode=mode,filt=filt)
            b,_,_=render(baked,cams[24],divisor=div,mode=mode)
            mae=float((a-b).abs().mean());mx=float((a-b).abs().max())
            assert mae<1e-5,(name,div,mae)
            roundtrip.append(dict(name=name,divisor=div,mae=mae,max_error=mx))
    metric=lpips.LPIPS(net='alex').cuda().eval()
    perceptual=[]
    for name in ['classic-refine-s0-r1','mip-refine-s0','depth-refine-s0','mip-depth-refine-s0']:
        for c in [c for c in cams if c['is_test']]:
            a=image_tensor(ROOT/name/'evaluation'/f'{c["img_name"]}-s1.png')
            b=image_tensor(BASE/'hover-sfm-001/dataset/images'/f'{c["img_name"]}.jpg')
            aa=a.permute(2,0,1);bb=b.permute(2,0,1)
            perceptual.append(dict(run=name,frame=c['img_name'],ssim=float(ssim(aa,bb)),
                                  lpips=float(metric(aa[None]*2-1,bb[None]*2-1))))
    difix=[];d=ROOT/'difix-appearance-test-r1'
    for c in [c for c in cams if c['is_test']]:
        gt=image_tensor(d/f'{c["img_name"]}-gt.png').permute(2,0,1)
        row=dict(frame=c['img_name'])
        for tag in ['input','difix']:
            a=image_tensor(d/f'{c["img_name"]}-{tag}.png').permute(2,0,1)
            row[tag]=dict(ssim=float(ssim(a,gt)),lpips=float(metric(a[None]*2-1,gt[None]*2-1)))
        difix.append(row)
    summary={name:{key:float(np.mean([r[key] for r in perceptual if r['run']==name])) for key in ['ssim','lpips']} for name in sorted({r['run'] for r in perceptual})}
    ds={tag:{key:float(np.mean([r[tag][key] for r in difix])) for key in ['ssim','lpips']} for tag in ['input','difix']}
    result=dict(status='passed',original_ply_unchanged=True,filter_baking=roundtrip,perceptual_seed0=perceptual,
                perceptual_summary=summary,difix=difix,difix_summary=ds,
                scope='SSIM/LPIPS only seed0 here; PSNR repeat comparison separately contains all three seeds; no claim of bit-identical native rasterizers')
    write_json(ROOT/'delivery-verification.json',result);print(json.dumps(dict(perceptual=summary,difix=ds),indent=2))


if __name__=='__main__':main()
