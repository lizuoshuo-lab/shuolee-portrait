"""Gate existing Difix image corrections using real training-view support.

This is a controlled follow-up to the saved first-stage image-repair experiment,
not a newly trained 3D model. Held-out images enter metrics only. The fixed masks
are not tuned on those metrics. No generated pixels enter Gaussian training.
"""
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch
from local_refinement import BASE,ROOT,cameras,render,ssim
from evaluate_geometry import restore

OUT=ROOT/'difix-training-support-gating-001'
SOURCE=ROOT/'difix-appearance-test-r1'

def read(path):return np.asarray(Image.open(path).convert('RGB').resize((576,1024),Image.Resampling.LANCZOS)).astype('float32')/255
def save(path,array):Image.fromarray((np.clip(array,0,1)*255).round().astype('uint8')).save(path)

def warp(target,source,z_target,z_source,alpha_target,alpha_source,source_values):
    h,w=z_target.shape;yy,xx=np.mgrid[:h,:w]
    xyz=np.stack([(xx-target['width']/2)/target['fx']*z_target,
                  (yy-target['height']/2)/target['fy']*z_target,z_target],-1).reshape(-1,3)
    v=target['viewmat'][0].cpu().numpy();s=source['viewmat'][0].cpu().numpy()
    world=(xyz-v[:3,3])@v[:3,:3];other=world@s[:3,:3].T+s[:3,3]
    zs=other[:,2].reshape(h,w)
    u=(other[:,0]/np.maximum(other[:,2],1e-6)*source['fx']+source['width']/2).reshape(h,w).astype('float32')
    vv=(other[:,1]/np.maximum(other[:,2],1e-6)*source['fy']+source['height']/2).reshape(h,w).astype('float32')
    def sample(arr):return cv2.remap(arr,u,vv,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    observed=sample(z_source);ap=sample(alpha_source)
    valid=(zs>.2)&(u>2)&(u<w-3)&(vv>2)&(vv<h-3)&(alpha_target>.95)&(ap>.95)&(np.abs(observed-zs)/np.maximum(zs,.2)<.05)
    return sample(source_values),valid,(u,vv)


@torch.no_grad()
def main():
    OUT.mkdir(exist_ok=False);cv2.setNumThreads(4);torch.set_num_threads(4)
    protocol=dict(started=time.time(),input_model='mip-depth-refine-s0',diffusion_source=str(SOURCE),
        no_new_diffusion_inference=True,geometry_changed=False,generated_pixels_in_training=False,
        image_size=[576,1024],required_training_references=2,depth_agreement=.05,minimum_alpha=.95,
        reference_color_agreement=.05,input_error_gate_start=.03,input_error_gate_width=.06,
        required_error_reduction=.005,maximum_mean_color_change=.15,neighbor_delta_tolerance=.04,
        references='two nearest real training-camera photos; moving background pixels excluded',
        heldout_usage='eight evaluation photos are loaded only for final scoring, never masks or selection',
        caveat='reprojection uses estimated geometry and cannot certify geometry correctness; same-video development evidence')
    protocol['source_manifest_sha256']=hashlib.sha256((SOURCE/'manifest.json').read_bytes()).hexdigest()
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    model,mode,filt=restore('mip-depth-refine-s0');cams=cameras();training=[c for c in cams if not c['is_test']];tests=[c for c in cams if c['is_test']]
    small={}
    for c in cams:
        s=dict(c);s.update(width=576,height=1024,fx=c['fx']*576/c['width'],fy=c['fy']*1024/c['height']);small[c['img_name']]=s
    refs={}
    for c in tests:
        r=np.array(c['rotation']);p=np.array(c['position'])
        def distance(other):return np.arccos(np.clip((np.trace(r.T@np.array(other['rotation']))-1)/2,-1,1))+np.linalg.norm(p-np.array(other['position']))/5
        refs[c['img_name']]=[x['img_name'] for x in sorted(training,key=distance)[:2]]
    protocol['selected_references']=refs;(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    needed={c['img_name'] for c in tests}|{n for names in refs.values() for n in names};maps={}
    for name in sorted(needed):
        pred,alpha,_=render(model,small[name],mode=mode,filt=filt,depth=True)
        maps[name]=(pred[...,3].cpu().numpy(),alpha[...,0].cpu().numpy())
    del model;torch.cuda.empty_cache()
    data={};mask_reports={}
    for c in tests:
        name=c['img_name'];inp=read(SOURCE/f'{name}-input.png');dif=read(SOURCE/f'{name}-difix.png')
        z,alpha=maps[name];colors=[];validity=[]
        for ref in refs[name]:
            rgb=read(BASE/'hover-sfm-001/dataset/images'/f'{ref}.jpg')
            sampled,valid,grid=warp(small[name],small[ref],z,maps[ref][0],alpha,maps[ref][1],rgb)
            motion=np.asarray(Image.open(ROOT/'motion-masks-001'/f'{ref}.png').convert('L').resize((576,1024),Image.Resampling.NEAREST)).astype('float32')
            moved=cv2.remap(motion,*grid,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=255)
            valid&=moved==0;colors.append(sampled);validity.append(valid)
        reference=(colors[0]+colors[1])*.5
        reliable=validity[0]&validity[1]&(np.mean(np.abs(colors[0]-colors[1]),-1)<.05)
        before=np.mean(np.abs(inp-reference),-1);after=np.mean(np.abs(dif-reference),-1)
        change=np.mean(np.abs(dif-inp),-1)
        valid=reliable&(after+.005<before)&(change<.15)
        gate=np.clip((before-.03)/.06,0,1)*valid
        gate=cv2.GaussianBlur(gate.astype('float32'),(7,7),1)*valid
        data[name]=dict(input=inp,difix=dif,delta=dif-inp,gate=gate)
        mask_reports[name]=dict(references=refs[name],two_reference_support=float(reliable.mean()),initial_gate_fraction=float((gate>0).mean()))
    # Test-view-generated deltas can reject inconsistent corrections, but real
    # test images are not used. At least one neighbor must support the delta.
    for i,c in enumerate(tests):
        name=c['img_name'];item=data[name];consistent=np.zeros((1024,576),dtype=bool)
        for j in [i-1,i+1]:
            if j<0 or j>=len(tests):continue
            other=tests[j]['img_name'];sampled,valid,_=warp(small[name],small[other],maps[name][0],maps[other][0],maps[name][1],maps[other][1],data[other]['delta'])
            consistent|=valid&(np.mean(np.abs(item['delta']-sampled),-1)<.04)
        gate=item['gate']*consistent;item['final_gate']=gate;item['gated']=item['input']+gate[:,:,None]*item['delta']
        mask_reports[name]['final_gate_fraction']=float((gate>0).mean())
        save(OUT/f'{name}-gated.png',item['gated']);save(OUT/f'{name}-gate.png',gate)
    # Freeze every gate before reading scoring images.
    (OUT/'masks-frozen-before-scoring.json').write_text(json.dumps(dict(frozen_at=time.time(),frames=mask_reports,
        sha256={c['img_name']:hashlib.sha256((OUT/f'{c["img_name"]}-gate.png').read_bytes()).hexdigest() for c in tests}),indent=2))
    import lpips
    metric=lpips.LPIPS(net='alex').cuda().eval();rows=[]
    for c in tests:
        name=c['img_name'];gt=read(SOURCE/f'{name}-gt.png');gt_tensor=torch.tensor(gt,device='cuda').permute(2,0,1)
        row=dict(frame=name,mask=mask_reports[name])
        for key in ['input','difix','gated']:
            arr=data[name][key];t=torch.tensor(arr,device='cuda').permute(2,0,1)
            row[key]=dict(psnr=float(-10*np.log10(max(float(np.mean((arr-gt)**2)),1e-12))),
                ssim=float(ssim(t,gt_tensor)),lpips_alex=float(metric(t[None]*2-1,gt_tensor[None]*2-1)))
        rows.append(row)
    summary=dict(status='complete',protocol=protocol,rows=rows,
        aggregate={key:{metric:float(np.mean([r[key][metric] for r in rows])) for metric in ['psnr','ssim','lpips_alex']} for key in ['input','difix','gated']},
        mean_changed_pixel_fraction=float(np.mean([r['mask']['final_gate_fraction'] for r in rows])),
        geometry_changed=False,accepted_as_3d_training_supervision=False)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ['rows','protocol']}),flush=True)

if __name__=='__main__':main()
