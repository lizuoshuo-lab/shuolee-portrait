"""Controlled post-training study from the immutable Mini-Splatting2 PLY.

gsplat is a different rasterizer. Run probe first to quantify equivalence to
official saved renders. This is fixed-topology refinement, not a reproduction
of Mini-Splatting2 densification or a full from-scratch Mip-Splatting pipeline.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import random
import sys
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
import torch
import torch.nn.functional as F
from gsplat import rasterization

LAB=Path(__file__).resolve().parents[2]
BASE=LAB/'outputs/mini-splatting2-computer00-001'
ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'
PLY=BASE/'viewer/models/TRAINED_18000/point_cloud.ply'


def write_json(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def read_model(path=PLY):
    v=PlyData.read(str(path))['vertex']
    def cols(names):return np.stack([np.asarray(v[n]) for n in names],-1).copy()
    a={'means':cols(['x','y','z']),'quats':cols([f'rot_{i}' for i in range(4)]),
       'scales':cols([f'scale_{i}' for i in range(3)]),'opacities':np.asarray(v['opacity']).copy(),
       'sh0':cols([f'f_dc_{i}' for i in range(3)])[:,None,:],
       'shN':cols([f'f_rest_{i}' for i in range(45)]).reshape(-1,3,15).transpose(0,2,1).copy()}
    return torch.nn.ParameterDict({k:torch.nn.Parameter(torch.tensor(x,device='cuda')) for k,x in a.items()})


def cameras():
    raw=sorted(json.loads((BASE/'viewer/cameras.json').read_text()),key=lambda c:c['img_name'])
    for i,c in enumerate(raw):
        R=np.array(c['rotation']);C=np.array(c['position'])
        V=np.eye(4);V[:3,:3]=R.T;V[:3,3]=-R.T@C
        c['viewmat']=torch.tensor(V,dtype=torch.float32,device='cuda')[None]
        c['is_test']=(i%8==0)
    return raw


@torch.no_grad()
def mip_filter(model,cams):
    xyz=model['means'];dist=torch.full((len(xyz),),1e5,device='cuda');valid_all=torch.zeros_like(dist,dtype=torch.bool)
    focal=max(c['fx'] for c in cams)
    for c in cams:
        V=c['viewmat'][0];p=xyz@V[:3,:3].T+V[:3,3]
        z=p[:,2].clamp_min(.001)
        x=p[:,0]/z*c['fx']+c['width']/2;y=p[:,1]/z*c['fy']+c['height']/2
        valid=(p[:,2]>.2)&(x>=-.15*c['width'])&(x<=1.15*c['width'])&(y>=-.15*c['height'])&(y<=1.15*c['height'])
        dist[valid]=torch.minimum(dist[valid],z[valid]);valid_all|=valid
    if not valid_all.any():raise ValueError('No model points inside training cameras')
    dist[~valid_all]=dist[valid_all].max()
    return (dist/focal*math.sqrt(.2))[:,None]


def render(model,c,divisor=1,mode='classic',filt=None,cx_offset=0.,depth=False):
    w,h=round(c['width']/divisor),round(c['height']/divisor)
    sx,sy=w/c['width'],h/c['height']
    K=torch.tensor([[c['fx']*sx,0,(c['width']/2+cx_offset)*sx],
                    [0,c['fy']*sy,(c['height']/2+cx_offset)*sy],[0,0,1]],device='cuda',dtype=torch.float32)[None]
    scales=model['scales'].exp();opacities=model['opacities'].sigmoid()
    if filt is not None:
        sq=scales.square();smooth=sq+filt.square()
        opacities=opacities*torch.sqrt((sq/smooth).prod(-1))
        scales=smooth.sqrt()
    rgb,alpha,meta=rasterization(means=model['means'],quats=model['quats'],scales=scales,
        opacities=opacities,colors=torch.cat([model['sh0'],model['shN']],1),sh_degree=3,
        viewmats=c['viewmat'],Ks=K,width=w,height=h,packed=False,near_plane=.2,
        rasterize_mode='antialiased' if mode in ['mip','aa'] else 'classic',
        render_mode='RGB+ED' if depth else 'RGB',eps2d=.1 if mode=='mip' else .3)
    return rgb[0],alpha[0],meta


def image_tensor(path,w=None,h=None):
    im=Image.open(path).convert('RGB')
    if w is not None and im.size!=(w,h):
        import cv2
        arr=cv2.resize(np.asarray(im),(w,h),interpolation=cv2.INTER_AREA).copy()
    else:arr=np.asarray(im).copy()
    return torch.from_numpy(arr).to('cuda',dtype=torch.float32)/255


def save_image(path,img):
    Image.fromarray((img.detach().clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)).save(path)


def scores(a,b):
    diff=a-b
    return {'psnr':float(-10*torch.log10(diff.square().mean().clamp_min(1e-12))),
            'mae_255':float(diff.abs().mean()*255)}


def ssim(a,b):
    """Standard 11x11 Gaussian SSIM, sigma=1.5, zero padding, C1=.01^2/C2=.03^2."""
    t=torch.arange(-5,6,device=a.device,dtype=a.dtype)
    g=torch.exp(-t.square()/(2*1.5**2));g=g/g.sum()
    kernel=(g[:,None]*g[None,:])[None,None].expand(3,1,11,11).contiguous()
    def blur(x):return F.conv2d(x[None],kernel,padding=5,groups=3)[0]
    ma,mb=blur(a),blur(b)
    va,vb,cov=blur(a*a)-ma*ma,blur(b*b)-mb*mb,blur(a*b)-ma*mb
    return (((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))).mean()


@torch.no_grad()
def probe():
    out=ROOT/'renderer-probe';out.mkdir(exist_ok=True)
    model=read_model();cams=cameras();rows=[]
    for j,c in enumerate([c for c in cams if c['is_test']]):
        official=image_tensor(BASE/f'delivery/renders/{j:05d}.png')
        gt=image_tensor(BASE/f'delivery/gt/{j:05d}.png')
        for offset in [0.,-.5]:
            pred,_,_=render(model,c,cx_offset=offset)
            row=dict(frame=c['img_name'],cx_offset=offset,official_difference=scores(pred.clamp(0,1),official),
                     gt=scores(pred.clamp(0,1),gt))
            rows.append(row)
            if offset==0:save_image(out/f'{c["img_name"]}.png',pred)
    write_json(out/'results.json',{'rows':rows,'torch':torch.__version__,'gaussians':len(model['means']),
                                 'scope':'same PLY, same cameras, different CUDA rasterizer; no training'})
    print(json.dumps(rows,indent=2),flush=True)


def export_ply(path,model,filt=None):
    v=PlyData.read(str(PLY))['vertex'].data.copy()
    def assign(prefix,a):
        a=a.detach().cpu().numpy()
        for i in range(a.shape[1]):v[f'{prefix}_{i}']=a[:,i]
    for i,k in enumerate(['x','y','z']):v[k]=model['means'][:,i].detach().cpu().numpy()
    assign('rot',model['quats']);assign('f_dc',model['sh0'][:,0]);assign('f_rest',model['shN'].transpose(1,2).flatten(1))
    if filt is None:
        scales=model['scales'];opacity=model['opacities']
    else:
        sq=model['scales'].exp().square();filtered=sq+filt.square()
        scales=.5*filtered.log()
        opacity=torch.logit((model['opacities'].sigmoid()*torch.sqrt((sq/filtered).prod(-1))).clamp(1e-8,1-1e-8))
    assign('scale',scales);v['opacity']=opacity.detach().cpu().numpy()
    PlyData([PlyElement.describe(v,'vertex')]).write(str(path))


# Rectangles annotated from real held-out frames before viewing any new model.
# They contain background pixels; these are subject crops, NOT segmentation masks.
SUBJECT_BOX={0:(.44,.27,.75,.93),8:(.49,.28,.82,.98),16:(.53,.27,.91,.98),
             24:(.34,.26,.78,.95),32:(.29,.28,.68,.98),40:(.23,.26,.55,.99),
             48:(.22,.26,.51,.98),56:(.26,.26,.58,.98)}


@torch.no_grad()
def evaluate(model,cams,out,mode,filt):
    out.mkdir(exist_ok=True)
    rows=[]
    for c in [c for c in cams if c['is_test']]:
        for div in [1,2,4,8]:
            pred,alpha,_=render(model,c,divisor=div,mode=mode,filt=filt,depth=False)
            pred=pred.clamp(0,1);h,w=pred.shape[:2]
            gt=image_tensor(BASE/'hover-sfm-001/dataset/images'/f'{c["img_name"]}.jpg',w,h)
            box=SUBJECT_BOX[int(c['img_name'])];x0,y0,x1,y1=[round(v*s) for v,s in zip(box,[w,h,w,h])]
            row=dict(frame=c['img_name'],divisor=div,full=scores(pred,gt),
                     subject_crop=scores(pred[y0:y1,x0:x1],gt[y0:y1,x0:x1]),
                     mean_alpha=float(alpha.mean()))
            rows.append(row)
            save_image(out/f'{c["img_name"]}-s{div}.png',pred)
            if div==1:save_image(out/f'{c["img_name"]}-subject.png',pred[y0:y1,x0:x1])
    aggregate={str(d):{'full_psnr':float(np.mean([r['full']['psnr'] for r in rows if r['divisor']==d])),
                      'subject_crop_psnr':float(np.mean([r['subject_crop']['psnr'] for r in rows if r['divisor']==d]))} for d in [1,2,4,8]}
    write_json(out/'metrics.json',dict(rows=rows,aggregate=aggregate,subject_boxes=SUBJECT_BOX,
               scale_scope='area downsampled observed cameras, not physical camera motion',
               subject_scope='manual bounding rectangles including background; not person-mask metric'))
    return aggregate


def train(args):
    # Reusing an output directory silently would mix or overwrite experiments.
    out=ROOT/args.name;out.mkdir(exist_ok=False)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    torch.set_num_threads(4)
    model=read_model();cams=cameras();training=[c for c in cams if not c['is_test']]
    filt=mip_filter(model,training) if args.mode=='mip' else None
    lr={'means':1.6e-6*5.,'scales':.0005,'quats':.0001,'opacities':.0025,'sh0':.00025,'shN':.00025/20}
    optimizer=torch.optim.Adam([dict(params=[v],lr=lr[k],name=k) for k,v in model.items()],eps=1e-15)
    data=[]
    for c in training:
        im=image_tensor(BASE/'hover-sfm-001/dataset/images'/f'{c["img_name"]}.jpg').cpu()
        depth=None
        if args.depth_weight:
            p=ROOT/'depth-priors'/f'{c["img_name"]}.npz'
            with np.load(p) as n:
                depth=(torch.from_numpy(n['inverse_depth'].astype('float32')),torch.from_numpy(n['weight'].astype('float32')))
        data.append((c,im,depth))
    config=dict(action='fixed-topology post-training refinement',initial_ply_sha256=hashlib.sha256(PLY.read_bytes()).hexdigest(),
                mode=args.mode,steps=args.steps,seed=args.seed,depth_weight=args.depth_weight,
                gaussian_count=len(model['means']),learning_rates=lr,train_views=51,test_views=8,
                loss='0.8 L1 + 0.2 (1-SSIM) + optional confidence-weighted inverse expected z-depth L1 / scene inverse depth median',
                training_resolution=[757,1347],mip_3d_sigma='min visible training z / max training focal * sqrt(0.2)',
                mip_2d_variance=.1,classic_2d_variance=.3,filter_refresh_steps=100,
                topology='fixed; no clone/split/prune; Mini-Splatting2 native training/densification not rerun',
                holdout_usage='never used in optimizer, depth preparation or 3D filter; all cameras originally used for SfM',
                torch=torch.__version__)
    write_json(out/'config.json',config)
    write_json(out/'status.json',dict(status='running',step=0))
    start=time.time();order=[]
    with (out/'train.jsonl').open('w',buffering=1) as log:
        for step in range(1,args.steps+1):
            if not order:order=random.sample(range(len(data)),len(data))
            c,im,dp=data[order.pop()];im=im.to('cuda')
            if filt is not None and step%100==0:filt=mip_filter(model,training)
            pred,alpha,_=render(model,c,mode=args.mode,filt=filt,depth=dp is not None)
            rgb=pred[...,:3];l1=(rgb-im).abs().mean()
            sl=1-ssim(rgb.permute(2,0,1),im.permute(2,0,1))
            loss=.8*l1+.2*sl;dl=torch.zeros((),device='cuda')
            if dp is not None:
                inv,weight=[v.to('cuda') for v in dp]
                weight=weight*(alpha[...,0].detach()>.95)
                pred_inv=pred[...,3].clamp_min(.2).reciprocal()
                normalizer=inv[weight>0].median() if (weight>0).any() else torch.ones((),device='cuda')
                dl=((pred_inv-inv).abs()*weight).sum()/weight.sum().clamp_min(1)/normalizer.clamp_min(1e-5)
                loss=loss+args.depth_weight*dl
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss at {step}')
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            if step==1 or step%100==0:
                torch.cuda.synchronize()
                record=dict(step=step,loss=float(loss),l1=float(l1),depth_loss=float(dl),elapsed_seconds=time.time()-start,
                            peak_allocated_gb=torch.cuda.max_memory_allocated()/2**30)
                log.write(json.dumps(record)+'\n');write_json(out/'status.json',dict(status='running',**record));print(args.name,record,flush=True)
    if filt is not None:filt=mip_filter(model,training)
    torch.save(dict(params={k:v.detach().cpu() for k,v in model.items()},filter_3d=None if filt is None else filt.cpu(),config=config),out/'checkpoint.pt')
    export_ply(out/'point_cloud.ply',model,filt)
    aggregate=evaluate(model,cams,out/'evaluation',args.mode,filt)
    write_json(out/'status.json',dict(status='complete',steps=args.steps,elapsed_seconds=time.time()-start,aggregate=aggregate,
                                   peak_allocated_gb=torch.cuda.max_memory_allocated()/2**30))
    print(json.dumps(aggregate),flush=True)


@torch.no_grad()
def render_only():
    model=read_model();cams=cameras();filt=mip_filter(model,[c for c in cams if not c['is_test']])
    for mode in ['classic','aa','mip']:
        out=ROOT/f'render-only-{mode}';out.mkdir(exist_ok=False)
        result=evaluate(model,cams,out,mode,filt if mode=='mip' else None)
        print(mode,json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['probe','render-only','train'])
    p.add_argument('--name');p.add_argument('--mode',choices=['classic','aa','mip'],default='classic')
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--depth-weight',type=float,default=0.)
    args=p.parse_args()
    if args.action=='probe':probe()
    elif args.action=='render-only':render_only()
    elif not args.name:p.error('--name is required for training')
    else:train(args)
