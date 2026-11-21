"""Minimal mask-guided shared scalp offset control, using the original VHAP optimizer."""
import argparse, dataclasses, fcntl, os, time, types, traceback
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import fit_demo_flame_rgb123 as g
from run_official_gaussians123 import read, save, sha, now

PHOTO=g.OUT
OUT=PHOTO.parent/'hat-mask-control'
SOURCE=PHOTO/'epoch-30/tracked-flame-with-surface.npz'
STAGE='rgb_global_tracking'

def register():
    assert not OUT.exists();OUT.mkdir()
    rows=read(g.f.DATA/'data/inputs.json')['records']
    with np.load(g.f.DATA/'data/observations.npz',allow_pickle=False) as z:points=z['keypoints'][:,23:91].copy()
    records=[]
    for i,row in enumerate(rows):
        alpha=np.array(Image.open(row['mask']).convert('L'))
        # The original SAM2 target is unchanged. Restrict this head-only control
        # to the upper-head band, avoiding clothing, hands and held devices.
        lower=int(np.ceil(points[i,17:27,1].max()))
        y,x=np.nonzero((alpha>127)&(np.arange(alpha.shape[0])[:,None]<=lower))
        assert len(x)>0
        p=np.r_[points[i],np.array([[x.min(),y.min()],[x.max(),y.max()]])]
        lo=p.min(0);hi=p.max(0)
        records.append(dict(frame=row['frame'],upper_head_band_ymax=lower,
            bound_min=lo.tolist(),bound_max=hi.tolist(),mask_sha256=sha(row['mask'])))
    side=int(np.ceil(max(max(np.array(r['bound_max'])-r['bound_min']) for r in records)+40))
    assert side<=256, 'Unexpected SAM2 component: inspect target, do not crop it away.'
    folder=OUT/'mask-inputs';folder.mkdir()
    for row,r in zip(rows,records):
        center=(np.array(r['bound_min'])+r['bound_max'])/2
        xy=np.floor(center-side/2).astype(int);r['crop_integer_xy']=xy.tolist()
        box=(*xy,*(xy+side));raw=Image.open(row['image']).convert('RGB').crop(box)
        alpha=Image.open(row['mask']).convert('L').crop(box)
        target=np.array(alpha);roi=(np.arange(side)+xy[1])[:,None]<=r['upper_head_band_ymax']
        guide=np.where(roi,target,127).astype('uint8')
        im=Image.new('RGB',(3*side,side),'white')
        for j,p in enumerate([raw,alpha.convert('RGB'),Image.fromarray(guide).convert('RGB')]):im.paste(p,(j*side,0))
        im.save(folder/(row['frame']+'_原图_第一个SAM2mask_头帽轮廓监督.png'))
    save(OUT/'plan.json',dict(updated=now(),authorization='User: 主要是现在没有帽子，mesh上；用第一个mask做指导；检查通过就开始高斯训练。',
        source=str(SOURCE),source_sha256=sha(SOURCE),script_sha256=sha(__file__),
        target='First/source SAM2 mask, unchanged; corresponding target for every one of59 frames. Not the rendered mesh mask.',
        crop_side=side,crop_contract='Native integer crop with translated completeK, enlarged only to include the SAM2 upper-head target. No resize/warp.',
        records=records,scope='All59 frames jointly optimize one shared scalp offset. Shape, face/neck/jaw/eye poses, expressions, texture, lights and camera fixed.',
        minimum_changes=['Enable the upstream commented normalized absolute alpha loss using existing photo weight30.',
            'Remove hair from boundary detach list, retaining bottomline exclusion.',
            'Restrict offset gradient to upstream hair vertices and mask supervision to observed upper-head band.'],
        original_RGB_and_regularizers='Preserved; original Adam with existing static-offset LR and global0.1 scale.',
        num_epochs=17,max_updates=1003,max_train_seconds=180,
        risks='A scalp topology can approximate a hat envelope but may stretch/fold at the brim. A silhouette alone cannot determine hidden thickness or complete3D shape.',
        admission='Actual full59-frame masks, side views and shared-mesh review; preserve face pose/projection; reject new severe folding or foreground leakage. No automatic Gaussian launch from loss alone.',quality_accepted=False))
    print('REGISTERED_HAT_MASK',side,flush=True)

def setup():
    g.OUT=OUT;t,rows=g.setup();plan=read(OUT/'plan.json')
    with np.load(SOURCE,allow_pickle=False) as z,torch.no_grad():
        for name in ['shape',*g.f.PER_FRAME,'scene_scale','tex_extra','lights','static_offset']:
            getattr(t,name).copy_(torch.as_tensor(z[name],device='cuda'))
    from vhap.util.camera import convert_camera_convention
    side=plan['crop_side'];t.dataset.side=side;t.image_size=(side,side)
    for i,(row,r) in enumerate(zip(rows,plan['records'])):
        assert row['frame']==r['frame'] and sha(row['mask'])==r['mask_sha256']
        xy=np.array(r['crop_integer_xy']);t.dataset.crops[i]=xy
        rgb=torch.from_numpy(np.array(Image.open(row['image']).convert('RGB').crop((*xy,*(xy+side))),copy=True)).permute(2,0,1).float()/255
        alpha=torch.from_numpy(np.array(Image.open(row['mask']).convert('L').crop((*xy,*(xy+side))),copy=True))[None].float()/255
        K=np.array(row['K']);K[:2,2]-=xy
        c2w,glK=convert_camera_convention('opencv->opengl',np.eye(4)[None],K[None],side,side)
        s=t.dataset.samples[i];weights=s['lmk2d'][:,2].numpy()
        s.update(rgb=rgb*alpha+(1-alpha),alpha=alpha,alpha_map=alpha,
            intrinsic=torch.tensor(glK[0],dtype=torch.float32),extrinsic=torch.tensor(np.linalg.inv(c2w[0]),dtype=torch.float32),
            lmk2d=torch.tensor(np.c_[t.dataset.obs['keypoints'][i,23:91]-xy+.5,weights],dtype=torch.float32),
            silhouette_roi=torch.tensor(np.broadcast_to((np.arange(side)+xy[1])[:,None]<=r['upper_head_band_ymax'],(side,side)).copy())[None].float())
    stage=t.cfg.pipeline[STAGE];stage.optimizable_params=['static_offset'];stage.num_epochs=17
    stage.align_boundary_except=[x for x in stage.align_boundary_except if x!='hair']
    original=t.compute_energy
    def mask_energy(self,sample,step_i=None,stage=None):
        total,logs,*rest=original(sample,step_i=step_i,stage=stage)
        if stage is not None:
            render=rest[-1];target=sample['alpha_map'].cuda();roi=sample['silhouette_roi'].cuda()
            silhouette=((render['pred_alpha']-target).abs()*roi).sum()/roi.sum().clamp_min(1)
            logs['silhouette']=self.cfg.w.photo*silhouette
            total=total+logs['silhouette'];logs['total']=total
        return total,logs,*rest
    t.compute_energy=types.MethodType(mask_energy,t)
    t.get_train_parameters(STAGE)
    hair=t.flame.mask.get_vid_by_region(['hair'])
    gradient_mask=torch.zeros_like(t.static_offset);gradient_mask[:,hair]=1
    t.static_offset.register_hook(lambda grad:grad*gradient_mask)
    return t,rows,hair

def mask_review(t,rows):
    metrics=[];folder=OUT/'epoch-30'
    for i,row in enumerate(rows):
        with torch.no_grad():
            t.clear_cache();s=t.dataset.one(i);_,_,v,faces,l,tex,r=t.compute_energy(s,stage=None)
        target=s['alpha_map'][0,0].numpy()>.5;roi=s['silhouette_roi'][0,0].numpy()>0
        pred=r['pred_alpha'][0,0].detach().cpu().numpy()>.5
        metrics.append(dict(frame=row['frame'],iou_upper=float(((target&pred)&roi).sum()/max(((target|pred)&roi).sum(),1)),
            missing_upper=int((target&~pred&roi).sum()),outside_upper=int((pred&~target&roi).sum())))
        xy=t.dataset.crops[i];side=t.dataset.side
        raw=Image.open(row['image']).convert('RGB').crop((*xy,*(xy+side)))
        im=Image.new('RGB',(3*side,side),'white')
        for j,p in enumerate([raw,Image.fromarray(target.astype('uint8')*255).convert('RGB'),Image.fromarray(pred.astype('uint8')*255).convert('RGB')]):im.paste(p,(j*side,0))
        im.save(folder/(row['frame']+'_原图_SAM2mask_拟合mesh遮罩.png'))
    save(folder/'mask-review.json',dict(updated=now(),rows=metrics,all59=True,supervision='upper-head only; originals unchanged'))

def run(mode):
    plan=read(OUT/'plan.json');assert plan['script_sha256']==sha(__file__) and plan['source_sha256']==sha(SOURCE)
    marker=OUT/(mode+'-started.json');assert not marker.exists()
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(marker,dict(updated=now(),pid=os.getpid()));start=time.monotonic()
    try:
        t,rows,hair=setup();loss,logs,*_=t.compute_energy(t.dataset.one(32),stage=STAGE)
        grad=torch.autograd.grad(loss,t.static_offset)[0]
        assert torch.isfinite(grad).all() and float(grad[:,hair].abs().max())>0
        save(OUT/'preflight.json',dict(updated=now(),optimizer_updates=0,finite=True,mask_loss=float(logs['silhouette']),hair_gradient_max=float(grad[:,hair].abs().max()),vertices=len(hair)))
        if mode=='preflight':return
        frozen={name:getattr(t,name).detach().clone() for name in ['shape',*g.f.PER_FRAME,'scene_scale','tex_extra','lights']}
        original_offset=t.static_offset.detach().clone();nonhair=torch.ones(original_offset.shape[1],dtype=torch.bool,device='cuda');nonhair[hair]=False
        g.export(t,rows,'before');t.started=time.monotonic()
        optimize_iter=t.optimize_iter
        def bounded(*args,**kwargs):
            if time.monotonic()-t.started>180:raise RuntimeError('Mask control180s training budget reached')
            return optimize_iter(*args,**kwargs)
        t.optimize_iter=bounded
        t.optimize_stage(STAGE,dataloader=DataLoader(t.dataset,batch_size=1,shuffle=True,num_workers=0),lr_scale=.1)
        seconds=time.monotonic()-t.started
        assert t.global_step==1003 and seconds<180
        assert all(torch.equal(getattr(t,n),x) for n,x in frozen.items())
        assert torch.equal(t.static_offset[:,nonhair],original_offset[:,nonhair])
        g.export(t,rows,'epoch-30');mask_review(t,rows)
        save(OUT/'result.json',dict(updated=now(),state='completed_pending_visual_review',frames=59,steps=1003,train_seconds=seconds,
            elapsed_seconds=time.monotonic()-start,frozen_face_parameters_unchanged=True,nonhair_offsets_unchanged=True,Gaussian_started=False,quality_accepted=False))
        save(OUT/'progress.json',dict(updated=now(),state='completed_pending_visual_review',pid=None,step=1003))
    except BaseException as e:
        save(OUT/(mode+'-failure.json'),dict(updated=now(),error=repr(e),traceback=traceback.format_exc()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['register','preflight','fit']);args=p.parse_args()
    register() if args.mode=='register' else run(args.mode)
